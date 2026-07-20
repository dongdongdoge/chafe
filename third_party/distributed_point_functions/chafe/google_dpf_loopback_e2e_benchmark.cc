/*
 * Process-separated Chafe write-path benchmark over TCP loopback.
 *
 * The benchmark has two modes.  In server mode, a process receives one DPF
 * key per admitted report, expands and accumulates its packed64 share, and
 * returns the epoch state only after receiving an epoch-close marker.  In
 * client mode, a client/analyzer process generates the two keys, sends one to
 * each independently running server and waits for both acknowledgements.  At
 * epoch close, the servers open occupancies to one another, apply Drop or Pool,
 * pad and permute locally, and return two fixed-length release shares.  The
 * client-side analyzer reconstructs only those M releases and checks them
 * against the benchmark driver's inputs.
 *
 * This is intentionally a localhost experiment.  It measures real process,
 * socket, serialization, DPF, accumulation, and epoch-close work, but it is
 * neither a multi-node deployment nor a transport-security benchmark.
 */

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "dpf/distributed_point_function.h"
#include "openssl/rand.h"
#include "openssl/sha.h"
#include "pir/prng/aes_128_ctr_seeded_prng.h"

namespace {

using distributed_point_functions::DistributedPointFunction;
using distributed_point_functions::Aes128CtrSeededPrng;
using distributed_point_functions::DpfKey;
using distributed_point_functions::DpfParameters;
using distributed_point_functions::EvaluationContext;
using distributed_point_functions::ToValueType;

using Clock = std::chrono::steady_clock;

constexpr int kCountBits = 8;
constexpr uint64_t kCountMask = (uint64_t{1} << kCountBits) - 1;
constexpr uint64_t kPayloadMask = (uint64_t{1} << (64 - kCountBits)) - 1;
constexpr char kResponseMagic[8] = {'C', 'H', 'A', 'F', 'E', '0', '0', '1'};

struct Config {
  std::string mode = "client";
  std::string host = "127.0.0.1";
  std::string decoder = "drop";
  int port = 0;
  int server_id = -1;
  int peer_port = 0;
  int server0_port = 0;
  int server1_port = 0;
  int log_n = 12;
  int reports = 16;
  int release_m = 32;
  uint64_t seed = 1;
};

struct Inputs {
  std::vector<uint64_t> slots;
  std::vector<uint64_t> payloads;
};

struct ServerResponse {
  uint64_t report_count = 0;
  uint64_t key_bytes = 0;
  uint64_t context_ns = 0;
  uint64_t eval_ns = 0;
  uint64_t accumulate_ns = 0;
  uint64_t epoch_close_ns = 0;
  uint64_t peer_bytes_sent = 0;
  std::vector<uint64_t> release_share;
};

template <typename T>
T Unwrap(absl::StatusOr<T> value, const std::string& context) {
  if (!value.ok()) {
    std::cerr << context << ": " << value.status() << "\n";
    std::exit(2);
  }
  return std::move(*value);
}

std::string GenerateCryptoSeed() {
  std::string seed(Aes128CtrSeededPrng::SeedSize(), '\0');
  if (RAND_bytes(reinterpret_cast<uint8_t*>(seed.data()), seed.size()) != 1) {
    std::cerr << "BoringSSL RAND_bytes failed\n";
    std::exit(2);
  }
  return seed;
}

std::string DeriveCryptoSeed(const std::string& seed, const char* label) {
  SHA256_CTX context;
  SHA256_Init(&context);
  SHA256_Update(&context, label, std::strlen(label));
  SHA256_Update(&context, seed.data(), seed.size());
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256_Final(digest, &context);
  return std::string(reinterpret_cast<const char*>(digest),
                     Aes128CtrSeededPrng::SeedSize());
}

class AesCtrWordStream {
 public:
  explicit AesCtrWordStream(const std::string& seed)
      : prng_(Unwrap(Aes128CtrSeededPrng::Create(seed),
                     "Create AES-CTR release PRNG")) {}

  uint64_t Next() {
    if (offset_ + sizeof(uint64_t) > buffer_.size()) {
      buffer_ = prng_->GetRandomBytes(4096);
      offset_ = 0;
    }
    uint64_t value = 0;
    std::memcpy(&value, buffer_.data() + offset_, sizeof(value));
    offset_ += sizeof(value);
    return value;
  }

 private:
  std::unique_ptr<Aes128CtrSeededPrng> prng_;
  std::string buffer_;
  size_t offset_ = 0;
};

void CryptoShuffle(std::vector<uint64_t>* values, const std::string& seed) {
  AesCtrWordStream stream(seed);
  for (size_t remaining = values->size(); remaining > 1; --remaining) {
    const uint64_t bound = remaining;
    const uint64_t threshold = (uint64_t{0} - bound) % bound;
    uint64_t random = 0;
    do {
      random = stream.Next();
    } while (random < threshold);
    std::swap((*values)[remaining - 1], (*values)[random % bound]);
  }
}

bool StartsWith(const std::string& value, const std::string& prefix) {
  return value.rfind(prefix, 0) == 0;
}

uint64_t ParseUintFlag(const std::string& arg, const std::string& prefix) {
  return static_cast<uint64_t>(std::stoull(arg.substr(prefix.size())));
}

Config ParseArgs(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (StartsWith(arg, "--mode=")) {
      cfg.mode = arg.substr(std::string("--mode=").size());
    } else if (StartsWith(arg, "--host=")) {
      cfg.host = arg.substr(std::string("--host=").size());
    } else if (StartsWith(arg, "--decoder=")) {
      cfg.decoder = arg.substr(std::string("--decoder=").size());
    } else if (StartsWith(arg, "--port=")) {
      cfg.port = static_cast<int>(ParseUintFlag(arg, "--port="));
    } else if (StartsWith(arg, "--server_id=")) {
      cfg.server_id = static_cast<int>(ParseUintFlag(arg, "--server_id="));
    } else if (StartsWith(arg, "--peer_port=")) {
      cfg.peer_port = static_cast<int>(ParseUintFlag(arg, "--peer_port="));
    } else if (StartsWith(arg, "--server0_port=")) {
      cfg.server0_port = static_cast<int>(ParseUintFlag(arg, "--server0_port="));
    } else if (StartsWith(arg, "--server1_port=")) {
      cfg.server1_port = static_cast<int>(ParseUintFlag(arg, "--server1_port="));
    } else if (StartsWith(arg, "--log_n=")) {
      cfg.log_n = static_cast<int>(ParseUintFlag(arg, "--log_n="));
    } else if (StartsWith(arg, "--reports=")) {
      cfg.reports = static_cast<int>(ParseUintFlag(arg, "--reports="));
    } else if (StartsWith(arg, "--release_m=")) {
      cfg.release_m = static_cast<int>(ParseUintFlag(arg, "--release_m="));
    } else if (StartsWith(arg, "--seed=")) {
      cfg.seed = ParseUintFlag(arg, "--seed=");
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      std::exit(2);
    }
  }
  if (cfg.mode != "client" && cfg.mode != "server") {
    std::cerr << "--mode must be client or server\n";
    std::exit(2);
  }
  if (cfg.log_n < 1 || cfg.log_n > 28) {
    std::cerr << "--log_n must be in [1, 28]\n";
    std::exit(2);
  }
  if (cfg.mode == "client") {
    if (cfg.server0_port <= 0 || cfg.server1_port <= 0) {
      std::cerr << "client mode requires --server0_port and --server1_port\n";
      std::exit(2);
    }
    if (cfg.reports <= 0 || cfg.release_m < cfg.reports) {
      std::cerr << "--reports must be positive and --release_m must be >= reports\n";
      std::exit(2);
    }
    if (cfg.decoder != "drop" && cfg.decoder != "pool") {
      std::cerr << "--decoder must be drop or pool\n";
      std::exit(2);
    }
  } else {
    if (cfg.server_id != 0 && cfg.server_id != 1) {
      std::cerr << "server mode requires --server_id=0 or --server_id=1\n";
      std::exit(2);
    }
    if (cfg.server_id == 0 && cfg.peer_port <= 0) {
      std::cerr << "server 0 requires server 1's --peer_port\n";
      std::exit(2);
    }
    if (cfg.release_m <= 0 || (cfg.decoder != "drop" && cfg.decoder != "pool")) {
      std::cerr << "server mode requires a positive --release_m and Drop/Pool decoder\n";
      std::exit(2);
    }
  }
  return cfg;
}

std::unique_ptr<DistributedPointFunction> CreatePackedDpf(int log_n) {
  DpfParameters parameters;
  parameters.set_log_domain_size(log_n);
  *(parameters.mutable_value_type()) = ToValueType<uint64_t>();
  return Unwrap(DistributedPointFunction::Create(parameters), "Create packed64 DPF");
}

uint64_t PackWrite(uint64_t payload) {
  return (payload << kCountBits) | uint64_t{1};
}

Inputs GenerateInputs(const Config& cfg) {
  const uint64_t n = uint64_t{1} << cfg.log_n;
  std::mt19937_64 rng(cfg.seed);
  std::uniform_int_distribution<uint64_t> slot_dist(0, n - 1);
  std::uniform_int_distribution<uint64_t> payload_dist(1, (uint64_t{1} << 31) - 1);
  Inputs inputs;
  inputs.slots.resize(cfg.reports);
  inputs.payloads.resize(cfg.reports);
  for (int i = 0; i < cfg.reports; ++i) {
    inputs.slots[i] = slot_dist(rng);
    inputs.payloads[i] = payload_dist(rng);
  }
  return inputs;
}

void FailSystemCall(const std::string& operation) {
  std::cerr << operation << " failed: " << std::strerror(errno) << "\n";
  std::exit(2);
}

void SendAll(int fd, const void* data, size_t length) {
  const char* cursor = static_cast<const char*>(data);
  while (length > 0) {
    const ssize_t sent = send(fd, cursor, length, 0);
    if (sent < 0 && errno == EINTR) continue;
    if (sent <= 0) FailSystemCall("send");
    cursor += sent;
    length -= static_cast<size_t>(sent);
  }
}

void RecvAll(int fd, void* data, size_t length) {
  char* cursor = static_cast<char*>(data);
  while (length > 0) {
    const ssize_t received = recv(fd, cursor, length, 0);
    if (received < 0 && errno == EINTR) continue;
    if (received == 0) {
      std::cerr << "peer closed the connection early\n";
      std::exit(2);
    }
    if (received < 0) FailSystemCall("recv");
    cursor += received;
    length -= static_cast<size_t>(received);
  }
}

void SendU32(int fd, uint32_t value) {
  const uint32_t wire = htonl(value);
  SendAll(fd, &wire, sizeof(wire));
}

uint32_t RecvU32(int fd) {
  uint32_t wire = 0;
  RecvAll(fd, &wire, sizeof(wire));
  return ntohl(wire);
}

void SendU64(int fd, uint64_t value) {
  unsigned char bytes[8];
  for (int i = 0; i < 8; ++i) bytes[7 - i] = static_cast<unsigned char>(value >> (8 * i));
  SendAll(fd, bytes, sizeof(bytes));
}

uint64_t RecvU64(int fd) {
  unsigned char bytes[8];
  RecvAll(fd, bytes, sizeof(bytes));
  uint64_t value = 0;
  for (unsigned char byte : bytes) value = (value << 8) | byte;
  return value;
}

int ListenLoopback(int requested_port, int* actual_port) {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) FailSystemCall("socket");
  int reuse = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
    FailSystemCall("setsockopt");
  }
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = htons(static_cast<uint16_t>(requested_port));
  if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
    FailSystemCall("bind");
  }
  if (listen(fd, 1) < 0) FailSystemCall("listen");
  socklen_t length = sizeof(address);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) < 0) {
    FailSystemCall("getsockname");
  }
  *actual_port = ntohs(address.sin_port);
  return fd;
}

int Connect(const std::string& host, int port) {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) FailSystemCall("socket");
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<uint16_t>(port));
  if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
    std::cerr << "invalid IPv4 address: " << host << "\n";
    std::exit(2);
  }
  if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
    FailSystemCall("connect");
  }
  return fd;
}

uint64_t NsBetween(Clock::time_point start, Clock::time_point end) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

void RunServer(const Config& cfg) {
  const size_t n = size_t{1} << cfg.log_n;
  int actual_port = 0;
  const int listener = ListenLoopback(cfg.port, &actual_port);
  int actual_peer_port = 0;
  const int peer_listener =
      cfg.server_id == 1 ? ListenLoopback(0, &actual_peer_port) : -1;
  std::cout << "READY," << actual_port << "," << actual_peer_port << "\n"
            << std::flush;
  const int client = accept(listener, nullptr, nullptr);
  if (client < 0) FailSystemCall("accept");
  close(listener);

  auto dpf = CreatePackedDpf(cfg.log_n);
  std::vector<uint64_t> packed(n, 0);
  uint64_t reports = 0;
  uint64_t key_bytes = 0;
  uint64_t context_ns = 0;
  uint64_t eval_ns = 0;
  uint64_t accumulate_ns = 0;
  while (true) {
    const uint32_t key_length = RecvU32(client);
    if (key_length == 0) break;
    std::string serialized(key_length, '\0');
    RecvAll(client, serialized.data(), serialized.size());
    DpfKey key;
    if (!key.ParseFromArray(serialized.data(), static_cast<int>(serialized.size()))) {
      std::cerr << "could not parse DPF key\n";
      std::exit(2);
    }
    key_bytes += key_length;

    const auto context_start = Clock::now();
    EvaluationContext context =
        Unwrap(dpf->CreateEvaluationContext(key), "Create server evaluation context");
    const auto context_end = Clock::now();
    auto share = Unwrap(dpf->EvaluateNext<uint64_t>({}, context), "Evaluate server key");
    const auto eval_end = Clock::now();
    for (size_t i = 0; i < n; ++i) packed[i] += share[i];
    const auto accumulate_end = Clock::now();
    context_ns += NsBetween(context_start, context_end);
    eval_ns += NsBetween(context_end, eval_end);
    accumulate_ns += NsBetween(eval_end, accumulate_end);
    ++reports;
    const unsigned char ack = 1;
    SendAll(client, &ack, sizeof(ack));
  }

  // Epoch close is the only server-to-server communication.  The servers open
  // the occupancy field to one another, derive the same keep predicate, and
  // independently build fixed-length release shares.  The benchmark driver
  // receives those M shares plus public instrumentation metadata, but never
  // either N-slot state array or a compaction map.
  const auto close_start = Clock::now();
  const int peer = cfg.server_id == 0
                       ? Connect(cfg.host, cfg.peer_port)
                       : accept(peer_listener, nullptr, nullptr);
  if (peer < 0) FailSystemCall("accept peer");
  if (peer_listener >= 0) close(peer_listener);
  std::vector<unsigned char> local_low(n);
  std::vector<unsigned char> peer_low(n);
  for (size_t i = 0; i < n; ++i) {
    local_low[i] = static_cast<unsigned char>(packed[i] & kCountMask);
  }
  const std::string local_dummy_seed = GenerateCryptoSeed();
  const std::string local_permutation_seed_share = GenerateCryptoSeed();
  std::string peer_permutation_seed_share(Aes128CtrSeededPrng::SeedSize(), '\0');
  if (cfg.server_id == 0) {
    SendAll(peer, local_low.data(), local_low.size());
    SendAll(peer, local_permutation_seed_share.data(),
            local_permutation_seed_share.size());
    RecvAll(peer, peer_low.data(), peer_low.size());
    RecvAll(peer, peer_permutation_seed_share.data(),
            peer_permutation_seed_share.size());
  } else {
    RecvAll(peer, peer_low.data(), peer_low.size());
    RecvAll(peer, peer_permutation_seed_share.data(),
            peer_permutation_seed_share.size());
    SendAll(peer, local_low.data(), local_low.size());
    SendAll(peer, local_permutation_seed_share.data(),
            local_permutation_seed_share.size());
  }
  close(peer);

  std::vector<uint64_t> release_share;
  release_share.reserve(cfg.release_m);
  for (size_t i = 0; i < n; ++i) {
    const uint64_t low0 = cfg.server_id == 0 ? local_low[i] : peer_low[i];
    const uint64_t low1 = cfg.server_id == 0 ? peer_low[i] : local_low[i];
    const uint64_t low_sum = low0 + low1;
    const uint32_t occupancy = static_cast<uint32_t>(low_sum & kCountMask);
    const bool emit = cfg.decoder == "drop" ? occupancy == 1 : occupancy > 0;
    if (!emit) continue;
    uint64_t payload_share = (packed[i] >> kCountBits) & kPayloadMask;
    if (cfg.server_id == 0) {
      payload_share = (payload_share + (low_sum >> kCountBits)) & kPayloadMask;
    }
    release_share.push_back(payload_share);
  }
  if (release_share.size() > static_cast<size_t>(cfg.release_m)) {
    std::cerr << "fixed release length is smaller than the real output count\n";
    std::exit(3);
  }
  // Each server samples dummy shares without learning the other server's
  // shares; their reconstructed sum is uniform over Z_(2^56).  The XORed seed
  // shares give both servers the same permutation seed without sending that
  // seed to the benchmark driver.
  AesCtrWordStream dummy_share_rng(local_dummy_seed);
  while (release_share.size() < static_cast<size_t>(cfg.release_m)) {
    release_share.push_back(dummy_share_rng.Next() & kPayloadMask);
  }
  std::string joint_permutation_seed(Aes128CtrSeededPrng::SeedSize(), '\0');
  for (size_t i = 0; i < joint_permutation_seed.size(); ++i) {
    joint_permutation_seed[i] =
        local_permutation_seed_share[i] ^ peer_permutation_seed_share[i];
  }
  CryptoShuffle(&release_share, joint_permutation_seed);

  // Re-randomize the two release-share vectors with a zero-sharing mask.  This
  // preserves every reconstructed value while making the share decomposition
  // simulatable from the fixed-length release itself.  The mask key is derived
  // independently from the joint permutation key and is unknown to the driver.
  const std::string mask_seed =
      DeriveCryptoSeed(joint_permutation_seed, "chafe-release-mask-v1");
  AesCtrWordStream mask_rng(mask_seed);
  for (uint64_t& value : release_share) {
    const uint64_t mask = mask_rng.Next() & kPayloadMask;
    value = cfg.server_id == 0 ? (value + mask) & kPayloadMask
                               : (value - mask) & kPayloadMask;
  }
  const uint64_t epoch_close_ns = NsBetween(close_start, Clock::now());

  SendAll(client, kResponseMagic, sizeof(kResponseMagic));
  SendU64(client, reports);
  SendU64(client, key_bytes);
  SendU64(client, context_ns);
  SendU64(client, eval_ns);
  SendU64(client, accumulate_ns);
  SendU64(client, epoch_close_ns);
  SendU64(client, n + Aes128CtrSeededPrng::SeedSize());
  SendU64(client, release_share.size());
  SendAll(client, release_share.data(), release_share.size() * sizeof(uint64_t));
  close(client);
}

void SendKey(int fd, const DpfKey& key) {
  std::string serialized;
  if (!key.SerializeToString(&serialized)) {
    std::cerr << "could not serialize DPF key\n";
    std::exit(2);
  }
  if (serialized.size() > UINT32_MAX) {
    std::cerr << "serialized DPF key exceeds protocol limit\n";
    std::exit(2);
  }
  SendU32(fd, static_cast<uint32_t>(serialized.size()));
  SendAll(fd, serialized.data(), serialized.size());
}

void RecvAck(int fd) {
  unsigned char ack = 0;
  RecvAll(fd, &ack, sizeof(ack));
  if (ack != 1) {
    std::cerr << "invalid server acknowledgement\n";
    std::exit(2);
  }
}

ServerResponse RecvServerResponse(int fd) {
  char magic[sizeof(kResponseMagic)];
  RecvAll(fd, magic, sizeof(magic));
  if (std::memcmp(magic, kResponseMagic, sizeof(magic)) != 0) {
    std::cerr << "invalid server response magic\n";
    std::exit(2);
  }
  ServerResponse response;
  response.report_count = RecvU64(fd);
  response.key_bytes = RecvU64(fd);
  response.context_ns = RecvU64(fd);
  response.eval_ns = RecvU64(fd);
  response.accumulate_ns = RecvU64(fd);
  response.epoch_close_ns = RecvU64(fd);
  response.peer_bytes_sent = RecvU64(fd);
  const uint64_t release_m = RecvU64(fd);
  response.release_share.resize(release_m);
  RecvAll(fd, response.release_share.data(),
          response.release_share.size() * sizeof(uint64_t));
  return response;
}

double Mean(const std::vector<double>& values) {
  double total = 0.0;
  for (double value : values) total += value;
  return total / values.size();
}

double Quantile(const std::vector<double>& values, double q) {
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const double index = q * static_cast<double>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(index));
  const size_t hi = static_cast<size_t>(std::ceil(index));
  return sorted[lo] * (1.0 - (index - lo)) + sorted[hi] * (index - lo);
}

std::vector<uint64_t> ReconstructFixedRelease(const Config& cfg,
                                               const ServerResponse& server0,
                                               const ServerResponse& server1,
                                               uint64_t* checksum) {
  if (server0.release_share.size() != static_cast<size_t>(cfg.release_m) ||
      server1.release_share.size() != static_cast<size_t>(cfg.release_m)) {
    std::cerr << "server release share has wrong length\n";
    std::exit(3);
  }
  std::vector<uint64_t> reconstructed_release(cfg.release_m);
  *checksum = 0;
  for (size_t i = 0; i < reconstructed_release.size(); ++i) {
    reconstructed_release[i] =
        (server0.release_share[i] + server1.release_share[i]) & kPayloadMask;
    *checksum ^= reconstructed_release[i] ^ i;
  }
  return reconstructed_release;
}

void VerifyFixedRelease(const Config& cfg, const Inputs& inputs,
                        const std::vector<uint64_t>& reconstructed_release,
                        size_t* real_output_count) {
  const size_t n = size_t{1} << cfg.log_n;
  std::vector<uint32_t> expected_occ(n, 0);
  std::vector<uint64_t> expected_payload(n, 0);
  for (size_t i = 0; i < inputs.slots.size(); ++i) {
    ++expected_occ[inputs.slots[i]];
    expected_payload[inputs.slots[i]] += inputs.payloads[i];
  }
  std::vector<uint64_t> expected_real;
  expected_real.reserve(cfg.release_m);
  for (size_t i = 0; i < n; ++i) {
    if (expected_occ[i] > kCountMask) {
      std::cerr << "packed occupancy exceeded the admitted 8-bit range\n";
      std::exit(3);
    }
    const bool emit =
        cfg.decoder == "drop" ? expected_occ[i] == 1 : expected_occ[i] > 0;
    if (emit) expected_real.push_back(expected_payload[i] & kPayloadMask);
  }
  *real_output_count = expected_real.size();
  if (expected_real.size() > static_cast<size_t>(cfg.release_m)) {
    std::cerr << "fixed release length is smaller than the real output count\n";
    std::exit(3);
  }
  // The permutation and dummy values are deliberately hidden from the
  // analyzer.  The synthetic workload driver can nevertheless verify that the
  // reconstructed fixed release contains every expected real output (with
  // multiplicity).  Accidental equality with a 56-bit uniform dummy has
  // negligible probability; packed-state reconstruction itself is separately
  // exhaustively checked by google_dpf_write_benchmark.
  std::sort(expected_real.begin(), expected_real.end());
  std::vector<uint64_t> sorted_release = reconstructed_release;
  std::sort(sorted_release.begin(), sorted_release.end());
  size_t expected_index = 0;
  for (uint64_t value : sorted_release) {
    if (expected_index < expected_real.size() && value == expected_real[expected_index]) {
      ++expected_index;
    }
  }
  if (expected_index != expected_real.size()) {
    std::cerr << "fixed release omitted at least one expected real output\n";
    std::exit(3);
  }
}

void RunClient(const Config& cfg) {
  auto dpf = CreatePackedDpf(cfg.log_n);
  const Inputs inputs = GenerateInputs(cfg);
  const int server0 = Connect(cfg.host, cfg.server0_port);
  const int server1 = Connect(cfg.host, cfg.server1_port);
  std::vector<double> latencies_ms;
  latencies_ms.reserve(cfg.reports);
  uint64_t keygen_ns = 0;
  uint64_t serialized_key_bytes = 0;

  const auto ingest_start = Clock::now();
  for (int report = 0; report < cfg.reports; ++report) {
    const auto report_start = Clock::now();
    const auto keygen_start = report_start;
    auto keys = Unwrap(
        dpf->GenerateKeys(absl::uint128(inputs.slots[report]),
                          PackWrite(inputs.payloads[report])),
        "Generate packed64 keys");
    const auto keygen_end = Clock::now();
    keygen_ns += NsBetween(keygen_start, keygen_end);
    serialized_key_bytes += keys.first.ByteSizeLong() + keys.second.ByteSizeLong();
    SendKey(server0, keys.first);
    SendKey(server1, keys.second);
    RecvAck(server0);
    RecvAck(server1);
    const auto report_end = Clock::now();
    latencies_ms.push_back(static_cast<double>(NsBetween(report_start, report_end)) / 1e6);
  }
  const auto ingest_end = Clock::now();

  const auto close_start = Clock::now();
  SendU32(server0, 0);
  SendU32(server1, 0);
  ServerResponse response0 = RecvServerResponse(server0);
  ServerResponse response1 = RecvServerResponse(server1);
  const auto release_shares_received = Clock::now();
  close(server0);
  close(server1);
  if (response0.report_count != static_cast<uint64_t>(cfg.reports) ||
      response1.report_count != static_cast<uint64_t>(cfg.reports)) {
    std::cerr << "server report count mismatch\n";
    std::exit(3);
  }
  uint64_t checksum = 0;
  const std::vector<uint64_t> reconstructed_release =
      ReconstructFixedRelease(cfg, response0, response1, &checksum);
  const auto protocol_end = Clock::now();
  size_t real_output_count = 0;
  VerifyFixedRelease(cfg, inputs, reconstructed_release, &real_output_count);
  const auto verification_end = Clock::now();

  const double ingest_seconds =
      static_cast<double>(NsBetween(ingest_start, ingest_end)) / 1e9;
  const double epoch_close_ms =
      static_cast<double>(NsBetween(close_start, protocol_end)) / 1e6;
  const double close_to_release_shares_received_ms =
      static_cast<double>(NsBetween(close_start, release_shares_received)) / 1e6;
  const double analyzer_ms =
      static_cast<double>(NsBetween(release_shares_received, protocol_end)) / 1e6;
  const double verification_ms =
      static_cast<double>(NsBetween(protocol_end, verification_end)) / 1e6;
  const uint64_t framing_bytes = static_cast<uint64_t>(cfg.reports) * 8 + 8;
  const uint64_t ack_bytes = static_cast<uint64_t>(cfg.reports) * 2;
  const uint64_t response_metadata_bytes = 2 * (sizeof(kResponseMagic) + 8 * 8);
  const uint64_t response_release_bytes =
      2 * static_cast<uint64_t>(cfg.release_m) * sizeof(uint64_t);
  const uint64_t epoch_close_inter_server_bytes =
      response0.peer_bytes_sent + response1.peer_bytes_sent;

  std::cout
      << "deployment_scope,transport,roles,server_processes,online_inter_server_bytes,"
         "epoch_close_inter_server_bytes,driver_socket_payload,"
         "benchmark_metadata_excluded_from_driver_view,"
         "instrumentation_metadata_view,online_protocol,"
         "dummy_law,release_share_rerandomization,permutation_seed_recipients,"
         "occupancy_opening_recipients,"
         "N,log_n,m,M,rho,tau,decoder,reports_per_second,mean_write_latency_ms,"
         "p50_write_latency_ms,p95_write_latency_ms,keygen_ms_per_report,"
         "server0_context_ms_per_report,server1_context_ms_per_report,"
         "server0_eval_ms_per_report,server1_eval_ms_per_report,"
         "server0_accumulate_ms_per_report,server1_accumulate_ms_per_report,"
         "server0_epoch_close_internal_ms,server1_epoch_close_internal_ms,"
         "epoch_close_ms,close_to_release_shares_received_ms,"
         "analyzer_reconstruct_release_ms,verification_ms,verification_scope,"
         "serialized_dpf_key_bytes,client_to_server_application_bytes,"
         "server_to_driver_application_bytes,real_output_count,real_output_count_provenance,"
         "release_checksum,"
         "correctness,source_type,benchmark,timed_scope\n";
  std::cout << std::fixed << std::setprecision(9)
            << "loopback_local_processes,plaintext_tcp_ipv4_loopback,"
               "workload_generator_analyzer_driver_plus_two_server_processes,2,0,"
            << epoch_close_inter_server_bytes << ","
               "two_fixed_length_M_release_shares_plus_benchmark_metadata,false,"
               "combined_workload_generator_analyzer_driver,"
               "sequential_ack_no_pipelining,"
               "uniform_mod_2^56_aes128ctr_independent_server_shares,"
               "joint_aes128ctr_zero_share_mask,"
               "server0_and_server1_only,server0_and_server1_only,"
            << (uint64_t{1} << cfg.log_n) << "," << cfg.log_n << ","
            << cfg.reports << "," << cfg.release_m << ","
            << static_cast<double>(cfg.reports) / (uint64_t{1} << cfg.log_n) << ","
            << static_cast<double>(cfg.release_m) / cfg.reports << ","
            << cfg.decoder << "," << cfg.reports / ingest_seconds << ","
            << Mean(latencies_ms) << "," << Quantile(latencies_ms, 0.50) << ","
            << Quantile(latencies_ms, 0.95) << ","
            << static_cast<double>(keygen_ns) / cfg.reports / 1e6 << ","
            << static_cast<double>(response0.context_ns) / cfg.reports / 1e6 << ","
            << static_cast<double>(response1.context_ns) / cfg.reports / 1e6 << ","
            << static_cast<double>(response0.eval_ns) / cfg.reports / 1e6 << ","
            << static_cast<double>(response1.eval_ns) / cfg.reports / 1e6 << ","
            << static_cast<double>(response0.accumulate_ns) / cfg.reports / 1e6 << ","
            << static_cast<double>(response1.accumulate_ns) / cfg.reports / 1e6 << ","
            << static_cast<double>(response0.epoch_close_ns) / 1e6 << ","
            << static_cast<double>(response1.epoch_close_ns) / 1e6 << ","
            << epoch_close_ms << "," << close_to_release_shares_received_ms << ","
            << analyzer_ms << "," << verification_ms
            << ",excluded_from_protocol_and_epoch_close_timing,"
            << serialized_key_bytes << "," << serialized_key_bytes + framing_bytes << ","
            << ack_bytes + response_metadata_bytes + response_release_bytes << ","
            << real_output_count << ",offline_workload_ground_truth," << checksum
            << ",true,microbenchmark,"
               "google_dpf_two_server_tcp_loopback,"
               "keygen_send_evalall_accumulate_ack_server_occupancy_open_compact_pad_permute_release_share_transfer_analyzer_reconstruct;verification_excluded;not_multi_node"
            << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  const Config cfg = ParseArgs(argc, argv);
  if (cfg.mode == "server") {
    RunServer(cfg);
  } else {
    RunClient(cfg);
  }
  return 0;
}
