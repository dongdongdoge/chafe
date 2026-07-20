/*
 * Chafe Shuffle Google DPF write-backend benchmark.
 *
 * This binary is intentionally small: it exercises Google's
 * DistributedPointFunction implementation as the anonymous point-write backend
 * for Chafe's hidden slots. For every report it generates two DPF keys,
 * evaluates both server keys over the full slot domain, accumulates server
 * shares, and checks that reconstruction matches the requested hidden-slot
 * writes.
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "dpf/distributed_point_function.h"
#include "dpf/tuple.h"

namespace {

using distributed_point_functions::DistributedPointFunction;
using distributed_point_functions::DpfKey;
using distributed_point_functions::DpfParameters;
using distributed_point_functions::EvaluationContext;
using distributed_point_functions::ToValueType;
using distributed_point_functions::Tuple;

struct Config {
  int log_n = 14;
  int reports = 16;
  int repeats = 5;
  int threads = 1;
  uint64_t seed = 1;
  bool breakdown = false;
};

struct Inputs {
  std::vector<uint64_t> slots;
  std::vector<uint64_t> payloads;
};

struct TimingSummary {
  double mean_seconds_per_report = 0.0;
  double se_seconds_per_report = 0.0;
  double p50_seconds_per_report = 0.0;
  double p95_seconds_per_report = 0.0;
  uint64_t checksum = 0;
};

struct RunResult {
  uint64_t checksum = 0;
  double keygen_seconds = 0.0;
  double context_seconds = 0.0;
  double eval_seconds = 0.0;
  double accumulate_seconds = 0.0;
  uint64_t key_bytes = 0;
};

struct WorkerState {
  explicit WorkerState(size_t n)
      : occ0(n, 0), occ1(n, 0), pay0(n, 0), pay1(n, 0) {}

  std::vector<uint32_t> occ0;
  std::vector<uint32_t> occ1;
  std::vector<uint64_t> pay0;
  std::vector<uint64_t> pay1;
  RunResult result;
};

struct PackedWorkerState {
  explicit PackedWorkerState(size_t n) : packed0(n, 0), packed1(n, 0) {}

  std::vector<uint64_t> packed0;
  std::vector<uint64_t> packed1;
  RunResult result;
};

struct DetailedTimingSummary {
  TimingSummary total;
  TimingSummary keygen;
  TimingSummary context;
  TimingSummary eval;
  TimingSummary accumulate;
  double key_bytes_per_report = 0.0;
};

constexpr int kCountBits = 8;
constexpr uint64_t kCountMask = (uint64_t{1} << kCountBits) - 1;
constexpr uint64_t kPayloadMask = (uint64_t{1} << (64 - kCountBits)) - 1;

template <typename T>
T Unwrap(absl::StatusOr<T> value, const std::string& context) {
  if (!value.ok()) {
    std::cerr << context << ": " << value.status() << "\n";
    std::exit(2);
  }
  return std::move(*value);
}

void CheckStatus(absl::Status status, const std::string& context) {
  if (!status.ok()) {
    std::cerr << context << ": " << status << "\n";
    std::exit(2);
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
    std::string arg(argv[i]);
    if (StartsWith(arg, "--log_n=")) {
      cfg.log_n = static_cast<int>(ParseUintFlag(arg, "--log_n="));
    } else if (StartsWith(arg, "--reports=")) {
      cfg.reports = static_cast<int>(ParseUintFlag(arg, "--reports="));
    } else if (StartsWith(arg, "--repeats=")) {
      cfg.repeats = static_cast<int>(ParseUintFlag(arg, "--repeats="));
    } else if (StartsWith(arg, "--threads=")) {
      cfg.threads = static_cast<int>(ParseUintFlag(arg, "--threads="));
    } else if (StartsWith(arg, "--seed=")) {
      cfg.seed = ParseUintFlag(arg, "--seed=");
    } else if (arg == "--breakdown" || arg == "--breakdown=1" ||
               arg == "--breakdown=true") {
      cfg.breakdown = true;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      std::exit(2);
    }
  }
  if (cfg.log_n < 1 || cfg.log_n > 28) {
    std::cerr << "--log_n must be in [1, 28] for this benchmark\n";
    std::exit(2);
  }
  if (cfg.reports <= 0 || cfg.repeats <= 0) {
    std::cerr << "--reports and --repeats must be positive\n";
    std::exit(2);
  }
  if (cfg.threads <= 0) {
    std::cerr << "--threads must be positive\n";
    std::exit(2);
  }
  return cfg;
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

template <typename T>
double Mean(const std::vector<T>& values) {
  double total = 0.0;
  for (T value : values) {
    total += static_cast<double>(value);
  }
  return total / values.size();
}

template <typename T>
double StandardError(const std::vector<T>& values) {
  if (values.size() <= 1) {
    return 0.0;
  }
  const double mean = Mean(values);
  double ss = 0.0;
  for (T value : values) {
    const double diff = static_cast<double>(value) - mean;
    ss += diff * diff;
  }
  const double variance = ss / static_cast<double>(values.size() - 1);
  return std::sqrt(variance) / std::sqrt(static_cast<double>(values.size()));
}

template <typename T>
double Quantile(const std::vector<T>& values, double q) {
  if (values.empty()) {
    return 0.0;
  }
  std::vector<double> sorted;
  sorted.reserve(values.size());
  for (T value : values) {
    sorted.push_back(static_cast<double>(value));
  }
  std::sort(sorted.begin(), sorted.end());
  const double index = q * static_cast<double>(sorted.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(index));
  const size_t upper = static_cast<size_t>(std::ceil(index));
  const double weight = index - static_cast<double>(lower);
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

std::unique_ptr<DistributedPointFunction> CreateProductDpf(int log_n) {
  DpfParameters parameters;
  parameters.set_log_domain_size(log_n);
  *(parameters.mutable_value_type()) = ToValueType<Tuple<uint32_t, uint64_t>>();
  auto dpf = Unwrap(DistributedPointFunction::Create(parameters), "Create product DPF");
  CheckStatus(dpf->RegisterValueType<Tuple<uint32_t, uint64_t>>(),
              "Register product value type");
  return dpf;
}

template <typename T>
std::unique_ptr<DistributedPointFunction> CreateIntegerDpf(int log_n) {
  DpfParameters parameters;
  parameters.set_log_domain_size(log_n);
  *(parameters.mutable_value_type()) = ToValueType<T>();
  return Unwrap(DistributedPointFunction::Create(parameters), "Create integer DPF");
}

uint64_t Checksum(const std::vector<uint32_t>& occ, const std::vector<uint64_t>& pay) {
  uint64_t out = 0;
  for (size_t i = 0; i < occ.size(); ++i) {
    out ^= (static_cast<uint64_t>(occ[i]) << 32) ^ pay[i] ^ static_cast<uint64_t>(i);
  }
  return out;
}

uint64_t PackWrite(uint64_t payload) {
  return (payload << kCountBits) | uint64_t{1};
}

uint32_t PackedCount(uint64_t packed) {
  return static_cast<uint32_t>(packed & kCountMask);
}

uint64_t PackedPayload(uint64_t packed) {
  return packed >> kCountBits;
}

uint64_t ChecksumPacked(const std::vector<uint64_t>& packed) {
  uint64_t out = 0;
  for (size_t i = 0; i < packed.size(); ++i) {
    out ^= (static_cast<uint64_t>(PackedCount(packed[i])) << 32) ^
           PackedPayload(packed[i]) ^ static_cast<uint64_t>(i);
  }
  return out;
}

int WorkerCount(const Config& cfg) {
  return std::max(1, std::min(cfg.threads, cfg.reports));
}

void AddTiming(RunResult& dst, const RunResult& src) {
  dst.keygen_seconds += src.keygen_seconds;
  dst.context_seconds += src.context_seconds;
  dst.eval_seconds += src.eval_seconds;
  dst.accumulate_seconds += src.accumulate_seconds;
  dst.key_bytes += src.key_bytes;
}

const char* CpuAcceleration() {
#if defined(__AES__)
  return "aes_ni";
#elif defined(__ARM_FEATURE_CRYPTO)
  return "arm_crypto";
#else
  return "portable";
#endif
}

void Verify(const Inputs& inputs, const std::vector<uint32_t>& occ,
            const std::vector<uint64_t>& pay) {
  std::vector<uint32_t> expected_occ(occ.size(), 0);
  std::vector<uint64_t> expected_pay(pay.size(), 0);
  for (size_t i = 0; i < inputs.slots.size(); ++i) {
    const uint64_t slot = inputs.slots[i];
    expected_occ[slot] += 1;
    expected_pay[slot] += inputs.payloads[i];
  }
  for (size_t i = 0; i < occ.size(); ++i) {
    if (occ[i] != expected_occ[i] || pay[i] != expected_pay[i]) {
      std::cerr << "reconstruction mismatch at slot " << i
                << ": got (" << occ[i] << ", " << pay[i] << ") expected ("
                << expected_occ[i] << ", " << expected_pay[i] << ")\n";
      std::exit(3);
    }
  }
}

void VerifyPacked(const Inputs& inputs, const std::vector<uint64_t>& packed) {
  std::vector<uint32_t> expected_occ(packed.size(), 0);
  std::vector<uint64_t> expected_pay(packed.size(), 0);
  for (size_t i = 0; i < inputs.slots.size(); ++i) {
    const uint64_t slot = inputs.slots[i];
    expected_occ[slot] += 1;
    expected_pay[slot] += inputs.payloads[i];
  }
  for (size_t i = 0; i < packed.size(); ++i) {
    const uint32_t occ = PackedCount(packed[i]);
    const uint64_t pay = PackedPayload(packed[i]);
    if (occ != expected_occ[i] || pay != expected_pay[i]) {
      std::cerr << "packed reconstruction mismatch at slot " << i
                << ": got (" << occ << ", " << pay << ") expected ("
                << expected_occ[i] << ", " << expected_pay[i] << ")\n";
      std::exit(3);
    }
  }
}

void VerifyPackedShares(const Inputs& inputs,
                        const std::vector<uint64_t>& packed0,
                        const std::vector<uint64_t>& packed1) {
  if (packed0.size() != packed1.size()) {
    std::cerr << "packed share arrays have different lengths\n";
    std::exit(3);
  }
  std::vector<uint32_t> expected_occ(packed0.size(), 0);
  std::vector<uint64_t> expected_pay(packed0.size(), 0);
  for (size_t i = 0; i < inputs.slots.size(); ++i) {
    const uint64_t slot = inputs.slots[i];
    expected_occ[slot] += 1;
    expected_pay[slot] += inputs.payloads[i];
  }
  for (size_t i = 0; i < packed0.size(); ++i) {
    const uint64_t low0 = packed0[i] & kCountMask;
    const uint64_t low1 = packed1[i] & kCountMask;
    const uint64_t low_sum = low0 + low1;
    const uint32_t occ = static_cast<uint32_t>(low_sum & kCountMask);
    const uint64_t carry = low_sum >> kCountBits;
    const uint64_t payload_share0 = ((packed0[i] >> kCountBits) + carry) &
                                    kPayloadMask;
    const uint64_t payload_share1 = (packed1[i] >> kCountBits) & kPayloadMask;
    const uint64_t pay = (payload_share0 + payload_share1) & kPayloadMask;
    if (occ != expected_occ[i] || pay != expected_pay[i]) {
      std::cerr << "packed share extraction mismatch at slot " << i
                << ": got (" << occ << ", " << pay << ") expected ("
                << expected_occ[i] << ", " << expected_pay[i] << ")\n";
      std::exit(3);
    }
  }
}

void RunPackedRange(const Config& cfg, const Inputs& inputs, int begin, int end,
                    PackedWorkerState* state) {
  auto dpf = CreateIntegerDpf<uint64_t>(cfg.log_n);

  for (int report = begin; report < end; ++report) {
    const auto keygen_start = std::chrono::steady_clock::now();
    auto keys = Unwrap(
        dpf->GenerateKeys(absl::uint128(inputs.slots[report]),
                          PackWrite(inputs.payloads[report])),
        "Generate packed64 keys");
    const auto keygen_end = std::chrono::steady_clock::now();
    state->result.keygen_seconds +=
        std::chrono::duration<double>(keygen_end - keygen_start).count();
    state->result.key_bytes += keys.first.ByteSizeLong() + keys.second.ByteSizeLong();

    const auto context_start = std::chrono::steady_clock::now();
    EvaluationContext ctx0 =
        Unwrap(dpf->CreateEvaluationContext(keys.first), "Create packed64 ctx0");
    EvaluationContext ctx1 =
        Unwrap(dpf->CreateEvaluationContext(keys.second), "Create packed64 ctx1");
    const auto context_end = std::chrono::steady_clock::now();
    state->result.context_seconds +=
        std::chrono::duration<double>(context_end - context_start).count();

    const auto eval_start = std::chrono::steady_clock::now();
    auto share0 = Unwrap(dpf->EvaluateNext<uint64_t>({}, ctx0),
                         "Evaluate packed64 server 0");
    auto share1 = Unwrap(dpf->EvaluateNext<uint64_t>({}, ctx1),
                         "Evaluate packed64 server 1");
    const auto eval_end = std::chrono::steady_clock::now();
    state->result.eval_seconds += std::chrono::duration<double>(eval_end - eval_start).count();

    const auto accumulate_start = std::chrono::steady_clock::now();
    const size_t n = state->packed0.size();
    for (size_t i = 0; i < n; ++i) {
      state->packed0[i] += share0[i];
      state->packed1[i] += share1[i];
    }
    const auto accumulate_end = std::chrono::steady_clock::now();
    state->result.accumulate_seconds +=
        std::chrono::duration<double>(accumulate_end - accumulate_start).count();
  }
}

RunResult RunPackedOnce(const Config& cfg, const Inputs& inputs) {
  const size_t n = size_t{1} << cfg.log_n;
  const int workers = WorkerCount(cfg);
  std::vector<std::unique_ptr<PackedWorkerState>> states;
  states.reserve(workers);
  for (int worker = 0; worker < workers; ++worker) {
    states.push_back(std::make_unique<PackedWorkerState>(n));
  }

  if (workers == 1) {
    RunPackedRange(cfg, inputs, 0, cfg.reports, states[0].get());
  } else {
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (int worker = 0; worker < workers; ++worker) {
      const int begin = worker * cfg.reports / workers;
      const int end = (worker + 1) * cfg.reports / workers;
      pool.emplace_back(
          [&, worker, begin, end]() { RunPackedRange(cfg, inputs, begin, end, states[worker].get()); });
    }
    for (std::thread& thread : pool) {
      thread.join();
    }
  }

  std::vector<uint64_t> packed0(n, 0);
  std::vector<uint64_t> packed1(n, 0);
  std::vector<uint64_t> packed(n, 0);
  RunResult result;
  for (const auto& state : states) {
    AddTiming(result, state->result);
    for (size_t i = 0; i < n; ++i) {
      packed0[i] += state->packed0[i];
      packed1[i] += state->packed1[i];
    }
  }
  for (size_t i = 0; i < n; ++i) {
    packed[i] = packed0[i] + packed1[i];
  }
  VerifyPackedShares(inputs, packed0, packed1);
  VerifyPacked(inputs, packed);
  result.checksum = ChecksumPacked(packed);
  return result;
}

void RunProductRange(const Config& cfg, const Inputs& inputs, int begin, int end,
                     WorkerState* state) {
  auto dpf = CreateProductDpf(cfg.log_n);

  for (int report = begin; report < end; ++report) {
    const auto keygen_start = std::chrono::steady_clock::now();
    auto keys = Unwrap(
        dpf->GenerateKeys(absl::uint128(inputs.slots[report]),
                          Tuple<uint32_t, uint64_t>{1, inputs.payloads[report]}),
        "Generate product keys");
    const auto keygen_end = std::chrono::steady_clock::now();
    state->result.keygen_seconds +=
        std::chrono::duration<double>(keygen_end - keygen_start).count();
    state->result.key_bytes += keys.first.ByteSizeLong() + keys.second.ByteSizeLong();

    const auto context_start = std::chrono::steady_clock::now();
    EvaluationContext ctx0 =
        Unwrap(dpf->CreateEvaluationContext(keys.first), "Create product ctx0");
    EvaluationContext ctx1 =
        Unwrap(dpf->CreateEvaluationContext(keys.second), "Create product ctx1");
    const auto context_end = std::chrono::steady_clock::now();
    state->result.context_seconds +=
        std::chrono::duration<double>(context_end - context_start).count();

    const auto eval_start = std::chrono::steady_clock::now();
    auto share0 = Unwrap(
        dpf->EvaluateNext<Tuple<uint32_t, uint64_t>>({}, ctx0),
        "Evaluate product server 0");
    auto share1 = Unwrap(
        dpf->EvaluateNext<Tuple<uint32_t, uint64_t>>({}, ctx1),
        "Evaluate product server 1");
    const auto eval_end = std::chrono::steady_clock::now();
    state->result.eval_seconds += std::chrono::duration<double>(eval_end - eval_start).count();

    const auto accumulate_start = std::chrono::steady_clock::now();
    const size_t n = state->occ0.size();
    for (size_t i = 0; i < n; ++i) {
      state->occ0[i] += std::get<0>(share0[i].value());
      state->pay0[i] += std::get<1>(share0[i].value());
      state->occ1[i] += std::get<0>(share1[i].value());
      state->pay1[i] += std::get<1>(share1[i].value());
    }
    const auto accumulate_end = std::chrono::steady_clock::now();
    state->result.accumulate_seconds +=
        std::chrono::duration<double>(accumulate_end - accumulate_start).count();
  }
}

RunResult RunProductOnce(const Config& cfg, const Inputs& inputs) {
  const size_t n = size_t{1} << cfg.log_n;
  const int workers = WorkerCount(cfg);
  std::vector<std::unique_ptr<WorkerState>> states;
  states.reserve(workers);
  for (int worker = 0; worker < workers; ++worker) {
    states.push_back(std::make_unique<WorkerState>(n));
  }

  if (workers == 1) {
    RunProductRange(cfg, inputs, 0, cfg.reports, states[0].get());
  } else {
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (int worker = 0; worker < workers; ++worker) {
      const int begin = worker * cfg.reports / workers;
      const int end = (worker + 1) * cfg.reports / workers;
      pool.emplace_back(
          [&, worker, begin, end]() { RunProductRange(cfg, inputs, begin, end, states[worker].get()); });
    }
    for (std::thread& thread : pool) {
      thread.join();
    }
  }

  std::vector<uint32_t> occ(n, 0);
  std::vector<uint64_t> pay(n, 0);
  RunResult result;
  for (const auto& state : states) {
    AddTiming(result, state->result);
    for (size_t i = 0; i < n; ++i) {
      occ[i] += state->occ0[i] + state->occ1[i];
      pay[i] += state->pay0[i] + state->pay1[i];
    }
  }
  Verify(inputs, occ, pay);
  result.checksum = Checksum(occ, pay);
  return result;
}

void RunDualRange(const Config& cfg, const Inputs& inputs, int begin, int end,
                  WorkerState* state) {
  auto occ_dpf = CreateIntegerDpf<uint32_t>(cfg.log_n);
  auto pay_dpf = CreateIntegerDpf<uint64_t>(cfg.log_n);

  for (int report = begin; report < end; ++report) {
    const auto keygen_start = std::chrono::steady_clock::now();
    auto occ_keys = Unwrap(
        occ_dpf->GenerateKeys(absl::uint128(inputs.slots[report]), uint32_t{1}),
        "Generate occupancy keys");
    auto pay_keys = Unwrap(
        pay_dpf->GenerateKeys(absl::uint128(inputs.slots[report]), inputs.payloads[report]),
        "Generate payload keys");
    const auto keygen_end = std::chrono::steady_clock::now();
    state->result.keygen_seconds +=
        std::chrono::duration<double>(keygen_end - keygen_start).count();
    state->result.key_bytes += occ_keys.first.ByteSizeLong() + occ_keys.second.ByteSizeLong() +
                               pay_keys.first.ByteSizeLong() + pay_keys.second.ByteSizeLong();

    const auto context_start = std::chrono::steady_clock::now();
    EvaluationContext occ_ctx0 =
        Unwrap(occ_dpf->CreateEvaluationContext(occ_keys.first), "Create occupancy ctx0");
    EvaluationContext occ_ctx1 =
        Unwrap(occ_dpf->CreateEvaluationContext(occ_keys.second), "Create occupancy ctx1");
    EvaluationContext pay_ctx0 =
        Unwrap(pay_dpf->CreateEvaluationContext(pay_keys.first), "Create payload ctx0");
    EvaluationContext pay_ctx1 =
        Unwrap(pay_dpf->CreateEvaluationContext(pay_keys.second), "Create payload ctx1");
    const auto context_end = std::chrono::steady_clock::now();
    state->result.context_seconds +=
        std::chrono::duration<double>(context_end - context_start).count();

    const auto eval_start = std::chrono::steady_clock::now();
    auto occ_share0 = Unwrap(occ_dpf->EvaluateNext<uint32_t>({}, occ_ctx0),
                             "Evaluate occupancy server 0");
    auto occ_share1 = Unwrap(occ_dpf->EvaluateNext<uint32_t>({}, occ_ctx1),
                             "Evaluate occupancy server 1");
    auto pay_share0 = Unwrap(pay_dpf->EvaluateNext<uint64_t>({}, pay_ctx0),
                             "Evaluate payload server 0");
    auto pay_share1 = Unwrap(pay_dpf->EvaluateNext<uint64_t>({}, pay_ctx1),
                             "Evaluate payload server 1");
    const auto eval_end = std::chrono::steady_clock::now();
    state->result.eval_seconds += std::chrono::duration<double>(eval_end - eval_start).count();

    const auto accumulate_start = std::chrono::steady_clock::now();
    const size_t n = state->occ0.size();
    for (size_t i = 0; i < n; ++i) {
      state->occ0[i] += occ_share0[i];
      state->occ1[i] += occ_share1[i];
      state->pay0[i] += pay_share0[i];
      state->pay1[i] += pay_share1[i];
    }
    const auto accumulate_end = std::chrono::steady_clock::now();
    state->result.accumulate_seconds +=
        std::chrono::duration<double>(accumulate_end - accumulate_start).count();
  }
}

RunResult RunDualOnce(const Config& cfg, const Inputs& inputs) {
  const size_t n = size_t{1} << cfg.log_n;
  const int workers = WorkerCount(cfg);
  std::vector<std::unique_ptr<WorkerState>> states;
  states.reserve(workers);
  for (int worker = 0; worker < workers; ++worker) {
    states.push_back(std::make_unique<WorkerState>(n));
  }

  if (workers == 1) {
    RunDualRange(cfg, inputs, 0, cfg.reports, states[0].get());
  } else {
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (int worker = 0; worker < workers; ++worker) {
      const int begin = worker * cfg.reports / workers;
      const int end = (worker + 1) * cfg.reports / workers;
      pool.emplace_back(
          [&, worker, begin, end]() { RunDualRange(cfg, inputs, begin, end, states[worker].get()); });
    }
    for (std::thread& thread : pool) {
      thread.join();
    }
  }

  std::vector<uint32_t> occ(n, 0);
  std::vector<uint64_t> pay(n, 0);
  RunResult result;
  for (const auto& state : states) {
    AddTiming(result, state->result);
    for (size_t i = 0; i < n; ++i) {
      occ[i] += state->occ0[i] + state->occ1[i];
      pay[i] += state->pay0[i] + state->pay1[i];
    }
  }
  Verify(inputs, occ, pay);
  result.checksum = Checksum(occ, pay);
  return result;
}

template <typename Fn>
DetailedTimingSummary TimeMode(const Config& cfg, Fn fn) {
  std::vector<double> seconds_per_report;
  std::vector<double> keygen_seconds_per_report;
  std::vector<double> context_seconds_per_report;
  std::vector<double> eval_seconds_per_report;
  std::vector<double> accumulate_seconds_per_report;
  std::vector<double> key_bytes_per_report;
  seconds_per_report.reserve(cfg.repeats);
  uint64_t checksum = 0;
  for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
    Config input_cfg = cfg;
    input_cfg.seed = cfg.seed + static_cast<uint64_t>(repeat);
    Inputs inputs = GenerateInputs(input_cfg);
    const auto start = std::chrono::steady_clock::now();
    RunResult run = fn(cfg, inputs);
    checksum ^= run.checksum;
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    seconds_per_report.push_back(elapsed.count() / cfg.reports);
    keygen_seconds_per_report.push_back(run.keygen_seconds / cfg.reports);
    context_seconds_per_report.push_back(run.context_seconds / cfg.reports);
    eval_seconds_per_report.push_back(run.eval_seconds / cfg.reports);
    accumulate_seconds_per_report.push_back(run.accumulate_seconds / cfg.reports);
    key_bytes_per_report.push_back(static_cast<double>(run.key_bytes) / cfg.reports);
  }
  DetailedTimingSummary summary;
  summary.total = TimingSummary{Mean(seconds_per_report), StandardError(seconds_per_report),
                                Quantile(seconds_per_report, 0.50),
                                Quantile(seconds_per_report, 0.95), checksum};
  summary.keygen =
      TimingSummary{Mean(keygen_seconds_per_report), StandardError(keygen_seconds_per_report),
                    Quantile(keygen_seconds_per_report, 0.50),
                    Quantile(keygen_seconds_per_report, 0.95), checksum};
  summary.context =
      TimingSummary{Mean(context_seconds_per_report), StandardError(context_seconds_per_report),
                    Quantile(context_seconds_per_report, 0.50),
                    Quantile(context_seconds_per_report, 0.95), checksum};
  summary.eval = TimingSummary{Mean(eval_seconds_per_report), StandardError(eval_seconds_per_report),
                               Quantile(eval_seconds_per_report, 0.50),
                               Quantile(eval_seconds_per_report, 0.95), checksum};
  summary.accumulate = TimingSummary{Mean(accumulate_seconds_per_report),
                                     StandardError(accumulate_seconds_per_report),
                                     Quantile(accumulate_seconds_per_report, 0.50),
                                     Quantile(accumulate_seconds_per_report, 0.95), checksum};
  summary.key_bytes_per_report = Mean(key_bytes_per_report);
  return summary;
}

void PrintRow(const std::string& method, const TimingSummary& summary,
              double normalized_work, double time_ratio_vs_packed,
              const Config& cfg) {
  const uint64_t n = uint64_t{1} << cfg.log_n;
  std::cout << method << ","
            << summary.mean_seconds_per_report * 1000.0 << ","
            << summary.se_seconds_per_report * 1000.0 << ","
            << summary.p50_seconds_per_report * 1000.0 << ","
            << summary.p95_seconds_per_report * 1000.0 << ","
            << normalized_work << ","
            << time_ratio_vs_packed << ","
            << "microbenchmark,"
            << "google_distributed_point_functions,"
            << n << ","
            << cfg.log_n << ","
            << cfg.reports << ","
            << cfg.repeats << ","
            << cfg.threads << ","
            << CpuAcceleration() << ","
            << summary.checksum << "\n";
}

void PrintBreakdownRow(const std::string& method, const std::string& phase,
                       const TimingSummary& summary, double key_bytes_per_report,
                       double normalized_work, const Config& cfg) {
  const uint64_t n = uint64_t{1} << cfg.log_n;
  std::cout << method << ","
            << phase << ","
            << summary.mean_seconds_per_report * 1000.0 << ","
            << summary.se_seconds_per_report * 1000.0 << ","
            << summary.p50_seconds_per_report * 1000.0 << ","
            << summary.p95_seconds_per_report * 1000.0 << ","
            << key_bytes_per_report << ","
            << normalized_work << ","
            << "microbenchmark,"
            << "google_distributed_point_functions,"
            << n << ","
            << cfg.log_n << ","
            << cfg.reports << ","
            << cfg.repeats << ","
            << cfg.threads << ","
            << CpuAcceleration() << ","
            << summary.checksum << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  const Config cfg = ParseArgs(argc, argv);
  const DetailedTimingSummary packed = TimeMode(cfg, RunPackedOnce);
  const DetailedTimingSummary tuple = TimeMode(cfg, RunProductOnce);
  const DetailedTimingSummary dual = TimeMode(cfg, RunDualOnce);
  const double tuple_ratio =
      tuple.total.mean_seconds_per_report / packed.total.mean_seconds_per_report;
  const double dual_ratio =
      dual.total.mean_seconds_per_report / packed.total.mean_seconds_per_report;

  if (cfg.breakdown) {
    std::cout << "method,phase,time_ms_per_report,standard_error_ms,p50_ms_per_report,p95_ms_per_report,"
                 "key_bytes_per_report,normalized_expansion_work,source_type,"
                 "benchmark,N,log_n,reports_per_repeat,repeats,threads,cpu_acceleration,checksum\n";
    PrintBreakdownRow("Google DPF packed64", "total", packed.total,
                      packed.key_bytes_per_report, 1.0, cfg);
    PrintBreakdownRow("Google DPF packed64", "keygen", packed.keygen,
                      packed.key_bytes_per_report, 1.0, cfg);
    PrintBreakdownRow("Google DPF packed64", "context", packed.context,
                      packed.key_bytes_per_report, 1.0, cfg);
    PrintBreakdownRow("Google DPF packed64", "eval", packed.eval,
                      packed.key_bytes_per_report, 1.0, cfg);
    PrintBreakdownRow("Google DPF packed64", "accumulate", packed.accumulate,
                      packed.key_bytes_per_report, 1.0, cfg);
    PrintBreakdownRow("Google DPF tuple product", "total", tuple.total,
                      tuple.key_bytes_per_report, 1.0, cfg);
    PrintBreakdownRow("Google DPF tuple product", "keygen", tuple.keygen,
                      tuple.key_bytes_per_report, 1.0, cfg);
    PrintBreakdownRow("Google DPF tuple product", "context", tuple.context,
                      tuple.key_bytes_per_report, 1.0, cfg);
    PrintBreakdownRow("Google DPF tuple product", "eval", tuple.eval,
                      tuple.key_bytes_per_report, 1.0, cfg);
    PrintBreakdownRow("Google DPF tuple product", "accumulate", tuple.accumulate,
                      tuple.key_bytes_per_report, 1.0, cfg);
    PrintBreakdownRow("Google DPF dual-track", "total", dual.total,
                      dual.key_bytes_per_report, 2.0, cfg);
    PrintBreakdownRow("Google DPF dual-track", "keygen", dual.keygen,
                      dual.key_bytes_per_report, 2.0, cfg);
    PrintBreakdownRow("Google DPF dual-track", "context", dual.context,
                      dual.key_bytes_per_report, 2.0, cfg);
    PrintBreakdownRow("Google DPF dual-track", "eval", dual.eval,
                      dual.key_bytes_per_report, 2.0, cfg);
    PrintBreakdownRow("Google DPF dual-track", "accumulate", dual.accumulate,
                      dual.key_bytes_per_report, 2.0, cfg);
    return 0;
  }

  std::cout << "method,time_ms_per_report,standard_error_ms,p50_ms_per_report,p95_ms_per_report,"
               "normalized_expansion_work,time_ratio_vs_packed,source_type,"
               "benchmark,N,log_n,reports_per_repeat,repeats,threads,cpu_acceleration,checksum\n";
  PrintRow("Google DPF packed64", packed.total, 1.0, 1.0, cfg);
  PrintRow("Google DPF tuple product", tuple.total, 1.0, tuple_ratio, cfg);
  PrintRow("Google DPF dual-track", dual.total, 2.0, dual_ratio, cfg);
  return 0;
}
