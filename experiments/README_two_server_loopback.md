# Process-separated Google-DPF loopback benchmark

This benchmark exercises the Chafe path with three independent local
processes: a synthetic workload-generator/analyzer driver, server 0, and
server 1.  The online path is

`key generation -> key serialization -> TCP send -> server EvalAll -> accumulation -> acknowledgement`.

At epoch close, the two servers open their occupancy-byte shares directly to
one another, apply Drop or Pool, generate independent uniform dummy shares,
use a jointly generated secret permutation, and send two fixed-length release
shares to the combined driver.  For instrumentation, the same driver socket
also receives per-server timing and byte metadata.  This harness therefore
does not instantiate the paper's analyzer view; it benchmarks the data path
and records the difference explicitly.

Release randomness is cryptographic: each server obtains a private 128-bit
seed from BoringSSL `RAND_bytes`, expands dummy shares with the repository's
AES-128-CTR PRNG, and exchanges one seed share to derive a permutation seed
that is not sent to the combined driver.  A domain-separated stream adds
opposite masks to the two release-share vectors before they reach the driver,
so their decomposition reveals no more than the reconstructed fixed-length
release.  The deterministic `mt19937_64` generator is used only to construct
reproducible synthetic client inputs, not for protocol randomness.

## Build and run

Build target:

```text
//chafe:google_dpf_loopback_e2e_benchmark
```

From the repository root, after building the target:

```text
python experiments/two_server_loopback_benchmark.py --repeats=5 --collision-repeats=3
```

The script writes:

- `data/fig_two_server_loopback_e2e.csv`: all measured repetitions;
- `data/fig_two_server_loopback_e2e_summary.csv`: means and standard errors.

These files are intentionally absent from the release until the corrected
benchmark completes successfully.  Do not substitute results produced by an
earlier harness: the current version excludes verification work from protocol
timing and uses cryptographic release randomness.

The output separates two scenarios.  `fixed_128_write_domain_sweep` varies
`N` with 128 writes to expose DPF domain cost.  `collision_load_rho_1` runs
`m=N` at `N=2^12` and `N=2^14`, with `M=2m`, so Drop and Pool are exercised in
the collision regime used by the privacy and utility analysis.  Summary byte
columns report mean, minimum, and maximum across repetitions rather than
copying a single run.

The reported `epoch_close_ms` ends after the analyzer reconstructs the `M`
released values.  Subsequent offline ground-truth verification is reported
separately as `verification_ms` and is excluded from both
`epoch_close_ms` and `analyzer_reconstruct_release_ms`.

The optional integration test is:

```text
CHAFE_RUN_LOOPBACK_INTEGRATION=1 pytest -q experiments/test_two_server_loopback_benchmark.py
```

## Evidence boundary

Every row is labeled `loopback_local_processes`.  This is a real
process-separated, socket-mediated write-path measurement, but it is not a
multi-node deployment: all processes run on one host, TCP is plaintext, and
the client waits for an acknowledgement after every write rather than
pipelining requests.  Consequently, the CSV can support claims about path
coverage, local process separation, byte counts, and localhost performance;
it cannot support WAN, cluster-scaling, TLS, or production-throughput claims.
The workload generator and analyzer are also colocated in the driver process;
the CSV records this as a harness limitation, and the actual driver socket
contains both two fixed-length release shares and instrumentation metadata.
Reported communication fields are application bytes and exclude TCP/IP
framing.
The driver also generates well-formed keys directly; it does not benchmark a
malformed-write validity proof or denial-of-service protection.
