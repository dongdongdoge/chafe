from __future__ import annotations

from chafe_eval_lib import DATA_DIR, write_csv


# Lam et al., ASPLOS 2024, and the authors' archived GPU-DPF artifact report
# these V100/AES-128 results.  They are copied verbatim so artifact generation
# is deterministic and does not require network access.
GPU_DPF_AES128_V100 = {
    1 << 14: 52_536,
    1 << 16: 15_392,
    1 << 18: 3_967,
    1 << 20: 923,
}


def generate() -> list[dict]:
    rows: list[dict] = []
    for N, dpf_per_second in GPU_DPF_AES128_V100.items():
        rows.append(
            {
                "method": "GPU-DPF AES-128 (external V100)",
                "N": N,
                "log_n": N.bit_length() - 1,
                "dpf_per_second": dpf_per_second,
                "hardware": "NVIDIA V100",
                "prf": "AES-128",
                "security_parameter_bits": 128,
                "entry_words": 16,
                "entry_word_bits": 32,
                "entry_bits": 512,
                "batch_size": 512,
                "client_keygen_included": "false",
                "second_server_included": "false",
                "chafe_packed_write_accumulation_included": "false",
                "directly_comparable_to_chafe": "false",
                "source_type": "external_artifact_measurement",
                "benchmark": "facebookresearch_gpu_dpf_eval_gpu_aes128",
                "timed_scope": "one server's batched DPF-PIR table evaluation, including key H2D and result D2H; table initialization and client key generation occur before timing",
                "deployment_scope": "external GPU-DPF PIR kernel; not a Chafe hidden-write backend or end-to-end two-server measurement",
                "citation_key": "lam2024gpudpf",
                "paper_doi": "10.1145/3617232.3624855",
                "artifact_url": "https://github.com/facebookresearch/GPU-DPF",
            }
        )
    write_csv(DATA_DIR / "fig_gpu_dpf_external.csv", rows)
    return rows


if __name__ == "__main__":
    print(f"fig_gpu_dpf_external.csv: {len(generate())} rows")
