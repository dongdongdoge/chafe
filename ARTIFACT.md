# Artifact Notes

Run the artifact check from the repository root:

```bash
python experiments/artifact_check.py
```

The check uses the committed CSV files and processed dataset snapshots. Rebuild
public datasets with the fetch scripts listed in `README.md`.

Third-party DPF backends:

- Google `distributed_point_functions`: `third_party/distributed_point_functions/`
- FacebookResearch GPU-DPF: `third_party/GPU-DPF/`

The GPU-DPF source checker is `experiments/build_gpu_dpf.py`; the CSV adapter is
`experiments/gpu_dpf_interface.py`.
