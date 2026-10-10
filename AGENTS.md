# User preferences

Recorded from explicit user instructions on 2026-10-09:

- This workspace is being used on NCSA Delta. Use Delta's x86 container and Slurm configuration; do not assume this session is on DeltaAI.
- Use the `bhdw` allocation for compute jobs: `bhdw-delta-cpu` for CPU work and `bhdw-delta-gpu` for GPU work. Do not use the `bgdb` allocation unless the user explicitly requests it.
- Run dataset preprocessing, tokenization, rebuilds, tests, training, and inference on compute nodes inside a Slurm allocation. Do not run these workloads on login nodes.
- Login nodes may be used for reading/editing files, small administrative checks, and submitting or monitoring jobs. Verify the node and Slurm allocation before starting processing work.
- These preferences apply unless the user explicitly requests a different cluster or allocation.

## Efficient Slurm allocation

Recorded from explicit user instructions on 2026-10-09:

- Request the smallest practical Slurm allocation supported by measured workload needs. Start with one GPU for single-GPU training/inference; use only the CPUs and memory required.
- If a job is pending for resources, promptly reduce the request or broaden compatible partitions instead of waiting on an oversized allocation. Do not wait for multiple GPUs when one can do the work.
- Prefer modest query/batch counts for initial evaluation and serialize independent work when that improves scheduling. Keep comparisons scientifically matched and report any sample-count reduction.
- Use compatible Delta interactive partitions for short runs. Choose walltime from measured runtime and stay within the partition limit; do not over-request by default.
- Modify or cancel only this task's pending jobs, preserve completed outputs, and rewire dependent jobs when resubmitting.
- Report submission, runtime, and evaluation failures promptly.

## Aging SKM Stage 2 protocol

User correction on 2026-10-09: train across the whole young-to-old trajectory. Use YM2 and one old donor (currently OM6) for joint temporal training, and hold the other old donor (currently OM9) out for validation. Do not restrict optimization to YM2. Use stored cell pseudotime differences for interval labels. OM6 and OM9 are alternative old donors, not sequential aging stages. Baselines must use the same training donors; validation is not an independent test.
