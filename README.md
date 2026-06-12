# EnergyEfficiencyProfile

An energy-efficiency benchmark for NVIDIA Jetson AGX Orin that profiles primitive GPU operations in bfloat16 and reports per-op latency, power, and energy.

Power is sampled in the background via `tegrastats`, and energy per op is derived from a warmup → static-baseline → measure protocol. Two classes of workloads are profiled:

- **Tensor (compute)**: square matrix multiply `A[N,N] @ B[N,N]` for N = 128 … 8192. Reports sustained TFLOPS and energy efficiency (GFLOP/J, mJ per matmul).
- **Vector (latency)**: softmax, layernorm, rmsnorm, AdaRMSNorm (DiT-style), add, mul, and silu, swept over B=1, seq ∈ {512, 1024}, hidden ∈ {512, 1024, 2048, 4096}. Reports per-call latency and energy per call (mJ).

## Requirements

- NVIDIA Jetson (tested on AGX Orin) with `tegrastats` on `PATH`
- Python with PyTorch (CUDA enabled)

## Usage

```bash
python energy_bench.py                                  # full run (~40+ min)
python energy_bench.py --tensor-only                    # matmul sweep only
python energy_bench.py --vector-only --output vec.json  # vector ops only
python energy_bench.py --warmup-seconds 3 --measure-seconds 8   # quick smoke test
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--static-seconds` | 30 | Idle-power baseline window measured once up front |
| `--warmup-seconds` | 10 | Warmup time before each workload |
| `--measure-seconds` | 30 | Measurement window per workload |
| `--burst` | 32 | Ops launched between CUDA syncs (amortizes launch latency) |
| `--tune-matmul` / `--no-tune-matmul` | on | Autotune the cuBLASLt algo per matmul size before benchmarking |
| `--tune-iters` | 30 | Timed iterations per candidate algo during autotune |
| `--rails` | `VDD_GPU_SOC,VDD_CPU_CV` | tegrastats rails summed into the measured power |
| `--output` | — | Write full results to a JSON file |

## Matmul autotuning

By default the matmul (tensor) workloads do **not** use `torch.matmul` — its cuBLAS
heuristic is badly miscalibrated on Orin's `sm_87` and leaves 40–70% of dense bf16
throughput on the table (e.g. ~24 vs ~40 TFLOPS at 8192²). Instead, before
benchmarking, `energy_bench.py` enumerates the cuBLASLt candidate algorithms for
each size, times them at steady-state clock, and caches the fastest (via
`libtuned_matmul.so`, auto-built from `tuned_matmul.cu` with `nvcc` on first run).
This makes the TENSOR results reflect the GPU's *achievable* peak efficiency
(~40 TFLOPS ≈ 95% of the dense bf16 ceiling) rather than the cuBLAS default. Pass
`--no-tune-matmul` to fall back to plain `torch.matmul`. Vector ops are elementwise
and unaffected. Requires `nvcc` (CUDA toolkit) on `PATH`.

## Measurement protocol

1. Measure a single **static** baseline of idle power (`--static-seconds`).
2. For every workload: warm up, then measure for `--measure-seconds` while counting completed ops.
3. Energy per op = mean active power × time per op. Both **gross** (active power) and **net** (active − static) values are reported.

Results are printed as tables and optionally written to JSON. A sample full run on AGX Orin is included as `energy_full.json` (with its log in `energy_full.log`).
