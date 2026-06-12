"""Tensor / vector energy-efficiency benchmark (bf16) for Jetson AGX Orin.

Mirrors the measurement protocol of ``power_bench.py`` (background ``tegrastats``
sampler, warmup -> static-baseline -> measure window, energy-per-unit derivation),
but instead of full VLA inference it profiles two classes of primitive ops:

  1. TENSOR (compute / "算力"): square matrix multiply A[N,N] @ B[N,N] for
     N = 128, 256, ..., 8192, using torch.matmul. Reports sustained TFLOPS and
     energy efficiency (GFLOP/J + mJ per matmul).

  2. VECTOR (latency / "延时"): the best available PyTorch operators ---
     softmax, layernorm, rmsnorm, adarms (DiT-style AdaRMSNorm), vector add,
     element-wise mul, and silu --- swept over input shapes B=1, seq in
     {512,1024}, hidden in {512,1024,2048,4096}. Reports per-call latency and
     energy per call (mJ).

Protocol (per requested spec):
  * One STATIC baseline up front: --static-seconds (default 30s) of idle power.
  * For EVERY workload: --warmup-seconds (default 10s) warmup, then
    --measure-seconds (default 30s) measurement counting completed ops.
  * Everything runs in bfloat16.

Power rails summed into "measured power" default to VDD_GPU_SOC + VDD_CPU_CV
(GPU/SoC + CPU), matching power_bench.py.

Run (on a Jetson with tegrastats on PATH, GPU available):
  python energy_bench.py                       # full run (~40+ min)
  python energy_bench.py --tensor-only
  python energy_bench.py --vector-only --output /tmp/vec_energy.json
  python energy_bench.py --warmup-seconds 3 --measure-seconds 8   # quick smoke
"""
from __future__ import annotations

import argparse
import ctypes
import json
import logging
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("energy_bench")

# Rails summed into the measured "op power" (mW). Same convention as power_bench.
DEFAULT_POWER_RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV")

# Workload sweep parameters (per spec).
MATMUL_SIZES = (128, 256, 512, 1024, 2048, 4096, 8192)
VECTOR_BATCH = 1
VECTOR_SEQS = (512, 1024)
VECTOR_HIDDENS = (512, 1024, 2048, 4096)


# ======================================================================
# Power sampling via tegrastats  (copied verbatim from power_bench.py)
# ======================================================================
_RAIL_RE = re.compile(r"([A-Z0-9_]+)\s+(\d+)mW")


class TegrastatsSampler:
    """Background reader of ``tegrastats`` that records per-rail mW samples.

    Each sample is (t_monotonic, {rail: mW, ...}). ``window_stats`` averages the
    samples whose timestamp falls inside a [t0, t1] window.
    """

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = int(interval_ms)
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.samples: list[tuple[float, dict[str, int]]] = []

    def start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "tegrastats not found -- this tool expects a Jetson with "
                "tegrastats on PATH for power measurement."
            ) from e
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        # Wait until at least one rail sample has arrived so callers don't race
        # the first reading.
        t0 = time.monotonic()
        while not self.samples and time.monotonic() - t0 < 5.0:
            time.sleep(0.02)
        if not self.samples:
            raise RuntimeError("tegrastats produced no power readings in 5s")

    def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            rails = {m.group(1): int(m.group(2)) for m in _RAIL_RE.finditer(line)}
            if rails:
                self.samples.append((time.monotonic(), rails))

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def window_stats(self, t0: float, t1: float, rails: tuple[str, ...]) -> dict:
        """Mean mW per rail + summed mean over `rails` for samples in [t0, t1]."""
        sel = [s for (t, s) in self.samples if t0 <= t <= t1]
        n = len(sel)
        per_rail: dict[str, float] = {}
        all_rails = set()
        for s in sel:
            all_rails.update(s.keys())
        for rail in sorted(all_rails):
            vals = [s[rail] for s in sel if rail in s]
            per_rail[rail] = (sum(vals) / len(vals)) if vals else float("nan")
        total = sum(per_rail.get(r, 0.0) for r in rails)
        return {"n_samples": n, "per_rail_mW": per_rail, "measured_total_mW": total}


# ======================================================================
# cuBLASLt autotuned matmul  (libtuned_matmul.so via ctypes)
# ======================================================================
class TunedMatmul:
    """Per-size autotuned bf16 GEMM backed by ``libtuned_matmul.so``.

    ``autotune(N, A, B, C)`` enumerates cuBLASLt candidate algorithms on the
    given device pointers, times each, and caches the fastest. ``run`` replays
    it with no per-call heuristic. Pointers are raw CUDA device addresses (int)
    from ``tensor.data_ptr()``. This recovers the achievable dense-bf16 peak
    that torch.matmul misses because it takes cuBLAS's (miscalibrated on
    sm_87) heuristic top pick instead of the best kernel.
    """

    def __init__(self, so_path: str, tune_iters: int = 30):
        self.lib = ctypes.CDLL(so_path)
        self.lib.tunedmm_init.restype = ctypes.c_int
        self.lib.tunedmm_prepare.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.tunedmm_prepare.restype = ctypes.c_int
        self.lib.tunedmm_run.argtypes = [ctypes.c_int]
        self.lib.tunedmm_run.restype = ctypes.c_int
        self.tune_iters = int(tune_iters)
        rc = self.lib.tunedmm_init()
        if rc != 0:
            raise RuntimeError(f"tunedmm_init failed (rc={rc})")

    def prepare(self, n) -> float:
        """Allocate buffers for NxN, autotune the cuBLASLt algo, cache it.
        Returns the best measured TFLOPS."""
        rc = self.lib.tunedmm_prepare(n, self.tune_iters)
        if rc < 0:
            raise RuntimeError(f"autotune/prepare failed for N={n} (rc={rc})")
        return rc / 100.0

    def run(self, n) -> None:
        rc = self.lib.tunedmm_run(n)
        if rc != 0:
            raise RuntimeError(f"tuned matmul run failed for N={n} (rc={rc})")


def ensure_tunedmm_lib(explicit: str | None, arch: str = "sm_87") -> str:
    """Return path to libtuned_matmul.so, building it from tuned_matmul.cu if absent."""
    here = Path(__file__).resolve().parent
    so = Path(explicit) if explicit else here / "libtuned_matmul.so"
    if so.exists():
        return str(so)
    src = here / "tuned_matmul.cu"
    if not src.exists():
        raise FileNotFoundError(f"{so} not found and {src} missing to build it")
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    cmd = [nvcc, "-O3", f"-arch={arch}", "-std=c++17", "-Xcompiler", "-fPIC",
           "-shared", str(src), "-o", str(so), "-lcublasLt"]
    log.info("building tuned matmul lib: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return str(so)


# ======================================================================
# Workload definitions: each builds a no-arg op() running ONE operation.
# ======================================================================
class Workload:
    """One benchmark unit: a no-arg ``op()`` plus metadata.

    ``flops_per_op`` (or None) lets the tensor path report TFLOPS; vector ops
    leave it None and report latency/energy only. ``op()`` must NOT call
    cuda.synchronize -- the driver bursts ``burst`` ops between syncs to amortize
    kernel-launch latency and keep the GPU saturated.
    """

    def __init__(self, category, name, shape_str, op, flops_per_op=None):
        self.category = category          # "tensor" | "vector"
        self.name = name                  # e.g. "matmul" | "softmax"
        self.shape_str = shape_str        # human-readable shape, e.g. "4096x4096"
        self.op = op
        self.flops_per_op = flops_per_op

    @property
    def label(self) -> str:
        return f"{self.category}/{self.name}[{self.shape_str}]"


def build_matmul_workloads(torch, device, tuned=None):
    """A[N,N] @ B[N,N] in bf16 for each N. Output buffer reused (out=) so a burst
    of matmuls runs back-to-back with a WAR dependency -> GPU stays saturated.

    If ``tuned`` (a TunedMatmul) is given, each size is autotuned via cuBLASLt
    enumeration up front and ``op`` replays the best algo; otherwise ``op`` uses
    torch.matmul (cuBLAS heuristic pick)."""
    works = []
    for n in MATMUL_SIZES:
        if tuned is not None:
            # The .so owns the A/B/C buffers (cudaMalloc, 512B-aligned) so the
            # fastest cuBLASLt kernels run; torch-tensor pointers would force a
            # slower fallback. autotune happens here, once per size.
            best = tuned.prepare(n)
            log.info("  [tune] matmul %dx%d -> best cuBLASLt algo %.2f TFLOPS", n, n, best)

            def op(n=n, _t=tuned):
                _t.run(n)
        else:
            a = torch.randn((n, n), dtype=torch.bfloat16, device=device)
            b = torch.randn((n, n), dtype=torch.bfloat16, device=device)
            c = torch.empty((n, n), dtype=torch.bfloat16, device=device)

            def op(a=a, b=b, c=c):
                torch.matmul(a, b, out=c)

        works.append(Workload("tensor", "matmul", f"{n}x{n}", op,
                              flops_per_op=2.0 * n * n * n))
    return works


def build_vector_workloads(torch, device):
    """softmax / layernorm / rmsnorm / adarms / add / mul / silu over the
    B=1, seq in {512,1024}, hidden in {512,1024,2048,4096} sweep, all bf16."""
    import torch.nn.functional as F
    works = []
    for seq in VECTOR_SEQS:
        for h in VECTOR_HIDDENS:
            shape = (VECTOR_BATCH, seq, h)
            sstr = f"{VECTOR_BATCH}x{seq}x{h}"
            x = torch.randn(shape, dtype=torch.bfloat16, device=device)
            y = torch.randn(shape, dtype=torch.bfloat16, device=device)
            # affine params for (layer/rms)norm
            ln_w = torch.randn((h,), dtype=torch.bfloat16, device=device)
            ln_b = torch.randn((h,), dtype=torch.bfloat16, device=device)
            rms_w = torch.randn((h,), dtype=torch.bfloat16, device=device)
            # adarms (DiT AdaRMSNorm): scale/shift from Linear(cond), cond=[B,H]
            cond = torch.randn((VECTOR_BATCH, h), dtype=torch.bfloat16, device=device)
            ada_w = torch.randn((2 * h, h), dtype=torch.bfloat16, device=device)
            ada_b = torch.randn((2 * h,), dtype=torch.bfloat16, device=device)

            def op_softmax(x=x):
                F.softmax(x, dim=-1)

            def op_layernorm(x=x, h=h, w=ln_w, b=ln_b):
                F.layer_norm(x, (h,), w, b)

            def op_rmsnorm(x=x, h=h, w=rms_w):
                F.rms_norm(x, (h,), w)

            def op_adarms(x=x, h=h, cond=cond, w=ada_w, b=ada_b):
                mod = F.linear(cond, w, b)                  # [B, 2H]
                scale, shift = mod.chunk(2, dim=-1)         # [B, H] each
                xn = F.rms_norm(x, (h,))                    # normalize, no weight
                return xn * (1.0 + scale).unsqueeze(1) + shift.unsqueeze(1)

            def op_add(x=x, y=y):
                torch.add(x, y)

            def op_mul(x=x, y=y):
                torch.mul(x, y)

            def op_silu(x=x):
                F.silu(x)

            for name, op in (
                ("softmax", op_softmax),
                ("layernorm", op_layernorm),
                ("rmsnorm", op_rmsnorm),
                ("adarms", op_adarms),
                ("add", op_add),
                ("mul", op_mul),
                ("silu", op_silu),
            ):
                works.append(Workload("vector", name, sstr, op))
    return works


# ======================================================================
# Benchmark driver
# ======================================================================
def measure_static(sampler, seconds, rails):
    log.info("measuring STATIC baseline power for %.0fs (GPU idle) ...", seconds)
    s0 = time.monotonic()
    time.sleep(seconds)
    s1 = time.monotonic()
    st = sampler.window_stats(s0, s1, rails)
    log.info("  static power = %.0f mW (%d samples)",
             st["measured_total_mW"], st["n_samples"])
    return st


def bench_workload(torch, w: Workload, sampler, *, warmup_seconds, measure_seconds,
                   burst, rails, p_static_mW, device) -> dict:
    # --- warmup ---
    log.info("[%s] warmup %.0fs ...", w.label, warmup_seconds)
    t = time.monotonic()
    while time.monotonic() - t < warmup_seconds:
        for _ in range(burst):
            w.op()
        torch.cuda.synchronize(device)

    # --- measure ---
    log.info("[%s] measure %.0fs ...", w.label, measure_seconds)
    a0 = time.monotonic()
    n_ops = 0
    while time.monotonic() - a0 < measure_seconds:
        for _ in range(burst):
            w.op()
        torch.cuda.synchronize(device)
        n_ops += burst
    a1 = time.monotonic()
    elapsed = a1 - a0

    active = sampler.window_stats(a0, a1, rails)
    p_active = active["measured_total_mW"]
    p_dyn = p_active - p_static_mW
    sec_per_op = elapsed / n_ops if n_ops else float("nan")

    # energy per op: P[W] * t[s] -> J, *1000 -> mJ.  gross = active, net = dynamic
    e_gross_mJ = (p_active / 1000.0) * sec_per_op * 1000.0
    e_net_mJ = (max(0.0, p_dyn) / 1000.0) * sec_per_op * 1000.0

    res = {
        "label": w.label,
        "category": w.category,
        "name": w.name,
        "shape": w.shape_str,
        "elapsed_s": round(elapsed, 4),
        "n_ops": n_ops,
        "ops_per_s": round(n_ops / elapsed, 4) if elapsed else None,
        "us_per_op": round(sec_per_op * 1e6, 4),
        "ms_per_op": round(sec_per_op * 1e3, 6),
        "power_mW": {
            "static": round(p_static_mW, 1),
            "active": round(p_active, 1),
            "dynamic": round(p_dyn, 1),
        },
        "n_power_samples": active["n_samples"],
        "per_rail_mW": {k: round(v, 1) for k, v in active["per_rail_mW"].items()},
        "energy_per_op_mJ": {
            "gross": round(e_gross_mJ, 6),
            "net": round(e_net_mJ, 6),
        },
    }

    if w.flops_per_op is not None:
        tflops = w.flops_per_op / sec_per_op / 1e12 if sec_per_op else float("nan")
        # energy efficiency = FLOP / J = (FLOP/s) / W.  Report GFLOP/J.
        eff_gross = (w.flops_per_op / (e_gross_mJ / 1000.0)) / 1e9 if e_gross_mJ else float("nan")
        eff_net = (w.flops_per_op / (e_net_mJ / 1000.0)) / 1e9 if e_net_mJ else float("nan")
        res["flops_per_op"] = w.flops_per_op
        res["tflops"] = round(tflops, 4)
        res["efficiency_gflop_per_J"] = {
            "gross": round(eff_gross, 4),
            "net": round(eff_net, 4),
        }

    log.info("  -> %.4f ms/op, %d ops, %.0f mW active%s",
             res["ms_per_op"], n_ops, p_active,
             f", {res['tflops']:.2f} TFLOPS" if "tflops" in res else "")
    return res


# ======================================================================
# Reporting
# ======================================================================
def _fmt_tensor_table(rows: list[dict]) -> str:
    lines = [
        "",
        "=" * 78,
        "  TENSOR  --  matmul A[N,N] @ B[N,N]  (bf16)",
        "=" * 78,
        f"  {'shape':>11} {'ms/op':>10} {'TFLOPS':>9} "
        f"{'mJ/op(net)':>11} {'GFLOP/J(net)':>13} {'active mW':>10}",
        "-" * 78,
    ]
    for r in rows:
        lines.append(
            f"  {r['shape']:>11} {r['ms_per_op']:>10.3f} {r['tflops']:>9.2f} "
            f"{r['energy_per_op_mJ']['net']:>11.3f} "
            f"{r['efficiency_gflop_per_J']['net']:>13.2f} "
            f"{r['power_mW']['active']:>10.0f}")
    lines.append("=" * 78)
    return "\n".join(lines)


def _fmt_vector_table(rows: list[dict]) -> str:
    lines = [
        "",
        "=" * 78,
        "  VECTOR  --  per-call latency + energy  (bf16)",
        "=" * 78,
        f"  {'op':>10} {'shape':>14} {'us/op':>11} "
        f"{'mJ/op(gross)':>13} {'mJ/op(net)':>11} {'active mW':>10}",
        "-" * 78,
    ]
    for r in rows:
        lines.append(
            f"  {r['name']:>10} {r['shape']:>14} {r['us_per_op']:>11.2f} "
            f"{r['energy_per_op_mJ']['gross']:>13.4f} "
            f"{r['energy_per_op_mJ']['net']:>11.4f} "
            f"{r['power_mW']['active']:>10.0f}")
    lines.append("=" * 78)
    return "\n".join(lines)


# ======================================================================
# CLI
# ======================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--static-seconds", type=float, default=30.0)
    p.add_argument("--warmup-seconds", type=float, default=10.0)
    p.add_argument("--measure-seconds", type=float, default=30.0)
    p.add_argument("--burst", type=int, default=32,
                   help="ops launched between cuda syncs (amortizes launch latency)")
    p.add_argument("--tegrastats-interval-ms", type=int, default=100)
    p.add_argument("--rails", default=",".join(DEFAULT_POWER_RAILS),
                   help="comma-separated tegrastats rails summed into 'op power'")
    p.add_argument("--tensor-only", action="store_true")
    p.add_argument("--vector-only", action="store_true")
    p.add_argument("--tune-matmul", action=argparse.BooleanOptionalAction, default=True,
                   help="autotune the cuBLASLt algo per matmul size before benchmarking "
                        "(default on; --no-tune-matmul uses torch.matmul's heuristic pick)")
    p.add_argument("--tunedmm-lib", default=None,
                   help="path to libtuned_matmul.so (auto-built from tuned_matmul.cu if absent)")
    p.add_argument("--tune-iters", type=int, default=30,
                   help="timed iterations per candidate algo during autotune")
    p.add_argument("--output", default=None, help="write full results to JSON")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    import torch
    if not torch.cuda.is_available():
        log.error("CUDA not available -- this benchmark requires a GPU.")
        return 2
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    rails = tuple(r.strip() for r in args.rails.split(",") if r.strip())

    # Optional cuBLASLt autotuner for the matmul (tensor) workloads.
    tuned = None
    if not args.vector_only and args.tune_matmul:
        try:
            cc = torch.cuda.get_device_capability(args.gpu)
            so = ensure_tunedmm_lib(args.tunedmm_lib, arch=f"sm_{cc[0]}{cc[1]}")
            tuned = TunedMatmul(so, tune_iters=args.tune_iters)
            log.info("matmul autotune ENABLED (cuBLASLt enumeration via %s)", so)
        except Exception as e:
            log.warning("matmul autotune unavailable (%s); using torch.matmul", e)
            tuned = None

    # Build workloads.
    works: list[Workload] = []
    if not args.vector_only:
        works += build_matmul_workloads(torch, device, tuned=tuned)
    if not args.tensor_only:
        works += build_vector_workloads(torch, device)
    log.info("built %d workloads (burst=%d); est. runtime ~%.0f min + %ds static",
             len(works), args.burst,
             len(works) * (args.warmup_seconds + args.measure_seconds) / 60.0,
             int(args.static_seconds))

    sampler = TegrastatsSampler(interval_ms=args.tegrastats_interval_ms)
    sampler.start()
    results: list[dict] = []
    try:
        static = measure_static(sampler, args.static_seconds, rails)
        p_static_mW = static["measured_total_mW"]
        for w in works:
            results.append(bench_workload(
                torch, w, sampler,
                warmup_seconds=args.warmup_seconds,
                measure_seconds=args.measure_seconds,
                burst=args.burst, rails=rails,
                p_static_mW=p_static_mW, device=device))
    finally:
        sampler.stop()

    # Reports.
    tensor_rows = [r for r in results if r["category"] == "tensor"]
    vector_rows = [r for r in results if r["category"] == "vector"]
    if tensor_rows:
        print(_fmt_tensor_table(tensor_rows))
    if vector_rows:
        print(_fmt_vector_table(vector_rows))

    payload = {
        "meta": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(args.gpu),
            "dtype": "bfloat16",
            "rails_summed": list(rails),
            "static_seconds": args.static_seconds,
            "warmup_seconds": args.warmup_seconds,
            "measure_seconds": args.measure_seconds,
            "burst": args.burst,
            "tegrastats_interval_ms": args.tegrastats_interval_ms,
            "matmul_autotune": bool(tuned),
        },
        "static_power_mW": round(static["measured_total_mW"], 1),
        "static_per_rail_mW": {k: round(v, 1) for k, v in static["per_rail_mW"].items()},
        "results": results,
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        log.info("wrote results -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
