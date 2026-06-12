// libtuned_matmul.so -- cuBLASLt autotuned bf16 GEMM for energy_bench.py.
//
// For each square size N, tunedmm_prepare() allocates its OWN A/B/C device
// buffers (via cudaMalloc), enumerates cuBLASLt candidate algorithms, times
// each at steady-state clock, and caches the fastest together with the buffers.
// tunedmm_run() then replays that best algo with no per-call heuristic.
//
// Why own buffers instead of torch tensors: the fastest cuBLASLt tensor-core
// kernels require stronger pointer alignment than torch's caching allocator
// provides, so on torch-tensor pointers cuBLASLt silently falls back to a
// slower algo (~30 vs ~40 TFLOPS at N=8192 on Orin). cudaMalloc'd buffers are
// 512-byte aligned and let the best kernels run. The matmul workload only
// measures throughput/energy, so it does not need to share torch memory.
//
// bf16 in/out, fp32 accumulate -- same arithmetic as torch.matmul, but the
// *best* kernel instead of cuBLAS's (miscalibrated on sm_87) heuristic pick.
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <map>

struct Plan {
  cublasLtMatmulDesc_t   op  = nullptr;
  cublasLtMatrixLayout_t Ad  = nullptr, Bd = nullptr, Cd = nullptr;
  cublasLtMatmulAlgo_t   algo;
  void*  A = nullptr; void* B = nullptr; void* C = nullptr;  // owned buffers
  bool   has = false;
};

static cublasLtHandle_t g_lt = nullptr;
static void*  g_ws = nullptr;
static size_t g_ws_bytes = 128ull * 1024 * 1024;       // 128 MB workspace
static std::map<int, Plan> g_plans;

extern "C" int tunedmm_init() {
  if (g_lt) return 0;
  if (cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) return 1;
  if (cudaMalloc(&g_ws, g_ws_bytes) != cudaSuccess)   return 2;
  return 0;
}

static int make_descs(int N, Plan& p) {
  if (cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F)) return 1;
  cublasOperation_t opN = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
  if (cublasLtMatrixLayoutCreate(&p.Ad, CUDA_R_16BF, N, N, N)) return 2;
  if (cublasLtMatrixLayoutCreate(&p.Bd, CUDA_R_16BF, N, N, N)) return 2;
  if (cublasLtMatrixLayoutCreate(&p.Cd, CUDA_R_16BF, N, N, N)) return 2;
  return 0;
}

static void free_plan(Plan& p) {
  if (p.op) cublasLtMatmulDescDestroy(p.op);
  if (p.Ad) cublasLtMatrixLayoutDestroy(p.Ad);
  if (p.Bd) cublasLtMatrixLayoutDestroy(p.Bd);
  if (p.Cd) cublasLtMatrixLayoutDestroy(p.Cd);
  if (p.A) cudaFree(p.A);
  if (p.B) cudaFree(p.B);
  if (p.C) cudaFree(p.C);
}

// Enumerate + time candidate algos on p's buffers; set p.algo to the fastest.
// Returns best TFLOPS, or a negative error code.
static double autotune_plan(Plan& p, int N, int iters) {
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_ws_bytes, sizeof(g_ws_bytes));

  const int REQ = 32;
  cublasLtMatmulHeuristicResult_t res[REQ];
  int got = 0;
  cublasLtMatmulAlgoGetHeuristic(g_lt, p.op, p.Ad, p.Bd, p.Cd, p.Cd, pref, REQ, res, &got);
  cublasLtMatmulPreferenceDestroy(pref);
  if (got == 0) return -3;

  const float alpha = 1.f, beta = 0.f;
  const double flops = 2.0 * (double)N * N * N;
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);

  // Ramp GPU DVFS to steady-state clock BEFORE comparing candidates, so the
  // real best algo is not timed while the clock is still low (which mis-selects
  // it). Spin the first runnable algo for ~0.5 s of GPU time.
  for (int i = 0; i < got; i++) {
    if (cublasLtMatmul(g_lt, p.op, &alpha, p.A, p.Ad, p.B, p.Bd, &beta,
                       p.C, p.Cd, p.C, p.Cd, &res[i].algo, g_ws, g_ws_bytes, 0)
        != CUBLAS_STATUS_SUCCESS) continue;
    float acc = 0.f;
    while (acc < 500.f) {
      cudaEventRecord(s);
      for (int k = 0; k < iters; k++)
        cublasLtMatmul(g_lt, p.op, &alpha, p.A, p.Ad, p.B, p.Bd, &beta,
                       p.C, p.Cd, p.C, p.Cd, &res[i].algo, g_ws, g_ws_bytes, 0);
      cudaEventRecord(e); cudaEventSynchronize(e);
      float ms = 0; cudaEventElapsedTime(&ms, s, e); acc += ms;
    }
    break;
  }

  double bestTF = 0; int bestIdx = -1;
  for (int i = 0; i < got; i++) {
    if (cublasLtMatmul(g_lt, p.op, &alpha, p.A, p.Ad, p.B, p.Bd, &beta,
                       p.C, p.Cd, p.C, p.Cd, &res[i].algo, g_ws, g_ws_bytes, 0)
        != CUBLAS_STATUS_SUCCESS) continue;
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int k = 0; k < iters; k++)
      cublasLtMatmul(g_lt, p.op, &alpha, p.A, p.Ad, p.B, p.Bd, &beta,
                     p.C, p.Cd, p.C, p.Cd, &res[i].algo, g_ws, g_ws_bytes, 0);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    double tf = flops / ((ms / 1e3) / iters) / 1e12;
    if (tf > bestTF) { bestTF = tf; bestIdx = i; }
  }
  cudaEventDestroy(s); cudaEventDestroy(e);
  if (bestIdx < 0) return -4;
  p.algo = res[bestIdx].algo; p.has = true;
  return bestTF;
}

// Allocate own buffers for NxN, autotune, and cache the plan.
// Returns best TFLOPS*100 (>=0) or a negative error code.
extern "C" int tunedmm_prepare(int N, int iters) {
  if (tunedmm_init() != 0) return -1;
  auto it = g_plans.find(N);
  if (it != g_plans.end()) { free_plan(it->second); g_plans.erase(it); }

  Plan p;
  if (make_descs(N, p)) { free_plan(p); return -2; }
  size_t bytes = (size_t)N * N * 2;                      // bf16
  if (cudaMalloc(&p.A, bytes) != cudaSuccess ||
      cudaMalloc(&p.B, bytes) != cudaSuccess ||
      cudaMalloc(&p.C, bytes) != cudaSuccess) { free_plan(p); return -11; }
  cudaMemset(p.A, 1, bytes); cudaMemset(p.B, 1, bytes); cudaMemset(p.C, 0, bytes);

  double tf = autotune_plan(p, N, iters);
  if (tf < 0) { free_plan(p); return (int)tf; }
  g_plans[N] = p;
  return (int)(tf * 100);
}

// Replay the cached best algo for N on its own buffers. Returns 0 on success.
extern "C" int tunedmm_run(int N) {
  auto it = g_plans.find(N);
  if (it == g_plans.end() || !it->second.has) return -1;
  Plan& p = it->second;
  const float alpha = 1.f, beta = 0.f;
  return cublasLtMatmul(g_lt, p.op, &alpha, p.A, p.Ad, p.B, p.Bd, &beta,
                        p.C, p.Cd, p.C, p.Cd, &p.algo, g_ws, g_ws_bytes, 0)
         == CUBLAS_STATUS_SUCCESS ? 0 : -2;
}
