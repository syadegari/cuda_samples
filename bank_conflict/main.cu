#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr)                                                \
  do {                                                                 \
    cudaError_t err = (expr);                                           \
    if (err != cudaSuccess) {                                           \
      std::fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(err));   \
      std::exit(1);                                                     \
    }                                                                  \
  } while (0)

static constexpr int BLOCKS = 256;
static constexpr int THREADS = 256;
static constexpr int READS = 1024;

__global__ void bank_read(unsigned* out, int stride) {
  // Fixed allocation for every stride: 1024 words, 4 KiB per block.
  __shared__ volatile unsigned sh[1024];
  const int t = threadIdx.x;
  const int lane = t & 31;

  for (int i = t; i < 1024; i += blockDim.x) sh[i] = i + 1;
  __syncthreads();

  unsigned sum = 0;
  // Volatile keeps the repeated shared loads; disabling unrolling keeps
  // one load instruction in the loop for the profiler's source view.
  #pragma unroll 1
  for (int i = 0; i < READS; ++i) {
    sum += sh[lane * stride];
  }

  out[blockIdx.x * blockDim.x + t] = sum;
}

int main(int argc, char** argv) {
  const int stride = argc > 1 ? std::atoi(argv[1]) : 1;
  assert(argc <= 2 && stride >= 0 && stride <= 33);
  // Largest index: 31 * 33 = 1023. All threads participate in every load.
  const int n = BLOCKS * THREADS;
  unsigned* d_out;
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(unsigned)));

  bank_read<<<BLOCKS, THREADS>>>(d_out, stride);
  CUDA_CHECK(cudaGetLastError());

  std::vector<unsigned> out(n);
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    const unsigned expected = READS * ((i & 31) * stride + 1);
    if (out[i] != expected) {
      std::fprintf(stderr, "stride=%d: out[%d]=%u, expected %u\n", stride, i, out[i], expected);
      return 1;
    }
  }
  std::printf("stride=%d: PASS (%d outputs)\n", stride, n);
  CUDA_CHECK(cudaFree(d_out));
}
