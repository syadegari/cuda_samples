#include <cute/tensor.hpp>
#include <cutlass/util/device_memory.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace cute;

template <class SmemLayout>
__global__ void transpose(const float* a, float* b, int m, int n, SmemLayout s_layout) {
  // Both views use input coordinates (row, column). B's strides perform
  // the physical transpose: B(column, row) = A(row, column).
  Tensor mA = make_tensor(make_gmem_ptr(a), make_shape(m, n), make_stride(n, _1{}));
  Tensor mB = make_tensor(make_gmem_ptr(b), make_shape(m, n), make_stride(_1{}, m));
  auto tile = make_shape(_32{}, _32{});
  auto coord = make_coord(blockIdx.y, blockIdx.x);
  Tensor gA = local_tile(mA, tile, coord);
  Tensor gB = local_tile(mB, tile, coord);

  __shared__ float smem[cosize_v<SmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smem), s_layout);

  // Input: lanes advance across columns. Output: lanes advance across rows.
  auto tA = make_layout(make_shape(_8{}, _32{}), make_stride(_32{}, _1{}));
  auto tB = make_layout(make_shape(_32{}, _8{}), make_stride(_1{}, _32{}));
  Tensor tAgA = local_partition(gA, tA, threadIdx.x);
  Tensor tAsA = local_partition(sA, tA, threadIdx.x);
  Tensor tBsA = local_partition(sA, tB, threadIdx.x);
  Tensor tBgB = local_partition(gB, tB, threadIdx.x);

  copy(tAgA, tAsA);
  __syncthreads();
  copy(tBsA, tBgB);
}

int main(int argc, char** argv) {
  const bool padded = argc > 1 && std::strcmp(argv[1], "pad") == 0;
  const int bits = padded ? 6 : (argc > 1 ? std::atoi(argv[1]) : 0);
  const int m = argc > 2 ? std::atoi(argv[2]) : 4096;
  const int n = argc > 3 ? std::atoi(argv[3]) : 8192;
  assert(argc <= 4 && bits >= 0 && bits <= 6);
  assert(m > 0 && n > 0 && m % 32 == 0 && n % 32 == 0);

  const size_t count = size_t(m) * n;
  std::vector<float> a(count), b(count);
  for (size_t i = 0; i < count; ++i) a[i] = float(i % 65521);
  cutlass::DeviceAllocation<float> d_a(count), d_b(count);
  d_a.copy_from_host(a.data());

  auto base = make_layout(make_shape(_32{}, _32{}), make_stride(_32{}, _1{}));
  dim3 grid(n / 32, m / 32);
  auto launch = [&](auto layout) {
    transpose<<<grid, 256>>>(d_a.get(), d_b.get(), m, n, layout);
  };
  switch (bits) {
    case 0: launch(composition(Swizzle<0, 0, 5>{}, base)); break;
    case 1: launch(composition(Swizzle<1, 0, 5>{}, base)); break;
    case 2: launch(composition(Swizzle<2, 0, 5>{}, base)); break;
    case 3: launch(composition(Swizzle<3, 0, 5>{}, base)); break;
    case 4: launch(composition(Swizzle<4, 0, 5>{}, base)); break;
    case 5: launch(composition(Swizzle<5, 0, 5>{}, base)); break;
    case 6: launch(make_layout(make_shape(_32{}, _32{}), make_stride(Int<33>{}, _1{}))); break;
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(err));
    return 1;
  }

  d_b.copy_to_host(b.data());  // Waits for the kernel; CUTLASS checks copy errors.
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      if (b[size_t(col) * m + row] != a[size_t(row) * n + col]) {
        std::fprintf(stderr, "Mismatch at A(%d, %d)\n", row, col);
        return 1;
      }
    }
  }
  std::printf("layout=%s: PASS (%d x %d -> %d x %d, %zu elements)\n",
              argc > 1 ? argv[1] : "0", m, n, n, m, count);
}
