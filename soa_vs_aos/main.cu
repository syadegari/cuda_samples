#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct Coefficients {
  float u, v, w;
  float x[8], y[8], z;
};
static_assert(sizeof(Coefficients) == 80);

__global__ void update_aos(Coefficients* data, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < n; i += stride) {
    data[i].u += 10.f;
    data[i].y[0] += 10.f;
  }
}

__global__ void update_soa(float* u, float* y0, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < n; i += stride) {
    u[i] += 10.f;
    y0[i] += 10.f;
  }
}

void check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(err));
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  const char* layout = argc > 1 ? argv[1] : "aos";
  const bool soa = std::strcmp(layout, "soa") == 0;
  const int n = argc > 2 ? std::atoi(argv[2]) : 1 << 23;
  const int launches = argc > 3 ? std::atoi(argv[3]) : 1;
  assert(argc <= 4 && (soa || std::strcmp(layout, "aos") == 0));
  assert(n > 0 && launches > 0 && launches <= 100);

  // Both layouts contain all twenty fields. Only u and y[0] are updated.
  const size_t count = size_t(n) * 20;
  std::vector<float> data(count);
  for (size_t i = 0; i < count; ++i) {
    const size_t logical = soa ? (i % n) * 20 + i / n : i;
    data[i] = float(logical % 251);
  }
  float* d_data;
  check(cudaMalloc(&d_data, count * sizeof(float)));
  check(cudaMemcpy(d_data, data.data(), count * sizeof(float), cudaMemcpyHostToDevice));

  for (int r = 0; r < launches; ++r) {
    if (soa) update_soa<<<1024, 256>>>(d_data, d_data + size_t(11) * n, n);
    else update_aos<<<1024, 256>>>(reinterpret_cast<Coefficients*>(d_data), n);
  }
  check(cudaGetLastError());
  check(cudaMemcpy(data.data(), d_data, count * sizeof(float), cudaMemcpyDeviceToHost));
  check(cudaFree(d_data));

  for (size_t i = 0; i < count; ++i) {
    const int field = soa ? i / n : i % 20;
    const size_t logical = soa ? (i % n) * 20 + field : i;
    const float expected = float(logical % 251) + ((field == 0 || field == 11) ? 10.f * launches : 0.f);
    if (data[i] != expected) {
      std::fprintf(stderr, "Mismatch at element %zu\n", i);
      return 1;
    }
  }
  std::printf("%s: PASS (%d records, %d launches)\n", layout, n, launches);
}
