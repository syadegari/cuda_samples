# Matrix transpose with CuTe C++ shared layouts

This is a conventional tiled matrix transpose expressed with CuTe C++. It reads a row-major `M x N` matrix and writes its row-major `N x M` transpose. Each block processes one 32x32 tile with 256 threads. Every element is loaded once, passes through shared memory, and is stored once. There is no repeated shared-read loop, volatile memory, warmup, or custom timing harness.

The host uses CUTLASS `DeviceAllocation` for allocation and transfers, launches one kernel, and checks every output element. Dimensions must be positive multiples of 32; the example does not need edge predication under that contract.

## Who controls bank conflicts?

CuTe's shared-memory layout is explicit. Its generic tensor/copy operations do not automatically replace a plain layout with a conflict-free layout. NVIDIA's [CuTe C++ GEMM tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html) passes shared layouts into the kernel and describes choosing them to avoid conflicts. Higher-level CUTLASS builders can select architecture-specific shared layouts for their operations; for example, the [SM90 GEMM builder](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl) uses shared-layout selectors. That selection is specific to the builder and its operation, rather than a general conflict-removal pass on arbitrary code.

Here, the same kernel accepts different shared layouts. Input and output thread partitions stay fixed. Lanes advance along input columns during the load and along input rows during the store, making both global accesses coalesced. The shared tile exchanges data between those thread partitions. Both global tensor views use input coordinates `(row, column)`:

```text
A(row, column) -> row * N + column
B(row, column) -> column * M + row
```

