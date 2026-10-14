# Shared-memory bank conflicts

Start with `main.cu`: one kernel, one launch, and a check of the output. Each lane repeatedly reads `sh[lane * stride]`. The stride is the only kernel parameter we change. Nsight Compute measures the duration and shared-memory traffic; the program has no timing harness or warmup loop.

## Predict the access pattern

For our 32-bit elements, consecutive words map to consecutive banks:

```text
word(lane) = lane * stride
bank(lane) = (lane * stride) % 32
```

For example, stride 2 gives banks `0, 2, ..., 30, 0, 2, ..., 30`. Each bank receives two distinct words, producing a 2-way access. Stride 32 puts all 32 distinct words in bank 0. Stride 33 wraps around the banks like stride 1. Stride 0 is different: all lanes read the same word, which is broadcast without a bank conflict. See NVIDIA's [shared-memory description](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

For nonzero strides here, the degree is `gcd(stride, 32)`. Strides 32 and 33 model the column-access address pattern of a tile with row pitches 32 and 33, respectively: adding one word of padding changes the bank mapping.

The kernel keeps the experiment controlled:

- All cases use 256 blocks, 256 threads per block, and 4 KiB of shared memory.
- Initialization writes consecutive words and finishes at `__syncthreads()`. The measured loop only reads shared memory; each thread writes one final result to its own global-memory location.
- `volatile` prevents the compiler from replacing repeated loads with a register value. `#pragma unroll 1` leaves one shared-load instruction in the loop. Repeating it 1024 times makes its traffic dominate initialization.
- All lanes participate, and the maximum shared index is `31 * 33 = 1023`.

This is an artificial access-pattern experiment. In an application, reading an unchanged value once and keeping it in a register would be preferable. The duration includes loop instructions and initialization, so it is not a measurement of the latency of one shared-memory read.
