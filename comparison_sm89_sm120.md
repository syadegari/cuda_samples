CUDA kernel development: Ada SM89 vs GeForce Blackwell SM120

Technical comparison, including SM100/TMA differences and implications for LLM inference
Prepared: 26 October 2026

Executive summary

The RTX 4070 Ti SUPER (Ada, sm_89) and RTX 5070 Ti (GeForce Blackwell, sm_120) are unusually useful comparison targets: both have 16 GB VRAM and nearly equal theoretical FP32 throughput, but the latter has ~33% more external-memory bandwidth and exposes Blackwell’s native FP4/FP6 and block-scaled Tensor Core instructions. The RTX 5060 Ti 16 GB exposes the same sm_120 instruction target, but with far fewer SMs and much less throughput.

Most important architectural distinction: sm_120 is not a cut-down sm_100 with every datacenter feature. It has native FP4/FP6 warp-level mma.sync, a Tensor Memory Accelerator (TMA) for tensor copies, thread-block clusters, and dynamic register redistribution. It does not have tcgen05.mma, Tensor Memory (TMEM), or TMA-load multicast. That changes how to design low-level GEMM kernels.

1. Concrete GPU comparison

|Property                     |RTX 4070 Ti SUPER     |RTX 5070 Ti          |RTX 5060 Ti 16 GB    |
|-----------------------------|---------------------:|--------------------:|--------------------:|
|Architecture / CUDA CC       |Ada / `sm_89`         |Blackwell / `sm_120` |Blackwell / `sm_120` |
|Streaming multiprocessors    |66                    |70                   |36                   |
|CUDA cores                   |8,448                 |8,960                |4,608                |
|Tensor Cores                 |264, fourth generation|280, fifth generation|144, fifth generation|
|VRAM                         |16 GB GDDR6X          |16 GB GDDR7          |16 GB GDDR7          |
|Nominal memory bandwidth     |672 GB/s              |**896 GB/s**         |448 GB/s             |
|L2 cache                     |48 MB                 |48 MB                |32 MB                |
|Theoretical FP32             |~44.1 TFLOP/s         |~43.9 TFLOP/s        |~23.7 TFLOP/s        |
|Maximum shared memory per CTA|99 KiB (opt-in)       |99 KiB (opt-in)      |99 KiB (opt-in)      |
|Nominal board power          |285 W                 |300 W                |180 W                |

Notes: Values are nominal board/boost-clock specifications, not measured scientific-kernel performance. FP32 peak does not establish BF16/FP16/FP4 performance. NVIDIA “AI TOPS” figures can use different precisions/sparsity assumptions and should not be compared without matching the calculation mode. Individual cards may use different boost clocks.

For conventional FP32 and BF16/FP16 work, 4070 Ti SUPER and 5070 Ti are in broadly similar compute classes; actual kernel speed will depend on instruction mix, clocks, memory behavior, and compiler scheduling. The 5070 Ti’s bandwidth advantage is especially relevant to memory-bound LLM decode.

2. Instruction and architectural feature matrix

|Feature                                          |Ada `sm_89`|GeForce Blackwell `sm_120`|Datacenter Blackwell `sm_100`          |
|-------------------------------------------------|-----------|--------------------------|---------------------------------------|
|Warp-level `mma.sync` FP16 / BF16 / TF32         |Yes        |Yes                       |Yes                                    |
|FP8 E4M3/E5M2 Tensor Core operations             |Yes        |Yes                       |Yes                                    |
|Native floating-point FP6 / FP4 MMA              |No         |**Yes**                   |**Yes**                                |
|MXFP4 / NVFP4 block-scaled MMA                   |No         |**Yes, warp-level**       |**Yes, `tcgen05` path**                |
|INT8/INT4 Tensor Core operations                 |Yes        |Yes                       |Yes                                    |
|Structured sparse Tensor Core MMA                |Yes        |Yes                       |Yes                                    |
|`cp.async` ordinary async copies                 |Yes        |Yes                       |Yes                                    |
|**Descriptor-based TMA tensor loads/stores**     |**No**     |**Yes**                   |**Yes**                                |
|**TMA cluster load multicast**                   |No         |**No**                    |**Yes**                                |
|Thread-block clusters / distributed shared memory|No         |**Yes**                   |**Yes**                                |
|`setmaxnreg.inc/dec` register redistribution     |No         |**Yes**                   |**Yes**                                |
|Hopper `wgmma.mma_async` PTX path                |No         |No                        |Not the native Blackwell `tcgen05` path|
|**`tcgen05.mma` and Tensor Memory (TMEM)**       |No         |**No**                    |**Yes**                                |
|Approximate maximum shared memory per CTA        |99 KiB     |99 KiB                    |~227 KiB                               |

The table distinguishes instruction availability from whether a particular CUTLASS kernel is implemented or optimized. Architecture-specific PTX legality must be checked against the actual PTX ISA version and compiler target.

3. What SM120 adds for FP4/FP6 GEMM

Native low-precision arithmetic

Ada already has FP8 (E4M3, E5M2), as well as integer INT8/INT4 MMA. However, integer INT4 is not floating-point FP4. On SM89, implementing a floating-point FP4 GEMM generally requires converting/repacking inputs into another Tensor Core datatype or using an alternative integer formulation.

SM120 adds support for FP6 (E2M3 and E3M2) and FP4 (E2M1) through Blackwell mma.sync instruction variants. Representative PTX instruction-family fragments are:

mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4...;

mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale...;

mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale...;

mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale...;

These lines show instruction families, not complete compilable PTX: real instructions need supported operand and scale types, fragment registers, shapes, and scale-vector qualifiers. Crucially, these are 32-thread warp-level instructions, not SM100 tcgen05 instructions.

Two FP4 scaling schemes worth distinguishing

● MXFP4: E2M1 FP4 data, typically one UE8M0 (power-of-two-style) block scale per 32 values.

● NVFP4: E2M1 FP4 data with finer scaling (typically an FP8-scale block per 16 values) plus higher-level scale handling. PTX uses scale types and vector-size qualifiers corresponding to the selected encoding.

Both are fundamentally different from GPTQ, AWQ, GGUF Q4, or generic signed INT4: a 4-bit model weight file does not automatically translate into native FP4 Tensor Core execution. Data encoding, scale layout, and arithmetic semantics must be made compatible with the mma.sync variant.

The dense compute advantage from SM120 is most interesting for quantized prefill, batched GEMM, and MoE expert kernels. It is not a promise of proportionately higher end-to-end tokens/s: dequantization, scale loads, conversion, epilogues, memory traffic, and model accuracy constraints can dominate.

4. TMA clarification: SM120 absolutely has it, but not all SM100 modes

This is the area most worth getting right.

TMA (Tensor Memory Accelerator) originated with Hopper sm_90; it is not exclusive to datacenter Blackwell sm_100. NVIDIA’s CUDA compute-capability feature table explicitly lists a hardware TMA unit for compute capability 12.x. NVIDIA’s SM120-focused CUTLASS GEMM examples use TMA pipelines.

Supported on SM120: descriptor-driven asynchronous tensor copies, including multidimensional tiled loads from global memory to shared memory and supported stores in the opposite direction. In PTX, the key family is cp.async.bulk.tensor (alongside bulk-copy mechanisms). A kernel may use tensor maps/descriptors, asynchronous transaction completion (mbarrier), and producer/consumer pipelining to reduce per-thread address generation and overlap transfer with MMA.

Not supported on SM120: TMA load multicast to distribute one tensor load to multiple CTAs in a cluster. NVIDIA’s official SM120 NVFP4 CUTLASS example explicitly says GeForce RTX 50-series GPUs do not support TMA-load multicast and uses a 1×1×1 cluster shape for that GEMM.

Important nuance: the 1×1×1 choice does not mean SM120 lacks all thread-block clusters. CUDA documents thread-block clusters and distributed shared memory for 12.x; other algorithms can use multi-CTA clusters (for example, cross-CTA reductions). It means one should not design an SM120 GEMM around the SM100 TMA multicast optimization.

TMA is not TMEM: TMA is primarily a data-transfer engine. Tensor Memory (TMEM) is SM100’s dedicated on-chip storage used in its tcgen05 Tensor Core execution model. SM120 has TMA but no TMEM and no tcgen05.mma. On SM120, operands normally flow through shared-memory/register fragments, using ldmatrix and warp-level MMA; accumulators remain register-resident.

A practical pipeline comparison:

Ada SM89:       global -> cp.async (per-thread cooperative copies) -> shared

                shared -> ldmatrix -> registers -> mma.sync -> register accumulators


Blackwell SM120: global -> TMA tensor-tile copy -> shared

                 shared -> ldmatrix -> registers -> mma.sync FP4/FP6/block-scale

                 -> register accumulators


Blackwell SM100: global -> TMA (including suitable multicast) -> shared

                 -> tcgen05 operand/TMEM dataflow -> tcgen05.mma / TMEM

This diagram is conceptual, not a claim that every kernel must take that route. The biggest SM120 opportunity is TMA + native block-scaled warp-level MMA, without attempting to port SM100’s TMEM kernel structure.

5. Other useful SM120 mechanisms

Dynamic register redistribution: setmaxnreg.inc.sync.aligned / setmaxnreg.dec.sync.aligned are supported on sm_120a/family targets. A producer warpgroup doing data movement may relinquish registers, leaving more for consumer warps accumulating matrix tiles. This is valuable because SM120’s MMA accumulators consume ordinary registers, and occupancy can collapse if tile sizes are too aggressive. This mechanism originated with Hopper; it is not unique to Blackwell, but is new relative to Ada SM89.

Thread-block clusters and distributed shared memory: useful for certain reductions, cross-CTA data exchange, or specialized scheduling. Cluster support and TMA multicast support are separate capabilities; do not conflate them.

More format/conversion options: FP4/FP6 packing, numeric conversions, block-scale layout and ldmatrix fragment ordering matter. A common pitfall is confusing the container representation used by unscaled FP4/FP6 f8f6f4 instructions with the tightly packed FP4 representation used by mxf4 block-scaled MMA.

Shared-memory limit: GeForce SM120 and Ada SM89 both have around 99 KiB maximum shared memory per CTA, far less than datacenter SM100’s ~227 KiB. A large SM100 GEMM tile may simply not fit, even if its arithmetic could otherwise be rewritten for SM120.

6. Implications for LLM and general CUDA kernels

(a) LLM decode, batch size 1. Often dominated by weight read bandwidth, cache reuse, and launch overhead, rather than peak GEMM FLOP/s. The RTX 5070 Ti’s 896 GB/s versus 4070 Ti SUPER’s 672 GB/s is a concrete advantage in bandwidth-limited regimes. Native FP4 can reduce weight traffic if the representation is compatible; however, scales, KV-cache traffic, conversion, and CPU/framework overhead still matter.

(b) Prefill and batched inference. Compute intensity tends to be higher. Here SM120 native NVFP4/MXFP4 MMA and optimized scale handling are more differentiated. Tune tile shapes (M/N/K), stage counts, epilogue fusion, and number of resident CTAs rather than relying on peak AI TOPS.

(c) MoE expert GEMMs. Good research area for grouped low-precision GEMM, per-expert tile scheduling, scale layout, and persistent work assignment; beware many small/irregular GEMMs and expert imbalance.

(d) Attention kernels. TMA-assisted tiled Q/K/V transport is interesting, but softmax numerics, masking, online normalization, and KV-cache access can dominate. Test BF16/FP16 baselines before introducing FP4 attention inputs. Precision/accuracy constraints matter.

(e) General kernels. Try TMA pipelines for contiguous multidimensional tiles, cluster-based reductions where justified, fused reductions/normalizations, sparse MMA with valid metadata, and register-specialized producer/consumer work partitioning. For memory-bound reductions, TMA is not necessarily superior to an optimized ordinary load: setup cost and shape constraints count.

(f) Scientific FP64. Neither RTX 4070 Ti SUPER nor RTX 5070 Ti is an appropriate replacement for an FP64-focused datacenter/compute GPU. Both are consumer GPUs with weak FP64 throughput relative to FP32.

7. What to benchmark rather than infer from specifications

Compare the cards with identical tensor shapes, precision, batch size, layout and measurement methodology:

1. BF16 GEMM (M,N,K sweep), using the same framework and a hand-tuned kernel on each architecture.

2. FP8 GEMM on both architectures, then native MXFP4/NVFP4 GEMM on SM120 (report the different numerical formats explicitly, not as an apples-to-apples speedup claim).

3. TMA versus cp.async on SM120 for different tile sizes, alignments, staging depths and occupancy limits.

4. Token decode (batch=1) separately from prompt prefill and larger-batch inference; record tokens/s, bytes moved, and achieved memory bandwidth.

5. Microbenchmarks for MMA occupancy/latency, register usage, spills, SMEM conflicts, barrier wait time and L2 hit rate (Nsight Compute).

6. Validate output error, quantization calibration and actual model quality whenever changing precision.

Avoid comparing sparse FP4 “AI TOPS” against dense BF16 FLOPS as if they were interchangeable.

8. Toolchain and compilation targets

A suitable modern CUDA toolkit and driver are required for RTX 50-series. NVIDIA introduced explicit RTX 50-series support with CUDA 12.8; newer CUDA 13.x versions offer additional feature-family compilation targets and compiler updates.

# Ordinary architecture target:

nvcc -O3 -arch=sm_120 kernel.cu


# Architecture-specific SM120 features (toolkit support required):

nvcc -O3 -arch=sm_120a kernel.cu


# Ada baseline:

nvcc -O3 -arch=sm_89 kernel.cu

sm_120f / compute_120f are available in newer toolchains for family-specific features with a different compatibility contract than sm_120a. Confirm toolchain support with nvcc --list-gpu-arch / nvcc --list-gpu-code and check the PTX target requirement for each intrinsic. Code compiled for sm_100a and relying on tcgen05 is not portable to SM120 merely because both GPUs are called Blackwell.

For low-level development, start with NVIDIA CUTLASS/CuTe DSL SM120 warp-MMA examples, then inspect generated PTX/SASS. SM100-focused tcgen05 tutorials are useful for conceptual contrasts but cannot be transplanted unmodified.

9. Recommendation

For SM120-targeted kernel research without a major conventional-performance sacrifice, the RTX 5070 Ti 16 GB is the sensible comparison point against the RTX 4070 Ti SUPER: similar FP32 peak, more bandwidth, equal VRAM, and native Blackwell low-precision MMA features. The RTX 5060 Ti 16 GB is useful for validating the same architecture-specific instruction paths at lower parallel throughput, but it is not a 4070 Ti SUPER-class performance replacement.

Most promising prototype: a fused NVFP4 W4A4 GEMM for prefill, using efficient block-scale loading, TMA global-to-shared staging, ldmatrix shared-to-register operand movement, warp-level block-scaled mma.sync, an optimized epilogue, and careful register/SMEM budgeting. Compare it against BF16 and FP8 baselines. This is distinctly an SM120 problem: no TMEM, no tcgen05, no TMA multicast, but genuine new low-precision MMA and TMA unicast capabilities.

Primary references

1. NVIDIA, CUDA compute capability device table: https://developer.nvidia.com/cuda/gpus

2. NVIDIA, CUDA programming guide — feature support for CC 8.9, 10.x, 12.x: https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/05-appendices/compute-capabilities.html

3. NVIDIA, PTX ISA reference (warp-level MMA, TMA copies, register operations): https://docs.nvidia.com/cuda/parallel-thread-execution/

4. NVIDIA CUTLASS, official GeForce SM120 NVFP4 GEMM example (explicit multicast limitation): https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu

5. NVIDIA CUTLASS, SM120 warp-level block-scaled MMA traits: https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm120.hpp

6. NVIDIA, RTX 5070 Ti official product specifications: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/

7. NVIDIA CCCL, TMA architecture and CUDA copying APIs: https://nvidia.github.io/cccl/unstable/cccl/tma.html

8. Dao-AILab QuACK, practitioner notes on SM120 TMA, clusters and setmaxnreg: https://github.com/Dao-AILab/quack/blob/main/AI/sm120_hardware.md
