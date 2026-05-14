# Third homework assignment


## Vectorizing kernel (4 points)

You are given a program that computes **scaled dot-product attention scores** — the core non-GEMM operation in Transformer models. The source code is located in the `workload/scaled_dot_product` folder. Your task is to vectorize the kernel using RVV instructions and analyze the performance improvement compared to a scalar implementation. Ensure that your implementation is correct and produces the same results as the original scalar version. 

1. Set the vector L1 cache size to 8KB.
2. Run the program for different Vector Processing unit (VPU) sizes (128, 256, 512, 1024, 2048 and 4096 bits) and compare the performance of the vectorized implementation against the scalar version.
3. To observe the improvements, report CPI (Cycles Per Instruction) and execution time for both implementations across the different vector lengths.

> In all experiments, set the O3 processor as scalar processor, with default settings.

## Stencil kernel (3 points)

The heat diffusion equation is a partial differential equation (PDE) that describes the distribution of temperature in a given region over time. In 1D setting:

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

The provided program computes one time step of heat diffusion difference equation using a **3-point stencil**:

$$u_{new}[i] = u[i] + \alpha \cdot (u[i-1] - 2 \cdot u[i] + u[i+1])$$

For each interior point `i`, the update reads three neighboring values, approximates the spatial curvature via the second-order finite difference `u[i-1] - 2*u[i] + u[i+1]`, scales it by the diffusion coefficient `alpha`, and advances the solution in time. Boundary points are fixed at `u[0] = u[N-1] = 0.0` (Dirichlet conditions).

You are given both a scalar and a vectorized implementation, which is located in the `workload/heat_stencil/` folder. Your task is to analyze their performance:

1. Set the vector L1 cache to 8KB. 
2. Run vectorized implementation for VPU sizes 128, 256, 512, 1024, 2048, and 4096 bits.
3. Report **CPI** and **L1 cache miss events** for each implementation across all vector sizes.
4. Repeat the experiment with a 64KB L1 cache and compare results to assess the impact of cache size on performance.

> In all experiments, set the O3 processor as scalar processor, with default settings.

## Sparse matrix-vector multiplication (3 points)

The provided program computes **Sparse Matrix-Vector Multiplication (SpMV)**: given a vector of non-zero values `val[]`, a source vector `x[]`, and column indices `col[]`, it accumulates:

$$y[j] \mathrel{+}= \texttt{val}[j] \times x[\texttt{col}[j]]$$

Five vectorized RVV kernels are implemented, each representing a different memory access pattern:

- **Unit-stride:** Loads `val[]`, `x[]`, and `y[]` sequentially — all accesses are contiguous in memory, maximizing cache efficiency.
- **Strided:** Loads `x[]` with a fixed stride of 8 elements between accesses, introducing gaps in the memory access pattern.
- **Gather (sorted):** Loads `x[col[j]]` using sorted column indices — access order is predictable and monotonically increasing, allowing the hardware prefetcher to partially anticipate future accesses.
- **Gather (random):** Architecturally identical to the sorted gather, but uses randomly shuffled column indices — access order is unpredictable, defeating the hardware prefetcher.


The source code is located in the `workload/spmv/` folder. Your task is to analyze how memory access patterns affect performance across different cache configurations:

1. Set the vector L1 cache to 8KB.
2. Run all four kernels for VPU vector length of 256, 512, and 1024 bits.
3. Report **CPI** and **L1 cache miss events** for each kernel.
4. Repeat the experiment with a 64KB L1 cache and compare results to assess the impact of cache size on performance.

> In all experiments, set the O3 processor as scalar processor, with default settings.
