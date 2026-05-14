/*
 * spmv.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <riscv_vector.h>

// Include the gem5 m5ops header file
#include <gem5/m5ops.h>


#define DEFAULT_NNZ  1024   /* number of non-zero entries                  */
#define DEFAULT_M    1024   /* size of source vector x (number of columns) */

/* ── Timer ──────────────────────────────────────────────────────────────── */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── Index generators ────────────────────────────────────────────────────── */

// make_sorted_indices()
static void make_sorted_indices(uint64_t *col, int nnz, int M)
{
    for (int j = 0; j < nnz; j++)
        col[j] = (uint64_t)(j % M);
}

// make_random_indices()
static void make_random_indices(uint64_t *col, int nnz, int M, unsigned seed)
{
    /* Build a shuffled base permutation of length M */
    int *perm = malloc(M * sizeof(int));
    for (int i = 0; i < M; i++) perm[i] = i;

    srand(seed);
    for (int i = M - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    /* Fill col[] by repeating the shuffled permutation */
    for (int j = 0; j < nnz; j++)
        col[j] = (uint64_t)(perm[j % M]);

    free(perm);
}

// make_byte_offsets()
static void make_byte_offsets(const uint64_t *col, uint64_t *offsets, int nnz)
{
    for (int j = 0; j < nnz; j++)
        offsets[j] = col[j] * sizeof(double);
}


void rvv_spmv_unit_stride(const double * restrict val,
                            const double * restrict x,
                            double       * restrict y,
                            int nnz)
{
    size_t i = 0;
    while (i < (size_t)nnz) {
        size_t vl = __riscv_vsetvl_e64m1((size_t)nnz - i);

        vfloat64m1_t vval = __riscv_vle64_v_f64m1(&val[i], vl);
        vfloat64m1_t vx   = __riscv_vle64_v_f64m1(&x[i],   vl);
        vfloat64m1_t vy   = __riscv_vle64_v_f64m1(&y[i],   vl);

        vy = __riscv_vfmacc_vv_f64m1(vy, vval, vx, vl);
        __riscv_vse64_v_f64m1(&y[i], vy, vl);
        i += vl;
    }
}

// Strided SpMV: y[j] += val[j] * x[j*stride]
void rvv_spmv_strided(const double * restrict val,
                       const double * restrict x,
                       double       * restrict y,
                       int nnz, int stride)
{
    ptrdiff_t stride_bytes = (ptrdiff_t)(stride * sizeof(double));

    size_t i = 0;
    while (i < (size_t)nnz) {
        size_t vl = __riscv_vsetvl_e64m1((size_t)nnz - i);

        vfloat64m1_t vval = __riscv_vle64_v_f64m1(&val[i], vl);

        vfloat64m1_t vx = __riscv_vlse64_v_f64m1(&x[i * stride],
                                                   stride_bytes, vl);

        vfloat64m1_t vy = __riscv_vle64_v_f64m1(&y[i], vl);
        vy = __riscv_vfmacc_vv_f64m1(vy, vval, vx, vl);
        __riscv_vse64_v_f64m1(&y[i], vy, vl);

        i += vl;
    }
}

// indexed gather SpMV: y[j] += val[j] * x[col[j]], with sorted column indices
void rvv_spmv_gather_sorted(const double   * restrict val,
                              const double   * restrict x,
                              const uint64_t * restrict offsets,  /* byte offsets */
                              double         * restrict y,
                              int nnz)
{
    size_t i = 0;
    while (i < (size_t)nnz) {
        size_t vl = __riscv_vsetvl_e64m1((size_t)nnz - i);

        vfloat64m1_t vval = __riscv_vle64_v_f64m1(&val[i], vl);

        /* Load the byte-offset index vector for this strip */
        vuint64m1_t vidx = __riscv_vle64_v_u64m1(&offsets[i], vl);
        vfloat64m1_t vx = __riscv_vluxei64_v_f64m1(x, vidx, vl);

        vfloat64m1_t vy = __riscv_vle64_v_f64m1(&y[i], vl);
        vy = __riscv_vfmacc_vv_f64m1(vy, vval, vx, vl);
        __riscv_vse64_v_f64m1(&y[i], vy, vl);

        i += vl;
    }
}

// indexed gather SpMV: y[j] += val[j] * x[col[j]], with random column indices
void rvv_spmv_gather_random(const double   * restrict val,
                              const double   * restrict x,
                              const uint64_t * restrict offsets,  /* byte offsets */
                              double         * restrict y,
                              int nnz)
{
    /* Architecturally identical to Variant 3 — same instruction sequence.
     * The performance difference comes entirely from the offsets[] content. */
    size_t i = 0;
    while (i < (size_t)nnz) {
        size_t vl = __riscv_vsetvl_e64m1((size_t)nnz - i);

        vfloat64m1_t vval = __riscv_vle64_v_f64m1(&val[i], vl);

        vuint64m1_t vidx = __riscv_vle64_v_u64m1(&offsets[i], vl);

        vfloat64m1_t vx = __riscv_vluxei64_v_f64m1(x, vidx, vl);

        vfloat64m1_t vy = __riscv_vle64_v_f64m1(&y[i], vl);
        vy = __riscv_vfmacc_vv_f64m1(vy, vval, vx, vl);
        __riscv_vse64_v_f64m1(&y[i], vy, vl);

        i += vl;
    }
}

// indexed scatter SpMV: y[col[j]] += val[j] * x[j], with random column indices
void rvv_spmv_scatter_random(const double   * restrict val,
                               const double   * restrict x,
                               const uint64_t * restrict offsets,  /* byte offsets into y */
                               double         * restrict y,
                               int nnz)
{
    size_t i = 0;
    while (i < (size_t)nnz) {
        size_t vl = __riscv_vsetvl_e64m1((size_t)nnz - i);

        /* Unit-stride loads of val[] and x[] — no scatter cost here */
        vfloat64m1_t vval = __riscv_vle64_v_f64m1(&val[i], vl);
        vfloat64m1_t vx   = __riscv_vle64_v_f64m1(&x[i],   vl);

        vuint64m1_t vidx = __riscv_vle64_v_u64m1(&offsets[i], vl);
        vfloat64m1_t vy_old = __riscv_vluxei64_v_f64m1(y, vidx, vl);
        vfloat64m1_t vy_new = __riscv_vfmacc_vv_f64m1(vy_old, vval, vx, vl);

        __riscv_vsoxei64_v_f64m1(y, vidx, vy_new, vl);

        i += vl;
    }
}



int main(int argc, char *argv[])
{
    int NNZ = DEFAULT_NNZ;
    int M   = DEFAULT_M;

   
    // Allocate arrays
    double   *val          = malloc(NNZ * sizeof(double));   // non-zero values
    double   *x            = malloc(M   * sizeof(double));   // source vector
    double   *x_strided    = malloc(NNZ * 8 * sizeof(double)); // padded for stride-8, NNZ*8 elements
    double   *y_ref        = calloc(NNZ,   sizeof(double));  // reference output
    double   *y_test       = calloc(NNZ,   sizeof(double));  // variant output
    double   *y_scatter_ref= calloc(M,     sizeof(double));  // scatter reference
    double   *y_scatter    = calloc(M,     sizeof(double));  // scatter output
    uint64_t *col_sorted   = malloc(NNZ * sizeof(uint64_t)); // sorted indices
    uint64_t *col_random   = malloc(NNZ * sizeof(uint64_t)); // random indices
    uint64_t *off_sorted   = malloc(NNZ * sizeof(uint64_t)); // byte offsets for sorted indices
    uint64_t *off_random   = malloc(NNZ * sizeof(uint64_t)); // byte offsets for random indices

    if (!val||!x||!x_strided||!y_ref||!y_test||
        !y_scatter_ref||!y_scatter||
        !col_sorted||!col_random||!off_sorted||!off_random) {
        fprintf(stderr, "malloc failed\n"); return 1;
    }

    /* ── Initialise val[], x[] ──────────────────────────────────────────── */
    srand(42);
    for (int i = 0; i < NNZ; i++)
        val[i] = (double)(rand() % 100) / 100.0 + 0.1;
    for (int i = 0; i < M; i++)
        x[i] = (double)(rand() % 200) / 100.0 - 1.0;

    /* Padded x for strided access: x_strided[j*stride] = x[j % M]
     * Allocate NNZ*8 doubles so j*8 is always in bounds for j < NNZ. */
    memset(x_strided, 0, NNZ * 8 * sizeof(double));
    for (int j = 0; j < NNZ; j++) {
        x_strided[j * 2] = x[j % M];   /* stride-2 lane */
        x_strided[j * 8] = x[j % M];   /* stride-8 lane */
    }

    /* ── Generate index sets ─────────────────────────────────────────────── */
    make_sorted_indices(col_sorted, NNZ, M);
    make_random_indices(col_random, NNZ, M, 12345);
    make_byte_offsets(col_sorted, off_sorted, NNZ);
    make_byte_offsets(col_random, off_random, NNZ);

    #ifdef GEM5
    m5_reset_stats(0, 0);
    #endif 
    rvv_spmv_unit_stride(val, x, y_test, NNZ);
    #ifdef GEM5
    m5_dump_stats(0, 0);
    #endif

    #ifdef GEM5
    m5_reset_stats(0, 0);
    #endif 
    rvv_spmv_strided(val, x_strided, y_test, NNZ, 8);
    #ifdef GEM5
    m5_dump_stats(0, 0);
    #endif
    
    #ifdef GEM5
    m5_reset_stats(0, 0);
    #endif
    rvv_spmv_gather_sorted(val, x, off_sorted, y_test, NNZ);
    #ifdef GEM5
    m5_dump_stats(0, 0);
    #endif


    #ifdef GEM5
    m5_reset_stats(0, 0);
    #endif
    rvv_spmv_gather_random(val, x, off_random, y_test, NNZ),
    #ifdef GEM5
    m5_dump_stats(0, 0);
    #endif



    free(val); free(x); free(x_strided);
    free(y_ref); free(y_test);
    free(y_scatter_ref); free(y_scatter);
    free(col_sorted); free(col_random);
    free(off_sorted); free(off_random);

    return 0;
}