/*
 * heat_stencil.c
 *
 * 1D Heat Equation Stencil — Scalar and RISC-V Vector (RVV 1.0) implementations
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>
// Include the gem5 m5ops header file
#include <gem5/m5ops.h>

/* ────────────────────────────────────────────────────────────────────────────
 * Parameters
 * ─────────────────────────────────────────────────────────────────────────── */
#define DEFAULT_N     1024
#define DEFAULT_STEPS 2
#define DEFAULT_ALPHA 0.1

/* Stability condition for explicit Euler: alpha <= 0.5 */
#define STABILITY_LIMIT 0.5


// Initialise temperature array: half sine wave
static void init_sine(double *u, int N)
{
    for (int i = 0; i < N; i++)
        u[i] = sin(M_PI * (double)i / (double)(N - 1));
}

// SCALAR implementation
void heat_step_scalar(const double * restrict u,
                      double       * restrict u_new,
                      int N, double alpha)
{
    // Boundary conditions (fixed Dirichlet)
    u_new[0]     = 0.0;
    u_new[N - 1] = 0.0;

    // Interior stencil 
    for (int i = 1; i < N - 1; i++) {
        double left   = u[i - 1];
        double center = u[i];
        double right  = u[i + 1];
        u_new[i] = center + alpha * (left - 2.0 * center + right);
    }
}

// Scalar baseline for verification of RVV results
static double *run_scalar(int N, int steps, double alpha,
                          double *buf_a, double *buf_b)
{
    double *u     = buf_a;
    double *u_new = buf_b;

    for (int t = 0; t < steps; t++) {
        heat_step_scalar(u, u_new, N, alpha);
        // Swap buffers — zero-copy ping-pong 
        double *tmp = u;
        u     = u_new;
        u_new = tmp;
    }
    return u;   
}

// RVV implementation
void heat_step_vector(const double * restrict u,
                      double       * restrict u_new,
                      int N, double alpha)
{
    /* Boundary conditions */
    u_new[0]     = 0.0;
    u_new[N - 1] = 0.0;

    size_t i = 1;                       
    double left_scalar = u[0];          

    while (i < (size_t)(N - 1)) {
        size_t remaining = (size_t)(N - 1) - i;
        size_t vl = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1_t vc = __riscv_vle64_v_f64m1(&u[i], vl);
        vfloat64m1_t vl1 = __riscv_vfslide1up_vf_f64m1(vc, left_scalar, vl);
        double right_scalar = u[i + vl];   /* u[N-1]=0 for last strip */
        vfloat64m1_t vr1 = __riscv_vfslide1down_vf_f64m1(vc, right_scalar, vl);
        vfloat64m1_t stencil_sum = __riscv_vfadd_vv_f64m1(vl1, vr1, vl);
        stencil_sum = __riscv_vfmacc_vf_f64m1(stencil_sum, -2.0, vc, vl);
        vfloat64m1_t result = __riscv_vfmacc_vf_f64m1(vc, alpha, stencil_sum, vl);
        __riscv_vse64_v_f64m1(&u_new[i], result, vl);
        left_scalar = u[i + vl - 1];
        i += vl;
    }
}


// Run vector version (which may be scalar fallback)
static double *run_vector(int N, int steps, double alpha,
                          double *buf_a, double *buf_b)
{
    double *u     = buf_a;
    double *u_new = buf_b;

    for (int t = 0; t < steps; t++) {
        heat_step_vector(u, u_new, N, alpha);
        double *tmp = u;
        u     = u_new;
        u_new = tmp;
    }
    return u;
}




// Verification: compute max absolute difference between two arrays
static double max_diff(const double *a, const double *b, int N)
{
    double d = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = fabs(a[i] - b[i]);
        if (diff > d) d = diff;
    }
    return d;
}



/* ────────────────────────────────────────────────────────────────────────────
 * Main
 * ─────────────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    int    N     = DEFAULT_N;
    int    steps = DEFAULT_STEPS;
    double alpha = DEFAULT_ALPHA;


    printf(" 1D Heat Equation Stencil \n");
    printf("  N=%d, steps=%d, alpha=%.4f\n\n", N, steps, alpha);

    // Allocate four buffers: two for scalar, two for vector
    double *s_a = (double *)malloc(N * sizeof(double));
    double *s_b = (double *)malloc(N * sizeof(double));
    double *v_a = (double *)malloc(N * sizeof(double));
    double *v_b = (double *)malloc(N * sizeof(double));
    if (!s_a || !s_b || !v_a || !v_b) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    // Scalar run
    init_sine(s_a, N);
    memcpy(s_b, s_a, N * sizeof(double));   /* s_b starts as a copy */


    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    double *scalar_result = run_scalar(N, steps, alpha, s_a, s_b);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif


    // Vector run
    init_sine(v_a, N);
    memcpy(v_b, v_a, N * sizeof(double));

    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    double *vector_result = run_vector(N, steps, alpha, v_a, v_b);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif
    



    /* ── Verification ────────────────────────────────────────────────────── */
    double err = max_diff(scalar_result, vector_result, N);
    printf("\nVerification: max |scalar - vector| = %.3e  %s\n",
           err, err < 1e-10 ? "[PASS]" : "[FAIL - check implementation]");

    /* ── Cleanup ─────────────────────────────────────────────────────────── */
    free(s_a); free(s_b);
    free(v_a); free(v_b);

    return err < 1e-10 ? 0 : 1;
}