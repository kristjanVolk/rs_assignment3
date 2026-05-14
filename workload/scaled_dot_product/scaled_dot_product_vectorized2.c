#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <riscv_vector.h>

/*
 * Vectorized scaled dot-product kernel.
 * Computes the scaled dot‑product attention scores between query (q) and key (k) vectors,
 * multiplies the result by a value vector (v), and stores the output.
 *
 * Parameters:
 *   q   : query vector (length n)
 *   k   : key vector (length n)
 *   v   : value vector (length n)
 *   out : output vector (length n)
 *   n   : vector length (must be >= 1)
 */
static inline void scaled_dot_product_vectorized(const float *q, const float *k, const float *v, float *out, int n) {
    float sum = 0.0f;
    int i = 0;
    while (i < n) {
        size_t vl = vsetvl_e32m1((unsigned long)(n - i));
        vfloat32m1_t vq = vle32_v_f32m1(q + i, vl);
        vfloat32m1_t vk = vle32_v_f32m1(k + i, vl);
        vfloat32m1_t vmul = vmul_vv_f32m1(vq, vk, vl);
        // Reduction over the vector to accumulate into scalar sum
        sum += vredsum_vs_f32m1_f32m1(vmul, sum, vl);
        i += vl;
    }

    // Scale factor is 1 / sqrt(n)
    float scale = 1.0f / sqrtf((float)n);
    float score = sum * scale;

    /* Apply the score to the value vector */
    i = 0;
    while (i < n) {
        size_t vl = vsetvl_e32m1((unsigned long)(n - i));
        vfloat32m1_t vv = vle32_v_f32m1(v + i, vl);
        vfloat32m1_t vout = vmul_vf_f32m1(vv, score, vl);
        vse32_v_f32m1(out + i, vout, vl);
        i += vl;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <vector_length>\n", argv[0]);
        return EXIT_FAILURE;
    }
    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "Vector length must be positive.\n");
        return EXIT_FAILURE;
    }

    float *q = malloc(sizeof(float) * n);
    float *k = malloc(sizeof(float) * n);
    float *v = malloc(sizeof(float) * n);
    float *out = malloc(sizeof(float) * n);
    if (!q || !k || !v || !out) {
        fprintf(stderr, "Memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        q[i] = (float)rand() / RAND_MAX;
        k[i] = (float)rand() / RAND_MAX;
        v[i] = (float)rand() / RAND_MAX;
    }

    scaled_dot_product_vectorized(q, k, v, out, n);

    /* Print first 10 output values as a sanity check */
    printf("Output (first 10 values):\n");
    for (int i = 0; i < n && i < 10; i++) {
        printf("%f\n", out[i]);
    }

    free(q);
    free(k);
    free(v);
    free(out);
    return EXIT_SUCCESS;
}
