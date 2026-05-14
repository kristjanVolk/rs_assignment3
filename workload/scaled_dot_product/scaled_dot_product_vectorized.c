#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <gem5/m5ops.h>
#include <riscv_vector.h>

#define SEQ_LEN 256
#define D_K 64
#define SCALE 0.125f  // 1/sqrt(64)

void softmax_row(float *scores, float *output, int len) {
    float max_val = scores[0];
    for (int i = 1; i < len; i++) {
        if (scores[i] > max_val)
            max_val = scores[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        output[i] = expf((scores[i] - max_val) * SCALE);
        sum += output[i];
    }

    for (int i = 0; i < len; i++) {
        output[i] /= sum;
    }
}

// Optimized vectorized dot product using LMUL=8 for maximum throughput
float dot_product_vectorized(float *q, float *k, int dim) {
    float result = 0.0f;
    
    // Initialize accumulator for reduction
    vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    
    // Use maximum vector length for the given dimension
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    
    // Initialize m8 accumulator for partial products
    vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    
    for (int i = 0; i < dim; ) {
        size_t vl = __riscv_vsetvl_e32m8(dim - i);
        
        vfloat32m8_t vq = __riscv_vle32_v_f32m8(q + i, vl);
        vfloat32m8_t vk = __riscv_vle32_v_f32m8(k + i, vl);
        
        // Accumulate product into vsum
        vsum = __riscv_vfmacc_vf_f32m8(vsum, 1.0f,
               __riscv_vfmul_vv_f32m8(vq, vk, vl), vl);
        
        i += vl;
    }
    
    // Final reduction to scalar
    size_t vl_reduce = __riscv_vsetvlmax_e32m8();
    vacc = __riscv_vfredusum_vs_f32m8_f32m1(vsum, vacc, vl_reduce);
    result = __riscv_vfmv_f_s_f32m1_f32(vacc);
    
    return result;
}

int main() {
    float Q[SEQ_LEN][D_K];
    float K[SEQ_LEN][D_K];
    float scores[SEQ_LEN];
    float attn_weights[SEQ_LEN];

    // Initialize input matrices
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < D_K; j++) {
            Q[i][j] = (float)(i + j) * 0.01f;
            K[i][j] = (float)(i - j) * 0.01f;
        }
    
    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    
    // Compute attention scores
    for (int j = 0; j < SEQ_LEN; j++)
        scores[j] = dot_product_vectorized(Q[0], K[j], D_K);

    // Apply softmax
    softmax_row(scores, attn_weights, SEQ_LEN);
    
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif
    
    // Print first 4 attention weights for verification
    for (int i = 0; i < 4; i++)
        printf("attn[%d] = %.6f\n", i, attn_weights[i]);

    return 0;
}