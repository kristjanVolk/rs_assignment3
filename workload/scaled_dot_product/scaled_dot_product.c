    #include <stdio.h>
    #include <math.h>
    #include <stdlib.h>
    #include <gem5/m5ops.h>


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

    float dot_product(float *q, float *k, int dim) {
        float result = 0.0f;
        for (int i = 0; i < dim; i++) {
            result += q[i] * k[i];
        }
        return result;
    }

    int main() {
        float Q[SEQ_LEN][D_K];
        float K[SEQ_LEN][D_K];
        float scores[SEQ_LEN];
        float attn_weights[SEQ_LEN];

        for (int i = 0; i < SEQ_LEN; i++)
            for (int j = 0; j < D_K; j++) {
                Q[i][j] = (float)(i + j) * 0.01f;
                K[i][j] = (float)(i - j) * 0.01f;
            }
        
        #ifdef GEM5
            m5_reset_stats(0, 0);
        #endif
        for (int j = 0; j < SEQ_LEN; j++)
            scores[j] = dot_product(Q[0], K[j], D_K);

        softmax_row(scores, attn_weights, SEQ_LEN);
        
        #ifdef GEM5
            m5_dump_stats(0, 0);
        #endif
        
        for (int i = 0; i < 4; i++)
            printf("attn[%d] = %.6f\n", i, attn_weights[i]);

        return 0;
    }