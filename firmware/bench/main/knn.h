#pragma once

typedef struct {
    int D, N, K;
    float *store;  /* N x D */
} knn_ctx_t;

void  knn_init(knn_ctx_t *ctx, int D, int N, int K);
float knn_score(const knn_ctx_t *ctx, const float *x);
void  knn_free(knn_ctx_t *ctx);
