#pragma once

typedef enum { COV_SPHERICAL = 1, COV_DIAG = 2, COV_FULL = 3 } gmm_cov_t;

typedef struct {
    int D, K;
    gmm_cov_t cov;
    float *log_weights; /* K */
    float *means;       /* K x D */
    float *prec;        /* spherical: K, diag: K*D, full: K*D*D */
} gmm_ctx_t;

void  gmm_init(gmm_ctx_t *ctx, int D, int K, gmm_cov_t cov);
float gmm_score(const gmm_ctx_t *ctx, const float *x);
void  gmm_free(gmm_ctx_t *ctx);
