#include "gmm.h"
#include <stdlib.h>
#include <math.h>
#include "esp_heap_caps.h"

#define ALLOC(n) heap_caps_malloc((n), MALLOC_CAP_INTERNAL)

static float frand(void)     { return (float)rand() / (float)RAND_MAX - 0.5f; }
static float frand_pos(void) { return 0.5f + (float)rand() / (float)RAND_MAX; }

void gmm_init(gmm_ctx_t *ctx, int D, int K, gmm_cov_t cov)
{
    ctx->D = D;
    ctx->K = K;
    ctx->cov = cov;
    ctx->log_weights = ALLOC(sizeof(float) * K);
    ctx->means       = ALLOC(sizeof(float) * K * D);

    int prec_n = (cov == COV_SPHERICAL) ? K :
                 (cov == COV_DIAG)      ? K * D : K * D * D;
    ctx->prec = ALLOC(sizeof(float) * prec_n);

    for (int k = 0; k < K; k++) {
        ctx->log_weights[k] = frand();
        for (int j = 0; j < D; j++) ctx->means[k * D + j] = frand();
    }
    for (int i = 0; i < prec_n; i++)
        ctx->prec[i] = (cov == COV_FULL) ? frand() : frand_pos();
}

float gmm_score(const gmm_ctx_t *ctx, const float *x)
{
    int D = ctx->D, K = ctx->K;
    float *log_probs = __builtin_alloca(sizeof(float) * K);
    float *delta     = __builtin_alloca(sizeof(float) * D);
    float *pd        = __builtin_alloca(sizeof(float) * D);

    for (int k = 0; k < K; k++) {
        const float *mu = &ctx->means[k * D];
        float maha2 = 0.0f;

        if (ctx->cov == COV_SPHERICAL) {
            float s = 0.0f;
            for (int j = 0; j < D; j++) { float d = x[j] - mu[j]; s += d * d; }
            maha2 = s * ctx->prec[k];
        } else if (ctx->cov == COV_DIAG) {
            const float *p = &ctx->prec[k * D];
            for (int j = 0; j < D; j++) { float d = x[j] - mu[j]; maha2 += p[j] * d * d; }
        } else {
            const float *P = &ctx->prec[k * D * D];
            for (int j = 0; j < D; j++) delta[j] = x[j] - mu[j];
            for (int i = 0; i < D; i++) {
                float s = 0.0f;
                for (int j = 0; j < D; j++) s += P[i * D + j] * delta[j];
                pd[i] = s;
            }
            for (int i = 0; i < D; i++) maha2 += delta[i] * pd[i];
        }
        log_probs[k] = ctx->log_weights[k] - 0.5f * maha2;
    }

    float m = log_probs[0];
    for (int k = 1; k < K; k++) if (log_probs[k] > m) m = log_probs[k];
    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += expf(log_probs[k] - m);
    return -(m + logf(sum));
}

void gmm_free(gmm_ctx_t *ctx)
{
    free(ctx->log_weights);
    free(ctx->means);
    free(ctx->prec);
}
