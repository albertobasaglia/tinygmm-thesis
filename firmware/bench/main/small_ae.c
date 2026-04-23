#include "small_ae.h"
#include <stdlib.h>
#include "esp_heap_caps.h"

#define ALLOC(n) heap_caps_malloc((n), MALLOC_CAP_INTERNAL)

static float frand(void) { return (float)rand() / (float)RAND_MAX - 0.5f; }
static inline float relu(float v) { return v > 0.0f ? v : 0.0f; }

void small_ae_init(small_ae_ctx_t *ctx, int D, int L)
{
    ctx->D = D;
    ctx->L = L;
    ctx->W1  = ALLOC(sizeof(float) * L * D);
    ctx->b1  = ALLOC(sizeof(float) * L);
    ctx->W2  = ALLOC(sizeof(float) * D * L);
    ctx->b2  = ALLOC(sizeof(float) * D);
    ctx->h   = ALLOC(sizeof(float) * L);
    ctx->rec = ALLOC(sizeof(float) * D);

    for (int i = 0; i < L * D; i++) ctx->W1[i] = frand();
    for (int i = 0; i < L;     i++) ctx->b1[i] = frand();
    for (int i = 0; i < D * L; i++) ctx->W2[i] = frand();
    for (int i = 0; i < D;     i++) ctx->b2[i] = frand();
}

float small_ae_score(const small_ae_ctx_t *ctx, const float *x)
{
    int D = ctx->D, L = ctx->L;
    float *h = ctx->h, *rec = ctx->rec;

    for (int i = 0; i < L; i++) {
        float s = ctx->b1[i];
        for (int j = 0; j < D; j++) s += ctx->W1[i * D + j] * x[j];
        h[i] = relu(s);
    }
    for (int i = 0; i < D; i++) {
        float s = ctx->b2[i];
        for (int j = 0; j < L; j++) s += ctx->W2[i * L + j] * h[j];
        rec[i] = s;
    }
    float acc = 0.0f;
    for (int i = 0; i < D; i++) {
        float d = x[i] - rec[i];
        acc += d * d;
    }
    return acc / (float)D;
}

void small_ae_free(small_ae_ctx_t *ctx)
{
    free(ctx->W1); free(ctx->b1);
    free(ctx->W2); free(ctx->b2);
    free(ctx->h);  free(ctx->rec);
}
