#include "cosine.h"
#include <stdlib.h>
#include <math.h>
#include "esp_heap_caps.h"

#define ALLOC(n) heap_caps_malloc((n), MALLOC_CAP_INTERNAL)

static float frand(void) { return (float)rand() / (float)RAND_MAX - 0.5f; }

void cosine_init(cosine_ctx_t *ctx, int D)
{
    ctx->D = D;
    ctx->proto = ALLOC(sizeof(float) * D);
    float n = 0.0f;
    for (int j = 0; j < D; j++) { float v = frand(); ctx->proto[j] = v; n += v * v; }
    ctx->proto_norm = sqrtf(n);  /* precomputed at fit time, not per query */
}

/* 1 - cosine similarity to the single enrollment prototype. The prototype
   norm is cached; only the query dot product and norm are paid per call. */
float cosine_score(const cosine_ctx_t *ctx, const float *x)
{
    int D = ctx->D;
    const float *p = ctx->proto;
    float dot = 0.0f, nz = 0.0f;
    for (int j = 0; j < D; j++) { dot += x[j] * p[j]; nz += x[j] * x[j]; }
    float cos = dot / (sqrtf(nz) * ctx->proto_norm + 1e-12f);
    return 1.0f - cos;
}

void cosine_free(cosine_ctx_t *ctx)
{
    free(ctx->proto);
}
