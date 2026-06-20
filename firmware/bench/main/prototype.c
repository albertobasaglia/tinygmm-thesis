#include "prototype.h"
#include <stdlib.h>
#include <math.h>
#include "esp_heap_caps.h"

#define ALLOC(n) heap_caps_malloc((n), MALLOC_CAP_INTERNAL)

static float frand(void) { return (float)rand() / (float)RAND_MAX - 0.5f; }

void proto_init(proto_ctx_t *ctx, int D)
{
    ctx->D = D;
    ctx->proto = ALLOC(sizeof(float) * D);
    for (int j = 0; j < D; j++) ctx->proto[j] = frand();
}

/* Euclidean distance to the single enrollment prototype. */
float proto_score(const proto_ctx_t *ctx, const float *x)
{
    int D = ctx->D;
    const float *p = ctx->proto;
    float s = 0.0f;
    for (int j = 0; j < D; j++) { float d = x[j] - p[j]; s += d * d; }
    return sqrtf(s);
}

void proto_free(proto_ctx_t *ctx)
{
    free(ctx->proto);
}
