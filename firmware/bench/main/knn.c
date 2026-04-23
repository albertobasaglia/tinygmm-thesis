#include "knn.h"
#include <stdlib.h>
#include "esp_heap_caps.h"

#define ALLOC(n) heap_caps_malloc((n), MALLOC_CAP_INTERNAL)

static float frand(void) { return (float)rand() / (float)RAND_MAX - 0.5f; }

static void sift_down(float *h, int n, int i)
{
    for (;;) {
        int l = 2 * i + 1, r = 2 * i + 2, m = i;
        if (l < n && h[l] > h[m]) m = l;
        if (r < n && h[r] > h[m]) m = r;
        if (m == i) return;
        float t = h[i]; h[i] = h[m]; h[m] = t;
        i = m;
    }
}

void knn_init(knn_ctx_t *ctx, int D, int N, int K)
{
    ctx->D = D;
    ctx->N = N;
    ctx->K = K;
    ctx->store = ALLOC(sizeof(float) * N * D);
    for (int i = 0; i < N * D; i++) ctx->store[i] = frand();
}

float knn_score(const knn_ctx_t *ctx, const float *x)
{
    int D = ctx->D, N = ctx->N, K = ctx->K;
    float *heap = __builtin_alloca(sizeof(float) * K);

    for (int i = 0; i < K; i++) {
        const float *p = &ctx->store[i * D];
        float s = 0.0f;
        for (int j = 0; j < D; j++) { float d = x[j] - p[j]; s += d * d; }
        heap[i] = s;
    }
    for (int i = K / 2 - 1; i >= 0; i--) sift_down(heap, K, i);

    for (int i = K; i < N; i++) {
        const float *p = &ctx->store[i * D];
        float s = 0.0f;
        for (int j = 0; j < D; j++) { float d = x[j] - p[j]; s += d * d; }
        if (s < heap[0]) {
            heap[0] = s;
            sift_down(heap, K, 0);
        }
    }
    return heap[0];
}

void knn_free(knn_ctx_t *ctx)
{
    free(ctx->store);
}
