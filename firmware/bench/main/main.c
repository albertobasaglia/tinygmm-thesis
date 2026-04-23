#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_cpu.h"

#include "config.h"
#include "small_ae.h"
#include "gmm.h"
#include "knn.h"

static volatile float sink;

static void fill_rand(float *buf, int n)
{
    for (int i = 0; i < n; i++) buf[i] = (float)rand() / (float)RAND_MAX - 0.5f;
}

typedef float (*score_fn)(void *ctx, const float *x);

static portMUX_TYPE bench_mux = portMUX_INITIALIZER_UNLOCKED;

static void bench(const char *tag, void *ctx, score_fn fn, const float *query, int D)
{
    for (int i = 0; i < BENCH_WARMUP; i++)
        sink = fn(ctx, query);

    uint32_t best_cyc = UINT32_MAX;
    int64_t  best_us  = INT64_MAX;

    for (int trial = 0; trial < BENCH_TRIALS; trial++) {
        /* Disable interrupts on this core for the timed window so the 100 Hz
           tick ISR (and any other interrupt) can't land in the cycle count.
           Keep the window under one tick period (~10 ms). */
        taskENTER_CRITICAL(&bench_mux);

        int64_t  t0 = esp_timer_get_time();
        uint32_t c0 = esp_cpu_get_cycle_count();

        for (int i = 0; i < BENCH_REPS; i++)
            sink = fn(ctx, query);

        uint32_t c1 = esp_cpu_get_cycle_count();
        int64_t  t1 = esp_timer_get_time();

        taskEXIT_CRITICAL(&bench_mux);

        uint32_t dcyc = c1 - c0;
        int64_t  dus  = t1 - t0;
        if (dcyc < best_cyc) best_cyc = dcyc;
        if (dus  < best_us)  best_us  = dus;
    }

    printf("result %-36s  us/inf=%8.2f  cyc/inf=%10.1f  sink=%f\n",
           tag,
           (double)best_us  / (double)BENCH_REPS,
           (double)best_cyc / (double)BENCH_REPS,
           (double)sink);
}

/* -- Typed wrappers so we can pass a uniform score_fn. -- */
static float wrap_ae(void *ctx, const float *x)  { return small_ae_score(ctx, x); }
static float wrap_gmm(void *ctx, const float *x) { return gmm_score(ctx, x); }
static float wrap_knn(void *ctx, const float *x) { return knn_score(ctx, x); }

/* ---- Configuration tables (edit here to change the sweep) ---- */

static const int DIMS[] = {16, 32};
#define N_DIMS (sizeof(DIMS) / sizeof(DIMS[0]))

static const int AE_LATENTS[] = {4, 8};
#define N_AE_L (sizeof(AE_LATENTS) / sizeof(AE_LATENTS[0]))

static const int GMM_KS[] = {1, 2, 3};
#define N_GMM_K (sizeof(GMM_KS) / sizeof(GMM_KS[0]))

static const gmm_cov_t GMM_COVS[] = {COV_SPHERICAL, COV_DIAG, COV_FULL};
static const char *COV_NAMES[]    = {"spherical",    "diag",    "full"};
#define N_GMM_COV (sizeof(GMM_COVS) / sizeof(GMM_COVS[0]))

static const int KNN_NS[] = {10, 50, 100};
#define N_KNN_N (sizeof(KNN_NS) / sizeof(KNN_NS[0]))

static const int KNN_KS[] = {5};
#define N_KNN_K (sizeof(KNN_KS) / sizeof(KNN_KS[0]))

void app_main(void)
{
    printf("== adapter inference benchmark (reps=%d warmup=%d trials=%d, min reported) ==\n",
           BENCH_REPS, BENCH_WARMUP, BENCH_TRIALS);

    float query[64];

    /* --- SmallAE --- */
    for (int di = 0; di < N_DIMS; di++) {
        int D = DIMS[di];
        for (int li = 0; li < N_AE_L; li++) {
            int L = AE_LATENTS[li];
            srand(BENCH_SEED);
            small_ae_ctx_t ctx;
            small_ae_init(&ctx, D, L);
            fill_rand(query, D);
            char tag[64];
            snprintf(tag, sizeof(tag), "small_ae D=%d L=%d", D, L);
            bench(tag, &ctx, wrap_ae, query, D);
            small_ae_free(&ctx);
        }
    }

    /* --- GMM --- */
    for (int di = 0; di < N_DIMS; di++) {
        int D = DIMS[di];
        for (int ki = 0; ki < N_GMM_K; ki++) {
            int K = GMM_KS[ki];
            for (int ci = 0; ci < N_GMM_COV; ci++) {
                srand(BENCH_SEED);
                gmm_ctx_t ctx;
                gmm_init(&ctx, D, K, GMM_COVS[ci]);
                fill_rand(query, D);
                char tag[64];
                snprintf(tag, sizeof(tag), "gmm D=%d K=%d cov=%s", D, K, COV_NAMES[ci]);
                bench(tag, &ctx, wrap_gmm, query, D);
                gmm_free(&ctx);
            }
        }
    }

    /* --- kNN --- */
    for (int di = 0; di < N_DIMS; di++) {
        int D = DIMS[di];
        for (int ni = 0; ni < N_KNN_N; ni++) {
            int N = KNN_NS[ni];
            for (int ki = 0; ki < N_KNN_K; ki++) {
                int K = KNN_KS[ki];
                if (K > N) continue;
                srand(BENCH_SEED);
                knn_ctx_t ctx;
                knn_init(&ctx, D, N, K);
                fill_rand(query, D);
                char tag[64];
                snprintf(tag, sizeof(tag), "knn D=%d N=%d k=%d", D, N, K);
                bench(tag, &ctx, wrap_knn, query, D);
                knn_free(&ctx);
            }
        }
    }

    printf("== done ==\n");
    for (;;) vTaskDelay(pdMS_TO_TICKS(1000));
}
