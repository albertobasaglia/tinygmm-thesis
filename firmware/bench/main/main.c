#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_cpu.h"
#include "esp_private/esp_clk.h"

#include "config.h"
#include "small_ae.h"
#include "gmm.h"
#include "knn.h"
#include "prototype.h"
#include "cosine.h"

static volatile float sink;

static void fill_rand(float *buf, int n)
{
    for (int i = 0; i < n; i++) buf[i] = (float)rand() / (float)RAND_MAX - 0.5f;
}

typedef float (*score_fn)(void *ctx, const float *x);

static portMUX_TYPE bench_mux = portMUX_INITIALIZER_UNLOCKED;

/* Buffered output. The ROM-level console path drops bytes under load
   (UART poll + USB-Serial/JTAG secondary, no driver installed), so we
   collect each result line in RAM and print them all after compute is
   done, with deliberate pacing per line. */
#define MAX_RESULTS 64
static char results[MAX_RESULTS][96];
static int  n_results = 0;

static void bench(const char *tag, void *ctx, score_fn fn, const float *query, int D)
{
    /* Warmup also estimates per-inference cost so we can size the timed
       chunk to stay under BENCH_WINDOW_US. */
    int64_t w0 = esp_timer_get_time();
    for (int i = 0; i < BENCH_WARMUP; i++)
        sink = fn(ctx, query);
    int64_t w1 = esp_timer_get_time();

    int est_us = (int)((w1 - w0) / BENCH_WARMUP);
    if (est_us < 1) est_us = 1;

    int chunk = BENCH_WINDOW_US / est_us;
    if (chunk < 1)          chunk = 1;
    if (chunk > BENCH_REPS) chunk = BENCH_REPS;

    /* Track the min per-inference cost across all chunks and trials. */
    double best_cyc_per = 1e18;
    double best_us_per  = 1e18;

    for (int trial = 0; trial < BENCH_TRIALS; trial++) {
        for (int done = 0; done < BENCH_REPS; done += chunk) {
            int n = (BENCH_REPS - done < chunk) ? BENCH_REPS - done : chunk;

            /* Disable interrupts on this core for the timed window so the
               100 Hz tick ISR (and any other interrupt) can't land in the
               cycle count. The chunk is sized to stay under one tick
               period and well under the interrupt watchdog limit. */
            taskENTER_CRITICAL(&bench_mux);

            int64_t  t0 = esp_timer_get_time();
            uint32_t c0 = esp_cpu_get_cycle_count();

            for (int i = 0; i < n; i++)
                sink = fn(ctx, query);

            uint32_t c1 = esp_cpu_get_cycle_count();
            int64_t  t1 = esp_timer_get_time();

            taskEXIT_CRITICAL(&bench_mux);

            double cyc_per = (double)(c1 - c0) / (double)n;
            double us_per  = (double)(t1 - t0) / (double)n;
            if (cyc_per < best_cyc_per) best_cyc_per = cyc_per;
            if (us_per  < best_us_per)  best_us_per  = us_per;
        }
        /* Let IDLE0 run so the task watchdog gets fed between trials. */
        vTaskDelay(1);
    }

    if (n_results < MAX_RESULTS) {
        snprintf(results[n_results], sizeof(results[0]),
                 "result %-36s  us/inf=%8.2f  cyc/inf=%10.1f  sink=%f",
                 tag, best_us_per, best_cyc_per, (double)sink);
        n_results++;
    }
}

/* ---- One-shot clock diagnostics (see CALIBRATION.md, steps 1-3) ----
   Answers, in one boot: what frequency the core actually runs at, which of
   the two counters (CCOUNT / esp_timer) is honest, and whether the measured
   per-MAC throughput of the benchmark is physically possible. All loops load
   their operands through volatile per iteration so -ffast-math cannot
   constant-fold or final-value-replace them. */
static void clock_diagnostics(void)
{
    /* Step 1: what does the clock tree claim? */
    printf("diag: cpu=%d MHz xtal=%d MHz apb=%d Hz\n",
           esp_clk_cpu_freq() / 1000000,
           esp_clk_xtal_freq() / 1000000,
           esp_clk_apb_freq());
    fflush(stdout);

    /* Step 2a: CCOUNT rate vs host wall clock. Watch the spacing of these
       lines with `idf.py monitor --timestamps`:
         ~6.25 s apart -> CCOUNT at 160 MHz
         ~25 s   apart -> CCOUNT at 40 MHz (XTAL)
         ~4.17 s apart -> CCOUNT at 240 MHz
       vTaskDelay keeps the watchdogs fed; CCOUNT keeps counting while idle
       because PM/tickless idle are disabled. */
    printf("diag: ccount ticks, 1e9 cycles each...\n");
    fflush(stdout);
    for (int i = 0; i < 3; i++) {
        uint32_t c0 = esp_cpu_get_cycle_count();
        while (esp_cpu_get_cycle_count() - c0 < 1000000000u)
            vTaskDelay(1);
        printf("diag: ccount tick %d\n", i + 1);
        fflush(stdout);
    }

    /* Step 2b: esp_timer vs host wall clock. These lines must be 5.0 s
       apart on the host; any other spacing means esp_timer is misscaled. */
    for (int i = 0; i < 2; i++) {
        int64_t t0 = esp_timer_get_time();
        while (esp_timer_get_time() - t0 < 5000000)
            vTaskDelay(1);
        printf("diag: esp_timer tick %d, 5 s nominal\n", i + 1);
        fflush(stdout);
    }

    /* Step 3a: dependent FP-add chain. Each iteration is a volatile load
       plus an add that depends on the previous one, so it cannot retire
       faster than 1 cycle/iter on any single-issue core; realistically
       ~4-5 cyc/iter given FP add latency. If either counter implies
       < 1 cyc/iter, that counter (or the build) is broken. */
    {
        volatile float one = 1.0f;
        float a = 0.0f;
        int64_t  t0 = esp_timer_get_time();
        uint32_t c0 = esp_cpu_get_cycle_count();
        for (int i = 0; i < 10000000; i++)
            a = a + one;
        uint32_t c1 = esp_cpu_get_cycle_count();
        int64_t  t1 = esp_timer_get_time();
        sink = a;
        printf("diag: dep-add 1e7 iters: cyc=%lu us=%lld cyc/iter=%.3f implied_mhz=%.1f\n",
               (unsigned long)(c1 - c0), (long long)(t1 - t0),
               (double)(c1 - c0) / 1e7,
               (double)(c1 - c0) / (double)(t1 - t0));
        fflush(stdout);
    }

    /* Step 3b: independent-MAC throughput ceiling. x evolves each iteration
       (dependent add) so nothing folds; the 4 accumulators are independent
       and pipeline freely. Per iteration: 1 load + 1 add + 4 MACs. The
       implied cyc/MAC here is the best this core+compiler can do; compare it
       with what the bench results imply (e.g. GMM full K=3 D=32 implied
       0.17 cyc/MAC from output.txt -- impossible if this ceiling is ~1). */
    {
        volatile float dv = 1e-7f, yv = 0.999999f;
        float d = dv, y = yv, x = 1.0f;
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int64_t  t0 = esp_timer_get_time();
        uint32_t c0 = esp_cpu_get_cycle_count();
        for (int i = 0; i < 10000000; i++) {
            x += d;
            s0 += x * y; s1 += x * y; s2 += x * y; s3 += x * y;
        }
        uint32_t c1 = esp_cpu_get_cycle_count();
        int64_t  t1 = esp_timer_get_time();
        sink = s0 + s1 + s2 + s3;
        printf("diag: indep-madd 4e7 MACs: cyc=%lu us=%lld cyc/mac=%.3f implied_mhz=%.1f\n",
               (unsigned long)(c1 - c0), (long long)(t1 - t0),
               (double)(c1 - c0) / 4e7,
               (double)(c1 - c0) / (double)(t1 - t0));
        fflush(stdout);
    }

    printf("diag: done\n");
    fflush(stdout);
}

/* -- Typed wrappers so we can pass a uniform score_fn. -- */
static float wrap_ae(void *ctx, const float *x)     { return small_ae_score(ctx, x); }
static float wrap_gmm(void *ctx, const float *x)    { return gmm_score(ctx, x); }
static float wrap_knn(void *ctx, const float *x)    { return knn_score(ctx, x); }
static float wrap_proto(void *ctx, const float *x)  { return proto_score(ctx, x); }
static float wrap_cosine(void *ctx, const float *x) { return cosine_score(ctx, x); }

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

    clock_diagnostics();

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

    /* --- Prototype (single mean, Euclidean distance) --- */
    for (int di = 0; di < N_DIMS; di++) {
        int D = DIMS[di];
        srand(BENCH_SEED);
        proto_ctx_t ctx;
        proto_init(&ctx, D);
        fill_rand(query, D);
        char tag[64];
        snprintf(tag, sizeof(tag), "prototype D=%d", D);
        bench(tag, &ctx, wrap_proto, query, D);
        proto_free(&ctx);
    }

    /* --- Cosine (single mean, 1 - cosine similarity) --- */
    for (int di = 0; di < N_DIMS; di++) {
        int D = DIMS[di];
        srand(BENCH_SEED);
        cosine_ctx_t ctx;
        cosine_init(&ctx, D);
        fill_rand(query, D);
        char tag[64];
        snprintf(tag, sizeof(tag), "cosine D=%d", D);
        bench(tag, &ctx, wrap_cosine, query, D);
        cosine_free(&ctx);
    }

    /* Pacing chosen so each line clears the UART (~7 ms at 115200 baud
       for ~80 chars) and the USB-Serial/JTAG host endpoint has time to
       poll. The [i/N] prefix lets you spot any drops at a glance. */
    printf("== printing %d results ==\n", n_results);
    fflush(stdout);
    vTaskDelay(pdMS_TO_TICKS(50));
    for (int i = 0; i < n_results; i++) {
        printf("[%02d/%02d] %s\n", i + 1, n_results, results[i]);
        fflush(stdout);
        vTaskDelay(pdMS_TO_TICKS(50));
    }
    printf("== done ==\n");
    fflush(stdout);
    for (;;) vTaskDelay(pdMS_TO_TICKS(1000));
}
