# Adapter Inference Benchmark

ESP-IDF app that measures per-inference cost (cycles and microseconds) of the
three candidate adaptive layers on an ESP32-S3, so the thesis can compare
measured silicon performance against the theoretical FLOPs formulas.

## Quick start

```bash
cd firmware/bench
idf.py set-target esp32s3
idf.py build flash monitor
```

Output goes to the serial monitor. Each line is one config:

```
result gmm D=32 K=3 cov=full       us/inf=   13.36  cyc/inf=     534.3  sink=-1.576541
```

`sink` is the last returned score, printed so the compiler can't dead-code-
eliminate the call and so runs can be diffed for numerical regressions.

## What is measured

Five adapters, all operating on a random query embedding:

| Adapter   | Source             | Score function                               |
|-----------|--------------------|----------------------------------------------|
| small_ae  | [main/small_ae.c](main/small_ae.c) | MSE between input and MLP reconstruction     |
| gmm       | [main/gmm.c](main/gmm.c)           | Negative log-likelihood (sph / diag / full)  |
| knn       | [main/knn.c](main/knn.c)           | Squared distance to k-th nearest stored point |
| prototype | [main/prototype.c](main/prototype.c) | Euclidean distance to single enrollment mean |
| cosine    | [main/cosine.c](main/cosine.c)     | 1 − cosine similarity to single enrollment mean |

Weights, means, precisions, and the k-NN store are filled with a deterministic
PRNG (`srand(BENCH_SEED)` before each `_init`). **Accuracy is not evaluated
here** -- only inference cost. This isolates the per-call compute pattern from
any training or data-loading concerns.

## Measurement protocol

Each config is benchmarked by `bench()` in [main/main.c](main/main.c):

1. **Warmup**: run the score function `BENCH_WARMUP` (50) times, discarded.
   Primes the instruction/data cache and branch predictor.
2. **Trials**: repeat `BENCH_TRIALS` (5) independent timed passes.
3. **Timed pass**: `taskENTER_CRITICAL` -> read `esp_cpu_get_cycle_count` and
   `esp_timer_get_time` -> run the score function `BENCH_REPS` (1000) times ->
   read the counters again -> `taskEXIT_CRITICAL`.
4. **Report**: print the **minimum** cyc/us across the five trials, divided by
   `BENCH_REPS`.

Knobs live in [main/config.h](main/config.h):

```c
#define BENCH_REPS        1000
#define BENCH_WARMUP      50
#define BENCH_SEED        42
#define BENCH_TRIALS      5
```

### Why these choices

- **Critical section around the timed loop.** The default 100 Hz FreeRTOS tick
  ISR would otherwise steal a few hundred cycles per tick at random, adding
  roughly 0-40% noise on the slowest configs. `taskENTER_CRITICAL` masks
  interrupts on this core and holds a spinlock so the other core cannot
  interfere either. Timed windows stay under one 10 ms tick period, so no tick
  is missed outright.
- **Min across trials** (rather than mean). Both interrupt jitter and
  heap-placement jitter can only *add* cycles -- they never subtract any. The
  minimum therefore converges to the "clean" inference cost as trials grow.
  This also collapses run-to-run variance caused by ESP-IDF startup
  non-determinism placing the adapter's `heap_caps_malloc` blocks at different
  addresses each boot.
- **`heap_caps_malloc(..., MALLOC_CAP_INTERNAL)`** keeps all weights/store
  buffers in on-chip SRAM (not PSRAM). Per-call scratch inside `gmm_score` and
  `knn_score` uses `__builtin_alloca` (stack) so it is never part of the
  heap layout.

## Sweep configuration

Tables at the top of `app_main()` in [main/main.c](main/main.c) control what
gets benchmarked. Edit there and rebuild:

```c
static const int DIMS[]        = {16, 32};
static const int AE_LATENTS[]  = {4, 8};
static const int GMM_KS[]      = {1, 2, 3};
static const gmm_cov_t GMM_COVS[] = {COV_SPHERICAL, COV_DIAG, COV_FULL};
static const int KNN_NS[]      = {10, 50, 100};
static const int KNN_KS[]      = {5};
```

Each `D`/`L`/`K`/`N` is independent; the sweep is the cartesian product within
each adapter.

## Compile flags

From [main/CMakeLists.txt](main/CMakeLists.txt):

- `-O2 -ffast-math` applied to all sources in `main/`.
- `-Wno-maybe-uninitialized` on `gmm.c` only: GCC cannot prove that the
  runtime-K loop writes `log_probs[0]` before the subsequent log-sum-exp reads
  it, which it always does because K >= 1.

## Interpreting the output against the FLOPs formulas

For a clean cycles-vs-FLOPs regression:

- The per-component cost of GMM scales linearly with K, and the full-cov
  variant is dominated by one `D x D` matvec per mixture -- the best configs
  for fitting `cycles = alpha * FLOPs + beta`.
- kNN's marginal cost per stored point (slope over N) is the cleanest signal
  because the heap / partial-sort overhead is amortized at large N.
- small_ae includes a fixed per-inference MSE pass of cost O(D); expect a
  non-zero intercept.
