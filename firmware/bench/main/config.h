#pragma once

/* Harness knobs */
#ifndef BENCH_REPS
#define BENCH_REPS        1000
#endif

#ifndef BENCH_WARMUP
#define BENCH_WARMUP      50
#endif

#ifndef BENCH_SEED
#define BENCH_SEED        42
#endif

/* Number of independent trials per config; reported cyc/us is the min across
   trials, which collapses both interrupt jitter and heap-placement jitter. */
#ifndef BENCH_TRIALS
#define BENCH_TRIALS      5
#endif
