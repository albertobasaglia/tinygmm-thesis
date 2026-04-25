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

/* Cap on the timed critical-section window in microseconds. Must stay well
   under one FreeRTOS tick (~10 ms) so the tick ISR can't land in the chunk,
   and well under the interrupt watchdog (default 300 ms) so we don't panic
   on slow configs. We chunk BENCH_REPS into pieces sized for this budget. */
#ifndef BENCH_WINDOW_US
#define BENCH_WINDOW_US   5000
#endif
