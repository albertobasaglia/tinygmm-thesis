#pragma once

typedef struct {
    int D, L;
    float *W1;   /* L x D */
    float *b1;   /* L */
    float *W2;   /* D x L */
    float *b2;   /* D */
    float *h;    /* L scratch */
    float *rec;  /* D scratch */
} small_ae_ctx_t;

void small_ae_init(small_ae_ctx_t *ctx, int D, int L);
float small_ae_score(const small_ae_ctx_t *ctx, const float *x);
void small_ae_free(small_ae_ctx_t *ctx);
