#pragma once

typedef struct {
    int D;
    float *proto;       /* D */
    float  proto_norm;  /* cached ||proto|| */
} cosine_ctx_t;

void  cosine_init(cosine_ctx_t *ctx, int D);
float cosine_score(const cosine_ctx_t *ctx, const float *x);
void  cosine_free(cosine_ctx_t *ctx);
