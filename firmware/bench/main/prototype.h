#pragma once

typedef struct {
    int D;
    float *proto;  /* D */
} proto_ctx_t;

void  proto_init(proto_ctx_t *ctx, int D);
float proto_score(const proto_ctx_t *ctx, const float *x);
void  proto_free(proto_ctx_t *ctx);
