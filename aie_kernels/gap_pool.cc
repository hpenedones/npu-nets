// SPDX-License-Identifier: Apache-2.0
//
// Global average pooling forward/backward kernels.

#define NOCPP

#include <aie_api/aie.hpp>

#ifndef BATCH_SIZE
#define BATCH_SIZE 8
#endif
#ifndef IN_C
#define IN_C 16
#endif
#ifndef IN_H
#define IN_H 4
#endif
#ifndef IN_W
#define IN_W 4
#endif

extern "C" {

void gap_forward_bf16(bfloat16 *feat_in, bfloat16 *pooled_out)
{
    constexpr int spatial = IN_H * IN_W;

    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int c = 0; c < IN_C; ++c) {
            float acc = 0.0f;
            for (int y = 0; y < IN_H; ++y) {
                for (int x = 0; x < IN_W; ++x) {
                    int idx = ((b * IN_H + y) * IN_W + x) * IN_C + c;
                    acc += (float)feat_in[idx];
                }
            }
            pooled_out[b * IN_C + c] = (bfloat16)(acc / (float)spatial);
        }
    }
}

void gap_backward_bf16(bfloat16 *d_pooled, bfloat16 *grad_feat)
{
    constexpr int spatial = IN_H * IN_W;

    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int y = 0; y < IN_H; ++y) {
            for (int x = 0; x < IN_W; ++x) {
                for (int c = 0; c < IN_C; ++c) {
                    int idx = ((b * IN_H + y) * IN_W + x) * IN_C + c;
                    grad_feat[idx] = (bfloat16)((float)d_pooled[b * IN_C + c] / (float)spatial);
                }
            }
        }
    }
}

} // extern "C"
