// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include <aie_api/aie.hpp>

#ifndef SGD_LR
#define SGD_LR 0.0005f
#endif

namespace {

constexpr int BATCH = 8;
constexpr int IN_H = 28;
constexpr int IN_W = 28;
constexpr int OUT_H = 14;
constexpr int OUT_W = 14;
constexpr int OUT_C = 4;
constexpr int INPUT_ELEMS = BATCH * IN_H * IN_W;
constexpr int OUTPUT_ELEMS = BATCH * OUT_H * OUT_W * OUT_C;
constexpr int WEIGHT_ELEMS = OUT_C * 3 * 3;

alignas(32) static float dW[WEIGHT_ELEMS];

} // namespace

extern "C" {

void conv1_forward_relu_bf16(bfloat16 *x, bfloat16 *w, bfloat16 *y, bfloat16 *ckpt)
{
    bfloat16 *mask = ckpt + INPUT_ELEMS;

    for (int i = 0; i < INPUT_ELEMS; ++i) {
        ckpt[i] = x[i];
    }

    for (int b = 0; b < BATCH; ++b) {
        int x_batch = b * IN_H * IN_W;
        int y_batch = b * OUT_H * OUT_W * OUT_C;
        for (int oy = 0; oy < OUT_H; ++oy) {
            int iy0 = oy * 2 - 1;
            for (int ox = 0; ox < OUT_W; ++ox) {
                int ix0 = ox * 2 - 1;
                int out_base = y_batch + (oy * OUT_W + ox) * OUT_C;
                for (int oc = 0; oc < OUT_C; ++oc) {
                    const bfloat16 *w_oc = w + oc * 9;
                    float acc = 0.0f;
                    for (int ky = 0; ky < 3; ++ky) {
                        int iy = iy0 + ky;
                        if (iy < 0 || iy >= IN_H) {
                            continue;
                        }
                        int row_base = x_batch + iy * IN_W;
                        for (int kx = 0; kx < 3; ++kx) {
                            int ix = ix0 + kx;
                            if (ix < 0 || ix >= IN_W) {
                                continue;
                            }
                            acc += (float)x[row_base + ix] * (float)w_oc[ky * 3 + kx];
                        }
                    }
                    if (acc > 0.0f) {
                        y[out_base + oc] = (bfloat16)acc;
                        mask[out_base + oc] = (bfloat16)1.0f;
                    } else {
                        y[out_base + oc] = (bfloat16)0.0f;
                        mask[out_base + oc] = (bfloat16)0.0f;
                    }
                }
            }
        }
    }
}

void conv1_backward_update_bf16(bfloat16 *ckpt, bfloat16 *w, bfloat16 *gy)
{
    bfloat16 *x = ckpt;
    bfloat16 *mask = ckpt + INPUT_ELEMS;

    for (int i = 0; i < WEIGHT_ELEMS; ++i) {
        dW[i] = 0.0f;
    }

    for (int b = 0; b < BATCH; ++b) {
        int x_batch = b * IN_H * IN_W;
        int gy_batch = b * OUT_H * OUT_W * OUT_C;
        for (int oy = 0; oy < OUT_H; ++oy) {
            int iy0 = oy * 2 - 1;
            for (int ox = 0; ox < OUT_W; ++ox) {
                int ix0 = ox * 2 - 1;
                int out_base = gy_batch + (oy * OUT_W + ox) * OUT_C;
                for (int oc = 0; oc < OUT_C; ++oc) {
                    float gz = (float)gy[out_base + oc] * (float)mask[out_base + oc];
                    if (gz == 0.0f) {
                        continue;
                    }
                    float *dw_oc = dW + oc * 9;
                    for (int ky = 0; ky < 3; ++ky) {
                        int iy = iy0 + ky;
                        if (iy < 0 || iy >= IN_H) {
                            continue;
                        }
                        int row_base = x_batch + iy * IN_W;
                        for (int kx = 0; kx < 3; ++kx) {
                            int ix = ix0 + kx;
                            if (ix < 0 || ix >= IN_W) {
                                continue;
                            }
                            dw_oc[ky * 3 + kx] += (float)x[row_base + ix] * gz;
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < WEIGHT_ELEMS; ++i) {
        w[i] = (bfloat16)((float)w[i] - SGD_LR * dW[i]);
    }
}

} // extern "C"
