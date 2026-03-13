// SPDX-License-Identifier: Apache-2.0
//
// Head tile kernel: matmul → softmax → cross-entropy → d_logits
//
// Fuses the classification head forward pass and loss/gradient computation
// into a single kernel so no intermediate buffers leave the tile.
//
// Forward:   logits[B×N_CLS] = y_hidden[B×H] @ w_head[H×N_CLS]
//            probs = softmax(logits, dim=1)
//            loss  = -mean(log(probs[labels]))
//
// Backward:  d_logits = probs - one_hot(labels)
//            (This is the gradient of CE loss w.r.t. logits, divided by B)
//
// Memory layout (tiled 8×8 blocks, N_CLS padded to multiple of 8):
//   y_hidden:  [B × H] bf16
//   w_head:    [H × N_CLS] bf16
//   labels:    [B] int32
//   d_logits:  [B × N_CLS] bf16 (output)
//   loss_out:  [1] float (output, for host logging)
//
// Compile with:
//   -DDIM_M=<B>  -DDIM_H=<hidden_dim>  -DDIM_N_CLS=<padded_num_classes>
//   -DNUM_CLASSES=<actual_classes>
//   -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16

#define NOCPP

#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 8
#endif
#ifndef DIM_H
#define DIM_H 32
#endif
#ifndef DIM_N_CLS
#define DIM_N_CLS 16    // padded to multiple of 8
#endif
#ifndef NUM_CLASSES
#define NUM_CLASSES 10  // actual number of classes
#endif

// ── Tiled matmul (same as matmul_plain) ───────────────────────────────

template <typename T, unsigned rowA, unsigned colA, unsigned colB>
static inline void
matmul_plain(const T *__restrict pA,
             const T *__restrict pB,
             T *__restrict pC)
{
    constexpr unsigned r = 8, s = 8, t = 8;
    using MMUL = aie::mmul<r, s, t, T, T, accauto>;
    const auto zeros = aie::zeros<accfloat, MMUL::size_C>();

    for (unsigned z = 0; z < rowA; ++z) {
        T *__restrict pC1 = pC + z * colB * MMUL::size_C;

        for (unsigned j = 0; j < colB; ++j)
            chess_prepare_for_pipelining chess_loop_range(3, )
        {
            const T *__restrict pA1 = pA + z * colA * MMUL::size_A;
            const T *__restrict pB1 = pB + j * MMUL::size_B;
            MMUL C00(zeros);

            for (unsigned i = 0; i < colA; ++i)
                chess_flatten_loop
            {
                auto A0 = aie::load_v<MMUL::size_A>(pA1);
                pA1 += MMUL::size_A;
                auto B0 = aie::load_v<MMUL::size_B>(pB1);
                pB1 += MMUL::size_B * colB;
                C00.mac(A0, B0);
            }

            aie::store_v(pC1, C00.template to_vector<T>());
            pC1 += MMUL::size_C;
        }
    }
}

// ── Un-tile a small [B×N] tiled buffer into row-major ─────────────────
// tiled layout: [B/8, N/8, 8, 8] → row-major [B, N]

static inline void
untile(const bfloat16 *tiled, float *row_major, int B, int N)
{
    int br = 8, bc = 8;
    for (int bi = 0; bi < B / br; ++bi) {
        for (int bj = 0; bj < N / bc; ++bj) {
            for (int r = 0; r < br; ++r) {
                for (int c = 0; c < bc; ++c) {
                    int tiled_idx = (bi * (N / bc) + bj) * br * bc + r * bc + c;
                    int rm_idx = (bi * br + r) * N + (bj * bc + c);
                    row_major[rm_idx] = (float)tiled[tiled_idx];
                }
            }
        }
    }
}

// ── Re-tile a small [B×N] row-major buffer back to tiled layout ──────

static inline void
retile(const float *row_major, bfloat16 *tiled, int B, int N)
{
    int br = 8, bc = 8;
    for (int bi = 0; bi < B / br; ++bi) {
        for (int bj = 0; bj < N / bc; ++bj) {
            for (int r = 0; r < br; ++r) {
                for (int c = 0; c < bc; ++c) {
                    int tiled_idx = (bi * (N / bc) + bj) * br * bc + r * bc + c;
                    int rm_idx = (bi * br + r) * N + (bj * bc + c);
                    tiled[tiled_idx] = (bfloat16)row_major[rm_idx];
                }
            }
        }
    }
}

extern "C" {

void head_forward_loss_bf16(bfloat16 *y_hidden, bfloat16 *w_head,
                             int32_t *labels, bfloat16 *d_logits,
                             float *loss_out)
{
    static_assert(DIM_M % 8 == 0);
    static_assert(DIM_H % 8 == 0);
    static_assert(DIM_N_CLS % 8 == 0);

    // Scratch space for logits in tiled layout, then row-major probs
    // logits: B × N_CLS bf16 tiled
    alignas(32) bfloat16 logits_tiled[DIM_M * DIM_N_CLS];

    // Step 1: logits = y_hidden @ w_head
    matmul_plain<bfloat16, (DIM_M / 8), (DIM_H / 8), (DIM_N_CLS / 8)>(
        y_hidden, w_head, logits_tiled);

    // Step 2: Un-tile logits for softmax (scalar math, needs row-major)
    float logits_rm[DIM_M * DIM_N_CLS];
    untile(logits_tiled, logits_rm, DIM_M, DIM_N_CLS);

    // Step 3: Softmax + cross-entropy loss + d_logits (all in row-major)
    float d_logits_rm[DIM_M * DIM_N_CLS];
    float total_loss = 0.0f;

    for (int b = 0; b < DIM_M; ++b) {
        float *row = logits_rm + b * DIM_N_CLS;

        // Find max for numerical stability (only over actual classes)
        float max_val = row[0];
        for (int c = 1; c < NUM_CLASSES; ++c) {
            if (row[c] > max_val) max_val = row[c];
        }

        // Exp and sum
        float exp_vals[DIM_N_CLS];
        float sum_exp = 0.0f;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            exp_vals[c] = __builtin_expf(row[c] - max_val);
            sum_exp += exp_vals[c];
        }
        // Padded classes get zero probability
        for (int c = NUM_CLASSES; c < DIM_N_CLS; ++c) {
            exp_vals[c] = 0.0f;
        }

        // Softmax probabilities
        float inv_sum = 1.0f / sum_exp;
        for (int c = 0; c < DIM_N_CLS; ++c) {
            exp_vals[c] *= inv_sum;
        }

        // Cross-entropy loss: -log(prob[label])
        int label = labels[b];
        float prob_label = exp_vals[label];
        // Clamp to avoid log(0)
        if (prob_label < 1e-7f) prob_label = 1e-7f;
        total_loss += -__builtin_logf(prob_label);

        // d_logits = (softmax - one_hot) / B
        float scale = 1.0f / (float)DIM_M;
        for (int c = 0; c < DIM_N_CLS; ++c) {
            float one_hot = (c == label) ? 1.0f : 0.0f;
            d_logits_rm[b * DIM_N_CLS + c] = (exp_vals[c] - one_hot) * scale;
        }
    }

    // Average loss
    *loss_out = total_loss / (float)DIM_M;

    // Step 4: Re-tile d_logits back to bf16 tiled layout
    retile(d_logits_rm, d_logits, DIM_M, DIM_N_CLS);
}

} // extern "C"
