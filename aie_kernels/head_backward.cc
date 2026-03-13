// SPDX-License-Identifier: Apache-2.0
//
// Head backward kernel: compute dy_hidden and update W_head via SGD.
//
// Given d_logits from the loss, this kernel:
//   1. dy_hidden[B×H] = d_logits[B×N_CLS] @ w_head[H×N_CLS]^T
//   2. dW_head = y_hidden^T @ d_logits
//   3. w_head -= lr * dW_head    (in-place SGD)
//
// Memory layout (tiled 8×8 blocks):
//   y_hidden:  [B × H] bf16 (checkpointed from forward)
//   w_head:    [H × N_CLS] bf16 (updated in-place)
//   d_logits:  [B × N_CLS] bf16
//   dy_hidden: [B × H] bf16 (output to residual backward chain)
//
// Compile with:
//   -DDIM_M=<B>  -DDIM_H=<hidden_dim>  -DDIM_N_CLS=<padded_num_classes>
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
#define DIM_N_CLS 16
#endif

template <typename T>
static inline void
transpose_tile_8x8(const T *__restrict src, T *__restrict dst)
{
    for (unsigned r = 0; r < 8; ++r) {
        for (unsigned c = 0; c < 8; ++c) {
            dst[c * 8 + r] = src[r * 8 + c];
        }
    }
}

// ── matmul_transpose_b: C[M×K] = A[M×N] @ B[K×N]^T ──────────────────
// (B is stored as [K×N] in tiled layout, we transpose each 8x8 block)

template <typename T, unsigned rowA, unsigned colA, unsigned colB>
static inline void
matmul_transpose_b(const T *__restrict pA,
                   const T *__restrict pB,
                   T *__restrict pC)
{
    constexpr unsigned r = 8, s = 8, t = 8;
    using MMUL = aie::mmul<r, s, t, T, T, accauto>;
    const auto zeros = aie::zeros<accfloat, MMUL::size_C>();
    alignas(32) T b_tile_t[MMUL::size_B];

    for (unsigned z = 0; z < rowA; ++z) {
        T *__restrict pC1 = pC + z * colB * MMUL::size_C;

        for (unsigned j = 0; j < colB; ++j)
            chess_prepare_for_pipelining chess_loop_range(3, )
        {
            MMUL C00(zeros);

            for (unsigned i = 0; i < colA; ++i)
                chess_flatten_loop
            {
                const T *__restrict pA_block =
                    pA + (z * colA + i) * MMUL::size_A;
                const T *__restrict pB_block =
                    pB + (j * colA + i) * MMUL::size_B;

                transpose_tile_8x8(pB_block, b_tile_t);
                auto A0 = aie::load_v<MMUL::size_A>(pA_block);
                auto B0 = aie::load_v<MMUL::size_B>(b_tile_t);
                C00.mac(A0, B0);
            }

            aie::store_v(pC1, C00.template to_vector<T>());
            pC1 += MMUL::size_C;
        }
    }
}

extern "C" {

void head_backward_bf16(bfloat16 *y_hidden, bfloat16 *w_head,
                         bfloat16 *d_logits, bfloat16 *dy_hidden)
{
    float lr = 0.01f;

    static_assert(DIM_M % 8 == 0);
    static_assert(DIM_H % 8 == 0);
    static_assert(DIM_N_CLS % 8 == 0);

    // Step 1: dy_hidden[M×H] = d_logits[M×N_CLS] @ w_head[H×N_CLS]^T
    matmul_transpose_b<bfloat16, (DIM_M / 8), (DIM_N_CLS / 8), (DIM_H / 8)>(
        d_logits, w_head, dy_hidden);

    // Step 2: dW_head = y_hidden^T @ d_logits, then SGD update
    // W_head[H×N_CLS] -= lr * (y_hidden^T[H×M] @ d_logits[M×N_CLS])

    constexpr unsigned r_sz = 8, s_sz = 8, t_sz = 8;
    using MMUL = aie::mmul<r_sz, s_sz, t_sz, bfloat16, bfloat16, accauto>;
    const auto zeros = aie::zeros<accfloat, MMUL::size_C>();
    alignas(32) bfloat16 a_tile_t[MMUL::size_A];

    constexpr unsigned rowsH = DIM_H / 8;
    constexpr unsigned rowsM = DIM_M / 8;
    constexpr unsigned colsN = DIM_N_CLS / 8;

    for (unsigned z = 0; z < rowsH; ++z) {
        for (unsigned j = 0; j < colsN; ++j) {
            bfloat16 *w_block = w_head + (z * colsN + j) * MMUL::size_C;

            MMUL C00(zeros);

            for (unsigned i = 0; i < rowsM; ++i)
                chess_flatten_loop
            {
                // y_hidden is [M/8, H/8, 8, 8]. For y^T block [z,i], read [i,z] and transpose
                const bfloat16 *pA_block = y_hidden + (i * rowsH + z) * MMUL::size_A;
                const bfloat16 *pB_block = d_logits + (i * colsN + j) * MMUL::size_B;

                transpose_tile_8x8<bfloat16>(pA_block, a_tile_t);
                auto A0 = aie::load_v<MMUL::size_A>(a_tile_t);
                auto B0 = aie::load_v<MMUL::size_B>(pB_block);
                C00.mac(A0, B0);
            }

            // SGD: W -= lr * dW
            auto w_vec = aie::load_v<MMUL::size_C>(w_block);
            auto lr_vec = aie::broadcast<bfloat16, 64>(lr);
            auto dw_vec = C00.template to_vector<bfloat16>();
            auto step_vec = aie::mul(lr_vec, dw_vec).template to_vector<bfloat16>();
            auto w_new = aie::sub(w_vec, step_vec);
            aie::store_v(w_block, w_new);
        }
    }
}

} // extern "C"
