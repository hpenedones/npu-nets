// SPDX-License-Identifier: Apache-2.0
//
// Embed forward kernel:  y = x @ W
//
// Pure matmul, no ReLU, no residual.  Used as the first tile in the
// full-NPU training pipeline to project 784-dim MNIST input down to H.
//
// Memory layout (tiled 8×8 blocks):
//   x: [B × K_EMBED] bf16, tiled as [B/8, K_EMBED/8, 8, 8]
//   w: [K_EMBED × H] bf16, tiled as [K_EMBED/8, H/8, 8, 8]
//   y: [B × H] bf16, tiled as [B/8, H/8, 8, 8]
//
// Compile with:
//   -DDIM_M=<B>  -DDIM_K_EMBED=<input_dim>  -DDIM_H=<hidden_dim>
//   -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16

#define NOCPP

#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 8
#endif
#ifndef DIM_K_EMBED
#define DIM_K_EMBED 784
#endif
#ifndef DIM_H
#define DIM_H 32
#endif

// ── Tiled matmul: C[M×N] = A[M×K] × B[K×N] ──────────────────────────
//
// Same structure as matmul_plain in residual_backward.cc.

template <typename T, unsigned rowA, unsigned colA, unsigned colB>
static inline void
matmul_plain_two_chunks(const T *__restrict pA,
                        const T *__restrict pB0,
                        const T *__restrict pB1,
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
            const T *__restrict pB0_1 = pB0 + j * MMUL::size_B;
            const T *__restrict pB1_1 = pB1 + j * MMUL::size_B;
            MMUL C00(zeros);

            for (unsigned i = 0; i < colA; ++i)
                chess_flatten_loop
            {
                auto A0 = aie::load_v<MMUL::size_A>(pA1);
                pA1 += MMUL::size_A;
                
                auto B0 = (i < colA / 2) ? aie::load_v<MMUL::size_B>(pB0_1) : aie::load_v<MMUL::size_B>(pB1_1);
                if (i < colA / 2) {
                    pB0_1 += MMUL::size_B * colB;
                } else {
                    pB1_1 += MMUL::size_B * colB;
                }
                
                C00.mac(A0, B0);
            }

            aie::store_v(pC1, C00.template to_vector<T>());
            pC1 += MMUL::size_C;
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────

extern "C" {

void embed_forward_bf16(bfloat16 *x, bfloat16 *w0, bfloat16 *w1, bfloat16 *y)
{
    static_assert(DIM_M % 8 == 0, "Batch must be multiple of 8");
    static_assert(DIM_K_EMBED % 8 == 0, "Input dim must be multiple of 8");
    static_assert(DIM_K_EMBED % 16 == 0, "Input dim must be multiple of 16 to split in 2");
    static_assert(DIM_H % 8 == 0, "Hidden dim must be multiple of 8");

    // y[M×H] = x[M×K] @ w[K×H]
    matmul_plain_two_chunks<bfloat16, (DIM_M / 8), (DIM_K_EMBED / 8), (DIM_H / 8)>(x, w0, w1, y);
}

} // extern "C"
