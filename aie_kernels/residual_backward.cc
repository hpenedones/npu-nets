// SPDX-License-Identifier: Apache-2.0
//
// Backward kernels for one residual layer:
//
//   y  = relu(x @ W) + x
//   gz = gy * relu_mask
//   dW = x^T @ gz
//   gx = gy + gz @ W^T
//
// Inputs use the same tiled bf16 layout as the forward kernel.  The backward
// path is intentionally split in two:
//
//   1. residual_grad_input_bf16  : computes gx from [gy | mask] and W^T
//   2. residual_weight_grad_bf16 : computes dW from [x | gy | mask]
//
// This keeps each phase within the tile SRAM budget during phase-0 validation.

#define NOCPP

#ifndef HOST_TEST
#include <aie_api/aie.hpp>
#endif

#ifndef DIM_M
#define DIM_M 8
#endif
#ifndef DIM_K
#define DIM_K 160
#endif
#ifndef DIM_N
#define DIM_N 160
#endif
#ifndef SGD_LR
#define SGD_LR 0.005f
#endif

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

template <typename T, unsigned rowA, unsigned colA, unsigned colB>
static inline void
matmul_transpose_a(const T *__restrict pA,
                   const T *__restrict pB,
                   T *__restrict pC)
{
    constexpr unsigned r = 8, s = 8, t = 8;
    using MMUL = aie::mmul<r, s, t, T, T, accauto>;
    const auto zeros = aie::zeros<accfloat, MMUL::size_C>();
    alignas(32) T a_tile_t[MMUL::size_A];

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
                    pA + (i * rowA + z) * MMUL::size_A;
                const T *__restrict pB_block =
                    pB + (i * colB + j) * MMUL::size_B;

                transpose_tile_8x8(pA_block, a_tile_t);
                auto A0 = aie::load_v<MMUL::size_A>(a_tile_t);
                auto B0 = aie::load_v<MMUL::size_B>(pB_block);
                C00.mac(A0, B0);
            }

            aie::store_v(pC1, C00.template to_vector<T>());
            pC1 += MMUL::size_C;
        }
    }
}

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

static inline void
elementwise_mul(const bfloat16 *__restrict a,
                const bfloat16 *__restrict b,
                bfloat16 *__restrict c)
{
    constexpr int total = DIM_M * DIM_N;
    static_assert(total % 32 == 0, "Total elements must be divisible by 32");

    for (int i = 0; i < total; i += 32) {
        auto va = aie::load_v<32>(a + i);
        auto vb = aie::load_v<32>(b + i);
        auto vc = aie::mul(va, vb).template to_vector<bfloat16>();
        aie::store_v(c + i, vc);
    }
}

static inline void
residual_add(const bfloat16 *__restrict a, bfloat16 *__restrict c)
{
    constexpr int total = DIM_M * DIM_N;
    static_assert(total % 32 == 0, "Total elements must be divisible by 32");

    for (int i = 0; i < total; i += 32) {
        auto vc = aie::load_v<32>(c + i);
        auto va = aie::load_v<32>(a + i);
        aie::store_v(c + i, aie::add(vc, va));
    }
}

extern "C" {

void residual_grad_input_from_ckpt_bf16(bfloat16 *ckpt,
                                        bfloat16 *w,
                                        bfloat16 *gy,
                                        bfloat16 *gx)
{
    static_assert(DIM_M % 8 == 0, "Batch must be multiple of 8");
    static_assert(DIM_K % 8 == 0, "Input dim must be multiple of 8");
    static_assert(DIM_N % 8 == 0, "Output dim must be multiple of 8");

    // Reuse the checkpoint mask buffer to avoid a large stack allocation.
    bfloat16 *gz = ckpt + DIM_M * DIM_K;

    // Step 1: gz = gy * relu_mask (in-place over the saved mask)
    elementwise_mul(gy, gz, gz);

    // Step 2: gx = gz @ W^T using the standard tiled layout of W.
    matmul_transpose_b<bfloat16, (DIM_M / 8), (DIM_N / 8), (DIM_K / 8)>(
        gz, w, gx
    );

    // Step 3: gx += gy   (skip-connection gradient)
    residual_add(gy, gx);
}

void residual_sgd_update_from_gz_bf16(bfloat16 *ckpt, bfloat16 *w)
{
    float lr = SGD_LR;
    static_assert(DIM_M % 8 == 0, "Batch must be multiple of 8");
    static_assert(DIM_K % 8 == 0, "Input dim must be multiple of 8");
    static_assert(DIM_N % 8 == 0, "Output dim must be multiple of 8");

    bfloat16 *x = ckpt;
    bfloat16 *gz = ckpt + DIM_M * DIM_K;

    constexpr unsigned r = 8, s = 8, t = 8;
    using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accauto>;
    const auto zeros = aie::zeros<accfloat, MMUL::size_C>();
    alignas(32) bfloat16 a_tile_t[MMUL::size_A];

    for (unsigned j = 0; j < (DIM_N / 8); ++j) {
        for (unsigned z = 0; z < (DIM_K / 8); ++z) {
            bfloat16 *w_block = w + (z * (DIM_N / 8) + j) * MMUL::size_C;

            MMUL C00(zeros);

            for (unsigned i = 0; i < (DIM_M / 8); ++i)
                chess_flatten_loop
            {
                const bfloat16 *pA_block = x + (i * (DIM_K / 8) + z) * MMUL::size_A;
                const bfloat16 *pB_block = gz + (i * (DIM_N / 8) + j) * MMUL::size_B;

                transpose_tile_8x8<bfloat16>(pA_block, a_tile_t);

                auto A0 = aie::load_v<MMUL::size_A>(a_tile_t);
                auto B0 = aie::load_v<MMUL::size_B>(pB_block);

                C00.mac(A0, B0);
            }

            auto w_vec = aie::load_v<MMUL::size_C>(w_block);
            auto lr_vec = aie::broadcast<bfloat16, 64>(lr);
            auto dw_vec = C00.template to_vector<bfloat16>();
            auto step_vec = aie::mul(lr_vec, dw_vec).template to_vector<bfloat16>();
            auto w_new = aie::sub(w_vec, step_vec);

            aie::store_v(w_block, w_new);
        }
    }
}

void residual_grad_input_bf16(bfloat16 *state,
                              bfloat16 *w_t,
                              bfloat16 *gx)
{
    static_assert(DIM_M % 8 == 0, "Batch must be multiple of 8");
    static_assert(DIM_K % 8 == 0, "Input dim must be multiple of 8");
    static_assert(DIM_N % 8 == 0, "Output dim must be multiple of 8");

    bfloat16 *gy = state;
    bfloat16 *mask = gy + DIM_M * DIM_N;

    alignas(32) bfloat16 gz[DIM_M * DIM_N];

    // Step 1: gz = gy * relu_mask
    elementwise_mul(gy, mask, gz);

    // Step 2: gx = gz @ W^T
    matmul_plain<bfloat16, (DIM_M / 8), (DIM_K / 8), (DIM_N / 8)>(
        gz, w_t, gx
    );

    // Step 3: gx += gy   (skip-connection gradient)
    residual_add(gy, gx);
}

void residual_sgd_update_bf16(bfloat16 *state, bfloat16 *w, float lr)
{
    static_assert(DIM_M % 8 == 0, "Batch must be multiple of 8");
    static_assert(DIM_K % 8 == 0, "Input dim must be multiple of 8");
    static_assert(DIM_N % 8 == 0, "Output dim must be multiple of 8");

    bfloat16 *x = state;
    bfloat16 *gy = state + DIM_M * DIM_K;
    bfloat16 *mask = gy + DIM_M * DIM_N;

    // Use the mask buffer itself to store gz to save 2.5 KB of tile SRAM!
    bfloat16 *gz = mask;

    // Step 1: gz = gy * relu_mask (in-place over mask)
    elementwise_mul(gy, mask, gz);

    // Step 2: W = W - lr * (x^T @ gz)
    // We compute the outer product and accumulate directly into the weights!

    constexpr unsigned r = 8, s = 8, t = 8;
    using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accauto>;
    const auto zeros = aie::zeros<accfloat, MMUL::size_C>();
    alignas(32) bfloat16 a_tile_t[MMUL::size_A];

    // Note: This assumes `w` is in the standard untransposed tiled layout.
    // In our model (H=160), W is 160x160.
    for (unsigned j = 0; j < (DIM_N / 8); ++j) {
        for (unsigned z = 0; z < (DIM_K / 8); ++z) {
            bfloat16 *w_block = w + (z * (DIM_N / 8) + j) * MMUL::size_C;
            
            // Load the current weight block
            // Compute dW for this block in an empty accumulator
            MMUL C00(zeros);

            for (unsigned i = 0; i < (DIM_M / 8); ++i)
                chess_flatten_loop
            {
                const bfloat16 *pA_block = x + (i * (DIM_K / 8) + z) * MMUL::size_A;
                const bfloat16 *pB_block = gz + (i * (DIM_N / 8) + j) * MMUL::size_B;

                // Transpose the activation block x -> x^T
                transpose_tile_8x8<bfloat16>(pA_block, a_tile_t);
                
                auto A0 = aie::load_v<MMUL::size_A>(a_tile_t);
                auto B0 = aie::load_v<MMUL::size_B>(pB_block);
                
                // dW += x^T @ gz
                C00.mac(A0, B0);
            }

            // We have the gradient block dW = C00.
            // We want W = W - lr * dW.
            // Load current W block
            auto w_vec = aie::load_v<MMUL::size_C>(w_block);
            
            // Broadcast learning rate into a 64-element vector to match dW
            auto lr_vec = aie::broadcast<bfloat16, 64>(lr);
            
            // Scaled subtraction: w_vec - lr * dW
            // AIE MAC output is 32-bit float accumulated. We convert dW to float/bfloat vector.
            auto dw_vec = C00.template to_vector<bfloat16>();
            
            // (lr * dW)
            auto step_vec = aie::mul(lr_vec, dw_vec).template to_vector<bfloat16>();
            
            // W - (lr * dW)
            auto w_new = aie::sub(w_vec, step_vec);
            
            // Store it back
            aie::store_v(w_block, w_new);
        }
    }
}

void residual_backward_and_update_bf16(bfloat16 *ckpt, bfloat16 *w, bfloat16 *gy, bfloat16 *gx)
{
    float lr = SGD_LR;
    static_assert(DIM_M % 8 == 0, "Batch must be multiple of 8");
    static_assert(DIM_K % 8 == 0, "Input dim must be multiple of 8");
    static_assert(DIM_N % 8 == 0, "Output dim must be multiple of 8");

    bfloat16 *x = ckpt;
    bfloat16 *mask = ckpt + DIM_M * DIM_K;

    // Reuse mask buffer to store gz and save 2.5KB of SRAM/stack!
    bfloat16 *gz = mask;

    // Step 1: gz = gy * mask
    elementwise_mul(gy, mask, gz);

    // Step 2: gx = gz @ W^T
    matmul_transpose_b<bfloat16, (DIM_M / 8), (DIM_N / 8), (DIM_K / 8)>(
        gz, w, gx
    );

    // Step 3: gx += gy   (skip-connection gradient)
    residual_add(gy, gx);

    // Step 4: W = W - lr * (x^T @ gz)
    // We compute the outer product and accumulate directly into the weights!

    constexpr unsigned r = 8, s = 8, t = 8;
    using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accauto>;
    const auto zeros = aie::zeros<accfloat, MMUL::size_C>();
    alignas(32) bfloat16 a_tile_t[MMUL::size_A];

    for (unsigned j = 0; j < (DIM_N / 8); ++j) {
        for (unsigned z = 0; z < (DIM_K / 8); ++z) {
            bfloat16 *w_block = w + (z * (DIM_N / 8) + j) * MMUL::size_C;
            
            MMUL C00(zeros);

            for (unsigned i = 0; i < (DIM_M / 8); ++i)
                chess_flatten_loop
            {
                const bfloat16 *pA_block = x + (i * (DIM_K / 8) + z) * MMUL::size_A;
                const bfloat16 *pB_block = gz + (i * (DIM_N / 8) + j) * MMUL::size_B;

                transpose_tile_8x8<bfloat16>(pA_block, a_tile_t);
                
                auto A0 = aie::load_v<MMUL::size_A>(a_tile_t);
                auto B0 = aie::load_v<MMUL::size_B>(pB_block);
                
                C00.mac(A0, B0);
            }

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
