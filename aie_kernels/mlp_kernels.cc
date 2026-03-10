// SPDX-License-Identifier: Apache-2.0
//
// In-place ReLU kernel for the spatial MLP pipeline.
// Applies max(x, 0) element-wise on a bfloat16 buffer.

#define NOCPP

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

void relu_inplace_bf16(bfloat16 *__restrict buf, int32_t size)
{
    event0();

    const int v_factor = 32;
    v32bfloat16 zeroes = broadcast_zero_to_v32bfloat16();

    for (int i = 0; i < size; i += v_factor) {
        v32bfloat16 val = *(v32bfloat16 *)(buf + i);
        *(v32bfloat16 *)(buf + i) = max(val, zeroes);
    }

    event1();
}

} // extern "C"

