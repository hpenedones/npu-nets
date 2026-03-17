// SPDX-License-Identifier: Apache-2.0
//
// Scalar buffer copy kernel for arbitrary bf16 buffer sizes.

#define NOCPP

#include <aie_api/aie.hpp>

#ifndef COPY_TOTAL
#define COPY_TOTAL 32
#endif
#ifndef COPY_BUFFER_NAME
#define COPY_BUFFER_NAME copy_buffer_bf16
#endif

extern "C" {

void COPY_BUFFER_NAME(bfloat16 *in, bfloat16 *out)
{
    for (int i = 0; i < COPY_TOTAL; ++i) {
        out[i] = in[i];
    }
}

} // extern "C"
