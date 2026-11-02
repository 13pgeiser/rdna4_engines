#pragma once

// Real-to-complex FFT backed by pocketfft (BSD-3-Clause, see
// pocketfft/LICENSE.md). This is the only place pocketfft is included so its
// configuration macros stay identical across translation units.
//
// Transforms are single-threaded; callers parallelize at a higher level.
#define POCKETFFT_NO_MULTITHREADING
// Cache plans so repeated per-frame calls reuse the twiddle tables.
#define POCKETFFT_CACHE_SIZE 16
#include "pocketfft/pocketfft_hdronly.hpp"

#include <complex>
#include <cstddef>

namespace audioproc {

    // Forward real-to-complex FFT over a batch of contiguous frames.
    // in:  num_frames rows of fft_length floats
    // out: num_frames rows of (fft_length / 2 + 1) complex bins, interleaved (re, im)
    // Unnormalized (no 1/N scaling).
    inline void rfft_many(const float* in, float* out, size_t num_frames, size_t fft_length) {
        const size_t n_bins = fft_length / 2 + 1;
        const pocketfft::shape_t shape{ num_frames, fft_length };
        const pocketfft::stride_t stride_in{
            static_cast<std::ptrdiff_t>(fft_length * sizeof(float)),
            static_cast<std::ptrdiff_t>(sizeof(float)) };
        const pocketfft::stride_t stride_out{
            static_cast<std::ptrdiff_t>(n_bins * sizeof(std::complex<float>)),
            static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)) };
        pocketfft::r2c<float>(shape, stride_in, stride_out, 1, pocketfft::FORWARD,
            in, reinterpret_cast<std::complex<float>*>(out), 1.0f);
    }

} // namespace audioproc
