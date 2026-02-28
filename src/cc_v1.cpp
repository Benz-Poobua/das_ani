// src/cc_v1.cpp
#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <algorithm>

// Helper macro for checking inputs
#define CHECK_CPU_FLOAT_CONTIGUOUS(x) \
    TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor"); \
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32"); \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous");

torch::Tensor correlation_v1_cpp(
    torch::Tensor signal_1,   // (B, N)
    torch::Tensor signal_2,   // (B, N)
    int64_t M,
    int64_t K,
    int64_t Lfft
) {
    CHECK_CPU_FLOAT_CONTIGUOUS(signal_1);
    CHECK_CPU_FLOAT_CONTIGUOUS(signal_2);

    const int64_t B = signal_1.size(0);
    const int64_t N = signal_1.size(1);
    const int64_t nblocks = (N + K - 1) / K;

    // -------------------------------------------------------------------------
    // 1. ALLOCATE BATCHED BUFFERS ONCE
    //    Shape: (B, nblocks, Lfft)
    //    We trade memory (RAM) for speed. For DAS data, this is usually safe.
    // -------------------------------------------------------------------------
    auto opts = signal_1.options();
    auto x_batch = torch::zeros({B, nblocks, Lfft}, opts);
    auto y_batch = torch::zeros({B, nblocks, Lfft}, opts);

    // -------------------------------------------------------------------------
    // 2. PARALLEL FILL
    //    Use at::parallel_for to fill buffers. This replaces the serial loop.
    //    We iterate over all (batch * block) tasks in parallel.
    // -------------------------------------------------------------------------
    int64_t total_tasks = B * nblocks;

    at::parallel_for(0, total_tasks, 1, [&](int64_t begin, int64_t end) {
        // Accessors for fast, safe index calculation
        // Note: Raw pointers would be marginally faster but riskier with strides.
        // Given the copy_ overhead, accessors are fine.
        
        for (int64_t i = begin; i < end; ++i) {
            int64_t b = i / nblocks;
            int64_t l = i % nblocks;

            int64_t start = l * K;
            int64_t end_idx = std::min(start + K, N);
            int64_t klen = end_idx - start;

            // --- Fill Y Batch (Zero Padded in middle) ---
            // y_batch[b, l, M : M+klen] = signal_2[b, start : end_idx]
            {
                auto y_dst = y_batch[b][l].narrow(0, M, klen);
                auto y_src = signal_2[b].narrow(0, start, klen);
                y_dst.copy_(y_src);
            }

            // --- Fill X Batch (Clamped window) ---
            // x_batch[b, l, ...] = signal_1[b, x0:x1]
            int64_t x0 = start - M;
            int64_t x1 = start + K + M;
            int64_t ix0 = std::max<int64_t>(0, x0);
            int64_t ix1 = std::min<int64_t>(N, x1);

            if (ix1 > ix0) {
                int64_t len = ix1 - ix0;
                int64_t dst0 = ix0 - x0;
                
                auto x_dst = x_batch[b][l].narrow(0, dst0, len);
                auto x_src = signal_1[b].narrow(0, ix0, len);
                x_dst.copy_(x_src);
            }
        }
    });

    // -------------------------------------------------------------------------
    // 3. BATCHED FFT & ACCUMULATION
    //    This is where the speedup happens: MKL processes the entire 3D tensor.
    // -------------------------------------------------------------------------
    
    // RFFT on the last dimension (Lfft)
    // Input: (B, nblocks, Lfft) -> Output: (B, nblocks, Lfft/2 + 1)
    auto X = torch::fft::rfft(x_batch, Lfft, -1);
    auto Y = torch::fft::rfft(y_batch, Lfft, -1);

    // Cross-Correlate and Sum over Blocks (dim=1) in one pass
    // Rspec = sum( X.conj() * Y, dim=1 )
    // Shape: (B, Lfft/2 + 1)
    auto Rspec = torch::sum(X.conj() * Y, 1);

    // -------------------------------------------------------------------------
    // 4. INVERSE FFT & SLICING
    // -------------------------------------------------------------------------
    auto r = torch::fft::irfft(Rspec, Lfft, -1);

    // Reorder lags
    auto neg_lags = r.narrow(1, Lfft - M, M);
    auto pos_lags = r.narrow(1, 0, M + 1);

    return torch::cat({neg_lags, pos_lags}, 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("correlation_v1", &correlation_v1_cpp, "ANI V1 Correlation Optimized (CPU)");
}