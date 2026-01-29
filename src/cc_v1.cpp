// src/cc_v1.cpp
#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <algorithm>

static inline void check_input(const torch::Tensor& x) {
    TORCH_CHECK(x.defined(), "Input tensor is undefined");
    TORCH_CHECK(x.device().is_cpu(), "Input tensor must be on CPU for this CPP path");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D (B, N)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32 for this CPP path");
}

torch::Tensor correlation_v1_cpp(
    torch::Tensor signal_1,   // (B, N) float32 CPU contiguous
    torch::Tensor signal_2,   // (B, N) float32 CPU contiguous
    int64_t M,                // max_lag_samples
    int64_t K,                // block_size (calculated in Python)
    int64_t Lfft              // fft_length (calculated in Python)
) {
    check_input(signal_1);
    check_input(signal_2);

    TORCH_CHECK(signal_1.sizes() == signal_2.sizes(), "signal_1 and signal_2 must have same shape");
    TORCH_CHECK(M > 0, "M must be > 0");
    TORCH_CHECK(K > 0, "K must be > 0");
    TORCH_CHECK(Lfft > 0, "Lfft must be > 0");
    TORCH_CHECK(Lfft >= (K + 2 * M), "Lfft must be >= K + 2M");

    const int64_t B = signal_1.size(0);
    const int64_t N = signal_1.size(1);
    const int64_t nfreq = Lfft / 2 + 1;

    auto opts_c = torch::TensorOptions().dtype(torch::kComplexFloat).device(signal_1.device());
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(signal_1.device());

    auto Rspec = torch::zeros({B, nfreq}, opts_c);

    // Allocate once
    auto x_t = torch::zeros({B, Lfft}, opts_f);
    auto y_t = torch::zeros({B, Lfft}, opts_f);

    const int64_t nblocks = (N + K - 1) / K;

    for (int64_t l = 0; l < nblocks; ++l) {
        const int64_t start = l * K;
        const int64_t end   = std::min(start + K, N);
        const int64_t klen  = end - start;

        // Fast + safe on CPU: one contiguous memset each
        x_t.zero_();
        y_t.zero_();

        // y_t[:, M:M+klen] = signal_2[:, start:end]
        {
            auto y_dst = y_t.narrow(/*dim=*/1, /*start=*/M, /*length=*/klen);
            auto y_src = signal_2.narrow(/*dim=*/1, /*start=*/start, /*length=*/klen);
            y_dst.copy_(y_src);
        }

        // x_t gets [start-M : start+K+M] (clipped)
        const int64_t x0 = start - M;
        const int64_t x1 = start + K + M;

        const int64_t ix0 = std::max<int64_t>(0, x0);
        const int64_t ix1 = std::min<int64_t>(N, x1);

        if (ix1 > ix0) {
            const int64_t len  = ix1 - ix0;
            const int64_t dst0 = ix0 - x0;  // offset into x_t

            auto x_dst = x_t.narrow(/*dim=*/1, /*start=*/dst0, /*length=*/len);
            auto x_src = signal_1.narrow(/*dim=*/1, /*start=*/ix0, /*length=*/len);
            x_dst.copy_(x_src);
        }

        // FFT
        auto X = torch::fft::rfft(x_t, /*n=*/Lfft, /*dim=*/-1);
        auto Y = torch::fft::rfft(y_t, /*n=*/Lfft, /*dim=*/-1);

        // Accumulate: Rspec += conj(X) * Y
        Rspec.addcmul_(X.conj(), Y);
    }

    // Inverse FFT
    auto r = torch::fft::irfft(Rspec, /*n=*/Lfft, /*dim=*/-1);

    // Reorder lags to [-M ... 0 ... +M]
    auto neg_lags = r.narrow(/*dim=*/1, /*start=*/Lfft - M, /*length=*/M);
    auto pos_lags = r.narrow(/*dim=*/1, /*start=*/0,        /*length=*/M + 1);

    return torch::cat({neg_lags, pos_lags}, /*dim=*/-1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("correlation_v1", &correlation_v1_cpp, "ANI V1 Correlation (CPU, float32)");
}