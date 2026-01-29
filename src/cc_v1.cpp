// src/cc_v1.cpp
#include <torch/extension.h>
#include <vector>
#include <cmath>

// Helper to confirm inputs are contiguous/correct device
void check_input(const torch::Tensor& x) {
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.device().is_cpu(), "Input tensor must be on CPU for this optimized CPP path");
}

torch::Tensor correlation_v1_cpp(
    torch::Tensor signal_1,   // (B, N)
    torch::Tensor signal_2,   // (B, N)
    int64_t M,                // max_lag_samples
    int64_t K,                // block_size (calculated in Python)
    int64_t Lfft              // fft_length (calculated in Python)
){
    check_input(signal_1);
    check_input(signal_2);

    auto B = signal_1.size(0);
    auto N = signal_1.size(1);

    // Output accumulator in frequency domain (Complex)
    // Size: (B, Lfft/2 + 1)
    auto nfreq = Lfft / 2 + 1;

    // Initialize accumulator with zeros
    auto options_complex = torch::TensorOptions().dtype(torch::kComplexFloat).device(signal_1.device());
    auto Rspec = torch::zeros({B, nfreq}, options_complex);

    // Pre-allocate time-domain buffers to avoid re-malloc inside loop
    // These will hold the zero-padded segments
    auto options_float = torch::TensorOptions().dtype(signal_1.dtype()).device(signal_1.device());
    auto x_t = torch::zeros({B, Lfft}, options_float);
    auto y_t = torch::zeros({B, Lfft}, options_float);

    // Slice helpers
    using namespace torch::indexing;

    // Calculate number of blocks
    // integer division ceiling: (N + K - 1) / K
    int64_t nblocks = (N + K - 1) / K;

    for (int64_t l=0; l < nblocks; ++l) {
        int64_t start = l * K;
        int64_t end = std::min(start + K, N);
        int64_t klen = end - start;

        // 1. Zero out buffers (reuse memory)
        x_t.zero_();
        y_t.zero_();

        // 2. Fill Y buffer (center block)
        // y_t[:, M : M + klen] = signal_2[:, start:end]
        y_t.index_put_({Slice(), Slice(M, M + klen)}, signal_2.index({Slice(), Slice(start, end)}));
   
        // 3. Fill X buffer (wider block with overlaps)
        int64_t x0 = start - M;
        int64_t x1 = start + K + M;
        
        // Clip to signal boundaries [0, N]
        int64_t ix0 = std::max((int64_t)0, x0);
        int64_t ix1 = std::min(N, x1);

        if (ix1 > ix0) {
            // Determine placement in x_t buffer
            int64_t dst0 = ix0 - x0;  
            int64_t dst1 = dst0 + (ix1 - ix0);  

            x_t.index_put_({Slice(), Slice(dst0, dst1)}, signal_1.index({Slice(), Slice(ix0, ix1)}));    
        }

        // 4. FFT
        auto X = torch::fft::rfft(x_t, Lfft, -1);
        auto Y = torch::fft::rfft(y_t, Lfft, -1);

        // 5. Accumulate: Rspec += conj(X) * Y
        Rspec.addcmul_(X.conj(), Y);
    }

    // Inverse FFT
    auto r = torch::fft::irfft(Rspec, Lfft, -1);

    // Circular shift / Reorder lags
    // Result is in [0...M] (pos lags) and [L-M...L] (neg lags)
    // We want standard linear order: [-M ... 0 ... +M]
    // The negative lags are at the END of r.

    auto neg_lags = r.index({Slice(), Slice(Lfft - M, Lfft)}); // Last M samples
    auto pos_lags = r.index({Slice(), Slice(0, M + 1)});       // First M+1 samples
    
    return torch::cat({neg_lags, pos_lags}, -1);
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("correlation_v1", &correlation_v1_cpp, "ANI V1 Correlation (CPU)");
}
