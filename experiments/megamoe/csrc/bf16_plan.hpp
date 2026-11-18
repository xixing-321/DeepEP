// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <torch/extension.h>

struct MegaMoeBf16TilingData;

namespace deepep::megamoe::bf16::ops {

struct ForwardResult {
    torch::Tensor y;
    torch::Tensor counts;
    torch::Tensor saved_state;
    torch::Tensor dispatch_output;
    torch::Tensor timer;
};

struct BackwardResult {
    torch::Tensor dx;
    torch::Tensor dtopk_weights;
    torch::Tensor dw1;
    torch::Tensor dw2;
    torch::Tensor timer;
};

ForwardResult MegaMoeBF16Forward(torch::Tensor context, torch::Tensor x, torch::Tensor topk_ids,
                                 torch::Tensor topk_weights, torch::Tensor weight1, torch::Tensor weight2,
                                 int64_t max_recv_tokens, int rank, int world_size, std::size_t symmetric_buffer_bytes,
                                 uint64_t ffts_addr, void* symmetric_buffer, double swiglu_limit = 0.0,
                                 const std::string& save_gmm1_output = "bf16", torch::Tensor out = {},
                                 torch::Tensor timer = {}, torch::Tensor workspace = {},
                                 bool weight_before_gmm2 = true);

BackwardResult MegaMoeBF16Backward(torch::Tensor context, torch::Tensor grad_y, torch::Tensor topk_weights,
                                   torch::Tensor weight1, torch::Tensor weight2, torch::Tensor expert_token_nums,
                                   torch::Tensor training_saved_state, torch::Tensor dispatch_output,
                                   int64_t max_recv_tokens, int rank, int world_size,
                                   std::size_t symmetric_buffer_bytes, uint64_t ffts_addr, void* symmetric_buffer,
                                   double swiglu_limit = 0.0, const std::string& save_gmm1_output = "bf16",
                                   torch::Tensor timer = {});

}  // namespace deepep::megamoe::bf16::ops

namespace deepep::megamoe::bf16 {

// Borrows the root Session window; Python controls its collective lifetime.
class BF16Plan {
public:
    BF16Plan(uintptr_t address, int64_t bytes, int rank, int world_size, torch::Tensor like, bool weight_before_gmm2);

    void close();
    void prepare(int64_t m, int64_t h, int64_t f, int64_t topk, int64_t local_experts, int64_t max_recv, double limit,
                 const std::string& save_type, bool fp32_p);
    int64_t workspace_bytes() const
    {
        return workspace_.defined() ? workspace_.numel() : 0;
    }

    ops::ForwardResult forward_out(torch::Tensor out, torch::Tensor x, torch::Tensor topk_ids,
                                   torch::Tensor topk_weights, torch::Tensor weight1, torch::Tensor weight2,
                                   int64_t max_recv_tokens, double swiglu_limit, const std::string& save_gmm1_output);

    ops::BackwardResult backward(torch::Tensor grad_y, torch::Tensor topk_weights, torch::Tensor weight1,
                                 torch::Tensor weight2, torch::Tensor counts, torch::Tensor saved_state,
                                 torch::Tensor dispatch_output, int64_t max_recv_tokens, double swiglu_limit,
                                 const std::string& save_gmm1_output);

    ops::ForwardResult forward_with_timer(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
                                          torch::Tensor weight1, torch::Tensor weight2, int64_t max_recv_tokens,
                                          double swiglu_limit, const std::string& save_gmm1_output,
                                          c10::optional<torch::Tensor> out);
    ops::BackwardResult backward_with_timer(torch::Tensor grad_y, torch::Tensor topk_weights, torch::Tensor weight1,
                                            torch::Tensor weight2, torch::Tensor counts, torch::Tensor saved_state,
                                            torch::Tensor dispatch_output, int64_t max_recv_tokens, double swiglu_limit,
                                            const std::string& save_gmm1_output);

private:
    torch::Tensor Context(const torch::Tensor& like);

    void* SymmetricBuffer() const;
    std::size_t SymmetricBufferBytes() const;
    uintptr_t address_;
    int64_t bytes_;
    int rank_;
    int world_size_;
    uint64_t ffts_addr_;
    torch::Tensor context_;
    torch::Tensor workspace_;
    bool weight_before_gmm2_;

    int context_rank_ = -1;
    int context_world_size_ = -1;
};

}  // namespace deepep::megamoe::bf16

namespace deepep::megamoe::bf16 {
int64_t buffer_size(int64_t max_tokens, int64_t hidden, int64_t intermediate, int64_t topk, int64_t world,
                    int64_t local_experts, int64_t max_recv);
}
