// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include <torch/extension.h>
#include "plan.hpp"
#include "shmem.h"
#include "kernels/mega_moe/common/ascend_timer_v2.hpp"
#include "kernels/mega_moe/common/mega_moe_constants.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_C, module)
{
    module.attr("api_protocol") = 6;
    module.attr("quantized_enabled") = true;
    using deepep::megamoe::Plan;
    module.attr("small_batch_max_tokens") = MegaMoeImpl::W4A8_SMALL_BATCH_MAX_TOKENS;
    module.def("buffer_size", [](int64_t ep, int64_t experts, int64_t tokens, int64_t topk, int64_t hidden) {
        return deepep::megamoe::buffer_size(ep, experts, tokens, topk, hidden);
    });
    module.def("w8a8_training_buffer_size",
               [](int64_t ep, int64_t experts, int64_t tokens, int64_t topk, int64_t hidden) {
                   return deepep::megamoe::buffer_size(ep, experts, tokens, topk, hidden, true);
               });
    using deepep::megamoe::W8A8ForwardState;
    py::class_<W8A8ForwardState>(module, "W8A8ForwardState")
        .def_readonly("counts", &W8A8ForwardState::counts)
        .def_readonly("dispatch", &W8A8ForwardState::dispatch)
        .def_readonly("scales", &W8A8ForwardState::scales)
        .def_readonly("routes", &W8A8ForwardState::routes)
        .def_readonly("gate_up", &W8A8ForwardState::gate_up)
        .def_readonly("timer", &W8A8ForwardState::timer);
    module.def("_inference_max_recv_tokens", &deepep::megamoe::InferenceMaxRecvTokens);
    py::class_<Plan>(module, "Plan")
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, uint64_t,
                      uint64_t, int64_t>())
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, uint64_t,
                      uint64_t, int64_t, bool, int64_t>())
        .def("forward", &Plan::forward)
        .def("training_forward", &Plan::training_forward)
        .def_property_readonly("workspace_bytes", &Plan::workspace_bytes);
    using deepep::megamoe::W8A8BackwardPlan;
    module.def("w8a8_backward_buffer_size", &deepep::megamoe::w8a8_backward_buffer_size);
    py::class_<W8A8BackwardPlan>(module, "W8A8BackwardPlan")
        .def(py::init<uintptr_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, const torch::Tensor&,
                      uint64_t>())
        .def("backward", &W8A8BackwardPlan::backward);
    module.attr("bf16_enabled") = true;
    module.def("bf16_buffer_size", &deepep::megamoe::bf16::buffer_size, py::arg("max_tokens"), py::arg("hidden"),
               py::arg("intermediate"), py::arg("topk"), py::arg("world"), py::arg("local_experts"),
               py::arg("max_recv"), py::arg("need_grad") = true);
    using deepep::megamoe::bf16::BackwardResult;
    using deepep::megamoe::bf16::BF16Plan;
    using deepep::megamoe::bf16::ForwardResult;
    py::class_<ForwardResult>(module, "ForwardResult")
        .def_readonly("y", &ForwardResult::y)
        .def_readonly("counts", &ForwardResult::counts)
        .def_readonly("saved_state", &ForwardResult::saved_state)
        .def_readonly("dispatch_output", &ForwardResult::dispatch_output)
        .def_readonly("timer", &ForwardResult::timer);
    py::class_<BackwardResult>(module, "BackwardResult")
        .def_readonly("dx", &BackwardResult::dx)
        .def_readonly("dtopk_weights", &BackwardResult::dtopk_weights)
        .def_readonly("dw1", &BackwardResult::dw1)
        .def_readonly("dw2", &BackwardResult::dw2)
        .def_readonly("timer", &BackwardResult::timer);
    py::class_<BF16Plan>(module, "BF16Plan")
        .def(py::init<uintptr_t, int64_t, int, int, torch::Tensor, bool>())
        .def("prepare", &BF16Plan::prepare)
        .def_property_readonly("workspace_bytes", &BF16Plan::workspace_bytes)
        .def("forward", &BF16Plan::forward, py::arg("out"), py::arg("x"), py::arg("topk_ids"), py::arg("topk_weights"),
             py::arg("weight1"), py::arg("weight2"), py::arg("max_recv_tokens"), py::arg("swiglu_limit"),
             py::arg("save_gmm1_output"), py::arg("capture_timer"), py::arg("save_dispatched_x") = true)
        .def("backward", &BF16Plan::backward)
        .def("close", &BF16Plan::close);
    module.def("ffts_config", [] { return shmemx_get_ffts_config(); });
    module.attr("timer_numel") = AscendTimer::TOTAL_BUFFER_SIZE;
#ifdef DEEPEP_MEGAMOE_TIMER
    module.attr("timer_enabled") = true;
#else
    module.attr("timer_enabled") = false;
#endif
}
