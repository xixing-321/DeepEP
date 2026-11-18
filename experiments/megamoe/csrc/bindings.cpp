// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include <torch/extension.h>
#include "forward.hpp"
#include "shmem.h"
#include "kernels/mega_moe/common/ascend_timer_v2.hpp"
#include "kernels/mega_moe/common/mega_moe_constants.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_C, module)
{
    module.attr("api_protocol") = 5;
    module.attr("quantized_enabled") = true;
    using deepep::megamoe::Plan;
    module.attr("small_batch_max_tokens") = MegaMoeImpl::W4A8_SMALL_BATCH_MAX_TOKENS;
    module.def("buffer_size", &deepep::megamoe::buffer_size);
    py::class_<Plan>(module, "Plan")
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, uint64_t,
                      uint64_t, int64_t>())
        .def("forward", &Plan::forward)
        .def_property_readonly("workspace_bytes", &Plan::workspace_bytes);
    module.attr("bf16_enabled") = true;
    module.def("bf16_buffer_size", &deepep::megamoe::bf16::buffer_size);
    using deepep::megamoe::bf16::BF16Plan;
    namespace ops = deepep::megamoe::bf16::ops;
    py::class_<ops::ForwardResult>(module, "ForwardResult")
        .def_readonly("y", &ops::ForwardResult::y)
        .def_readonly("counts", &ops::ForwardResult::counts)
        .def_readonly("saved_state", &ops::ForwardResult::saved_state)
        .def_readonly("dispatch_output", &ops::ForwardResult::dispatch_output)
        .def_readonly("timer", &ops::ForwardResult::timer);
    py::class_<ops::BackwardResult>(module, "BackwardResult")
        .def_readonly("dx", &ops::BackwardResult::dx)
        .def_readonly("dtopk_weights", &ops::BackwardResult::dtopk_weights)
        .def_readonly("dw1", &ops::BackwardResult::dw1)
        .def_readonly("dw2", &ops::BackwardResult::dw2)
        .def_readonly("timer", &ops::BackwardResult::timer);
    py::class_<BF16Plan>(module, "BF16Plan")
        .def(py::init<uintptr_t, int64_t, int, int, torch::Tensor, bool>())
        .def("prepare", &BF16Plan::prepare)
        .def_property_readonly("workspace_bytes", &BF16Plan::workspace_bytes)
        .def("forward_out", &BF16Plan::forward_out)
        .def("backward", &BF16Plan::backward)
        .def("forward_with_timer", &BF16Plan::forward_with_timer)
        .def("backward_with_timer", &BF16Plan::backward_with_timer)
        .def("close", &BF16Plan::close);
    module.def("ffts_config", [] { return shmemx_get_ffts_config(); });
    module.attr("timer_numel") = AscendTimer::TOTAL_BUFFER_SIZE;
#ifdef DEEPEP_MEGAMOE_TIMER
    module.attr("timer_enabled") = true;
#else
    module.attr("timer_enabled") = false;
#endif
}
