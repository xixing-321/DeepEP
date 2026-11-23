// Copyright (c) 2026, Lu Lu

#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#endif
#include "basic_api/kernel_basic_intf.h"
#include "kernel_operator.h"
#include "kernel_launch.hpp"
#include "common/mega_moe_w8a8_backward_params.hpp"

#if defined(__NPU_DEVICE__)
#include "shmem.h"
#include "mega_moe_w8a8_backward.hpp"
#endif

__schedmode__(1) __global__
    __mix__(1, 2) void mega_moe_w8a8_backward(MegaMoeTrainW8A8::TrainParams params, uint64_t ffts, GM_ADDR timerAddr)
{
#if defined(__NPU_DEVICE__)
    AscendC::InitSocState();
    util_set_ffts_config(ffts);
    MegaMoeTrainTimerDevice timer;
    timer.Init(timerAddr);
    timer.Start(MegaMoeTrainTimer::KERNEL_TIMING);
    auto* capture = timer.Enabled() ? &timer : nullptr;
    if constexpr (g_coreType == AscendC::AIV) {
        AscendC::TPipe pipe;
        MegaMoeTrainW8A8::FusedTrain(params, capture).Backward();
    } else {
        MegaMoeTrainW8A8::FusedTrain(params, capture).Backward();
    }
    timer.End(MegaMoeTrainTimer::KERNEL_TIMING);
    timer.FlushDynamicCounts();
#endif
}

extern "C" __attribute__((visibility("default"))) void mega_moe_w8a8_backward_do(
    uint32_t blocks, uint32_t ub_bytes, void* stream, const MegaMoeTrainW8A8::TrainParams* params, uint64_t ffts,
    uint8_t* timer)
{
    mega_moe_w8a8_backward<<<blocks, ub_bytes, stream>>>(*params, ffts, timer);
}
