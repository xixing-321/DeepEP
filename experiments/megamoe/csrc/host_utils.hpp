// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

// Host-side shared helpers for MegaMoE quantized and BF16 paths.
// Device/kernel code must not include this header.

#pragma once

#include <cstdint>
#include <algorithm>
#include <vector>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "tiling/platform/platform_ascendc.h"

namespace deepep::megamoe {

// Remember the allocations made on a producer stream without retaining tensors across microbatches.
// An unpack hook may replace the storage; views into the original allocation still match.
class StreamAllocations {
public:
    void RememberAllocation(const torch::Tensor& tensor)
    {
        const auto& storage = tensor.storage();
        allocations_.erase(std::remove_if(allocations_.begin(), allocations_.end(),
                                          [](const Allocation& a) { return a.storage.expired(); }),
                           allocations_.end());
        allocations_.push_back({storage.getWeakStorageImpl(), storage.data(), storage.data_ptr().get_context(),
                                storage.nbytes(), c10_npu::getCurrentNPUStream(tensor.get_device()).unwrap()});
    }

    void RecordIfForeign(const torch::Tensor& tensor) const
    {
        const auto stream = c10_npu::getCurrentNPUStream(tensor.get_device());
        const auto& storage = tensor.storage();
        for (const auto& a : allocations_) {
            if (a.stream == stream.unwrap() && a.storage.lock().get() == storage.unsafeGetStorageImpl() &&
                a.data == storage.data() && a.context == storage.data_ptr().get_context() &&
                a.bytes == storage.nbytes()) {
                return;
            }
        }
        c10_npu::NPUCachingAllocator::recordStream(storage.data_ptr(), stream);
    }

private:
    struct Allocation {
        c10::weak_intrusive_ptr<c10::StorageImpl> storage;
        const void* data;
        const void* context;
        size_t bytes;
        c10::Stream stream;
    };
    std::vector<Allocation> allocations_;
};

struct PlatformInfo {
    uint32_t aic;
    uint32_t aiv;
    uint32_t ub;
    uint32_t block_dim;
    uint32_t system_workspace;
};

template <typename T, typename U>
constexpr auto CeilDiv(T value, U divisor)
{
    return (value + divisor - 1) / divisor;
}

template <typename T, typename U>
constexpr auto AlignUp(T value, U alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

inline PlatformInfo GetPlatformInfo()
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "failed to get AscendC platform information");
    uint64_t ub = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const uint32_t aic = platform->GetCoreNumAic();
    const uint32_t aiv = platform->GetCoreNumAiv();
    TORCH_CHECK(aic > 0 && aiv > 0 && ub > 0, "invalid A5 platform resources");
    return {aic, aiv, static_cast<uint32_t>(ub), platform->CalcTschBlockDim(aiv, aic, aiv),
            platform->GetLibApiWorkSpaceSize()};
}

inline void CheckNpuContiguous(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void CheckTensor(const torch::Tensor& t, int64_t dims, const char* name)
{
    TORCH_CHECK(t.defined(), name, " must be defined");
    CheckNpuContiguous(t, name);
    TORCH_CHECK(t.dim() == dims, name, " must be ", dims, "D");
}

}  // namespace deepep::megamoe
