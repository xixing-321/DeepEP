# MegaMoE BF16代码组织

公开入口为 `SymmBuffer + bf16_mega_moe`。量化推理、BF16推理及BF16训练使用同一个包、Host扩展和运行时管理。

| 层 | 文件/目录 | 职责 |
| --- | --- | --- |
| Python | api.py | 共同buffer、输入字段与公开接口 |
| BF16调用 | _bf16.py、_autograd.py | 校验、Plan缓存及每次调用独立的反向状态 |
| Host | csrc/forward.cpp、csrc/backward.cpp | 前反向校验、分配和启动 |
| Plan | csrc/plan.hpp、csrc/plan_bf16.cpp | 借用根通信窗口，按配置准备基础 tiling 和前向 workspace。每次调用复制 tiling 并填入权重布局 |
| Launch | csrc/kernel_launch.hpp | Host 和设备共用的启动符号声明 |
| 前向 | mega_moe_wave_bf16.hpp | 推理使用模式 0（Wave 交错）。训练默认模式 0，编译时可选模式 1（分阶段）。 |
| 反向 | mega_moe_bf16_backward.hpp | dY分发、dHp、激活梯度、dX回传及dW |
| stage | 路由、peer映射、通信、Unpermute及ready | 数据移动和完成条件 |
| catlass_gmm | mega_moe_bf16_catlass_impl.hpp及_backward.hpp | BF16 GEMM与布局供数 |
| epilogue | BF16激活、保存和梯度 | 共用适用的FP32激活数学，保留各自数学模式 |
| common | 保存及workspace布局 | Host分配与Device寻址共用字节偏移公式 |

## 共用机制

- Wave预算复用 `csrc/kernels/common/mega_moe_wave_math.hpp`。各路径传入自身tile数及预算，保持原调度量。
- BF16与量化SwiGLU复用 `epilogue/activation/swiglu_math.hpp`。BF16保留准确除法，量化路径保留默认除法。
- 前乘及后乘Unpermute复用 `stage/route_accumulate.hpp`，保持TopK slot顺序、首项初始化及FP32累加。
- peer配额、行区间及映射使用共同工具；量化token/scale与BF16 token/p保留各自payload布局。
- BF16反向workspace由 `common/mega_moe_workspace_backward.hpp` 同时供Host和Device使用。

## 路由与payload边界

BF16 与量化直发路径共用按专家提取 route 的逻辑，同一专家内按原 token、TopK slot 顺序排列。量化路径把列表发到对端。BF16 在源卡根据专家计数计算前缀，建立排序行与原始 slot 的双向映射，再按排序行打包 X。训练将同行的 TopK weight（p）发给专家，在 GMM2 前乘权；推理将 p 留在源卡，在 Unpermute 时乘权。

前向按 Wave 从多个源 rank 取行，供 GMM1 和 SwiGLU 连续计算。供反向使用的数据统一按本地专家、源 EP rank、源内 token/TopK slot 排列。同一行的 X、p、Gate 和 Up 对应同一条 route。

## 状态与生命周期

一个槽可容纳一个 Wave 的全部专家切片。接收容量超过 8 个 Wave 时，Dispatch 的临时 X 和 GMM2 输出共用 8 个循环槽；可借用的 Y 空间计入这批槽。路由表、供前向末尾转序的 p 和供反向使用的张量按逻辑行寻址。

| 调度 | Hp 的容量 | 槽何时可覆盖 |
|---|---|---|
| 模式 0（交错） | 与临时 X 相同的行数，独立存放 | 同槽上一轮的所有 Combine 发送结束后，Dispatch 才写下一轮。 |
| 模式 1（分阶段） | 完整接收容量 | GMM1 阶段，上一轮的整行 Hp 就绪后即可覆盖 X。所有 SwiGLU tile 完成后，GMM2 接管这些槽，每次回绕等待上轮 Combine 发送结束。 |

SHMEM 窗口由根 Runtime 持有。BF16 Plan 在 prepare 或首次调用时准备前向临时 workspace，后续前向可以复用。训练中，每个需要反向的 microbatch 由自己的 autograd context 持有 counts 和反向状态。

`save_dispatched_x=True`（默认）时，前向保存规范行序的 X 和 p。`save_dispatched_x=False` 时，前向记录本次调用的 `dispatch_rebuilder`，反向通过它取得相同行序的 X/p。两种来源都可用于 `bf16`、`mxfp8` 和 `none` 三种 GMM1 保存模式。

调用流与 Runtime 流通过事件衔接，异步使用的张量通过 `record_stream` 登记生命周期。框架异步重建 X/p 时，回调须让调用流等待重建完成。

## GMM1 保存与反向

| 保存类型 | 反向如何获得 Gate 和 Up |
| --- | --- |
| `bf16` | 读取前向保存的 BF16 值。在整行路径中，一次读取 dHp、Gate、Up，生成 dP、Hp 和 dZ。 |
| `mxfp8` | 按专家内的规范行序读取量化值和比例因子，在寄存器中以 FP32 解码并舍入为 BF16。启用裁剪时，梯度使用同行的裁剪掩码。 |
| `none` | 从本次前向保存或框架重建的 X 中取出每个专家的连续行，按专家重算 GMM1，得到 Gate 和 Up。 |

反向根据保存类型选择调度。`none` 沿用整专家重算和原有反向流水。`bf16` 在 F 为 64 的倍数且 F ≤ 16384 时使用 Wave，`mxfp8` 在支持的 32 列量化分组上使用 Wave。每个专家沿规范行序按 128 行发布 dO 就绪，尾块按实际行数处理。AIC 在本块 dO 就绪后计算 dHp，AIV 在所需列全部就绪后计算 SwiGLUGrad。

AIC 等待 dO 时，可以先算前面已备齐 dO 和 Hp 的专家的一个 dW2 输出 tile。每个 tile 沿完整专家行做归约。算完即检查 dO，恢复 dHp 或继续借用等待时间。后续 dW2 阶段跳过本核已完成的 tile。需要整专家就绪时，各 rank 可设 `MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS=0`，此时 dW2 全部在原阶段计算。

dW1 同样按专家读取连续的 X，与同行的 dZ 计算权重梯度。dP 归约计划使用整个 rank 的路由行数和全部 AIV，保持各分片的归约顺序。专家按源 rank 前缀将 dP 写入源卡独立的 SHMEM 回传区，每条路由占用一个与 TopK weight 同类型的元素。跨卡写入时，发送核回读这一批梯度，逐位确认写入完成后再参加屏障。屏障结束后，源卡按路由索引恢复原来的 TopK slot。
dHp覆盖为Hp前完成本行dP；dX覆盖dO前，等待dHp和dW2的全部读者退出。
