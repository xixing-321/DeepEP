# BF16 训练前向 Wave 实现

Wave 将本 rank 收到的专家输入分成若干批，在批次间交错 Dispatch、GMM1、SwiGLU、GMM2 和 Combine。
GMM1 的 Gate/Up 和 GMM2 的输出都写 GM，计算与通信通过完成事件和 ready 计数交接。
Python 调用、SO 名称和反向接口沿用[训练组件](README.md)的约定。

## 适用条件

设备入口 `UseWavePath()` 在以下条件同时成立时选择 Wave：

- 使用 packed 训练路由布局和分离的训练保存状态。
- `save_gmm1_output` 解析后的实际保存方式为 BF16；`auto` 按裁剪配置解析后再判断。
- SwiGLU 选择 UB 路径，UB 和路由元数据预算检查通过，AIC 数量不超过 36。
- 构建时没有启用 `DEEPEP_BF16_FORCE_EXPERT_READY`。

其余配置使用既有前向路径。BF16/FP32 TopK weight 通过模板沿用输入类型，Wave 不增加调用参数。
反向、保存模式和裁剪的对外约定见训练 README。

## Wave、M-group 与专家 slice

GMM1 和 GMM2 的 M 块均为 192 行，N 块为 256 列。一个 M-group 最多包含同一专家的 192 行，
专家尾块不足 192 行时按实际行数处理。Wave 可以包含多个专家的 M-group，也可以切分一个大专家。
一个专家与一个 Wave 的交集称为专家 slice。

每 Wave 的组数由 AIC 数量和两次矩阵乘的 N 方向任务数决定：

```text
G1_N = ceil(2F / 256)
G2_N = ceil(H / 256)
groups_per_wave = max(ceil(4 × AIC / G1_N), ceil(AIC / G2_N))
```

例如 AIC=32、H7168、F3072：G1_N=24、G2_N=28，每 Wave 为 6 组，最多 1152 行。
专家尾块和最后一个 Wave 的有效行数可以更少。

GMM1 和 GMM2 均使用 `<3,0>` swizzle，各自维护任务起始核游标，AIV0 的 SwiGLU 游标与 GMM1 一致。
同一个 AIC 复用持久的矩阵乘对象及 L1/L0 资源，连续完成其分配到的 GMM1 任务后即可转入 GMM2；
GMM2 按所需 Hp 行块就绪情况等待。

## 三类执行单元的分工

| 执行单元 | 职责 | 消费条件 |
| --- | --- | --- |
| AIC | GMM1、GMM2，结果经 FIX 写 GM | GMM1 等待专家 slice 的 Dispatch 完成；GMM2 等待对应 M192 行块的全部 Hp 列块完成 |
| AIV0 | 从 GM 读取 Gate/Up，计算 SwiGLU、乘路由权重并写出 BF16 Hp | 对应 Gate/Up 的真实 FIX 完成事件到达 |
| AIV1 | Dispatch、GMM2 完成通知接收、Combine | Combine 等待当前专家 slice 的实际生产核全部完成输出卸载 |

当每源 M≥512 时，AIC 让 GMM2 滞后一个 Wave：

```text
GMM1(W0) → GMM1(W1) → GMM2(W0) → GMM1(W2) → GMM2(W1) → … → GMM2(最后一批)
```

每源 M<512 时，同一 Wave 的 GMM1 后紧接 GMM2。AIV0 独立推进 SwiGLU。
AIV1 启动时先 Dispatch 最多两个 Wave，随后根据 Combine 就绪情况和交替优先级推进发送或下一个 Dispatch。
这是一套就绪调度，不要求固定执行一次 Dispatch 后立即执行一次 Combine。

## GM 交接与 ready 协议

### Dispatch → GMM1

路由重排后，源 rank 的同专家 token 段在 SHMEM 中连续存放。AIV1 将当前 Wave 的行范围分给各工作核，
再投影到专家及源 rank 的连续段，搬到本地 packed X，并搬运对应的 TopK weight。
每个工作核确认本 slice 的 X 和权重都写完后增加 Dispatch ready；没有数据交集的工作核也要完成计数。
GMM1 等待该 slice 的计数达到 AIC 数量后读取输入。

### GMM1 → SwiGLU → GMM2

GMM1 将 BF16 Gate/Up 写到各 packed 行对应的保存区。两份完成信用允许下一对 Gate/Up 的计算
与前一对的 SwiGLU 重叠，并约束事件复用。AIV0 等待真实 FIX 完成事件，读取 GM，完成 SwiGLU、
乘权和 Hp 写回后归还信用。

Hp 使用按 M192 行块记录的列块计数。GMM2 仅在该行块的全部 `ceil(F/256)` 个列块完成后读取 Hp。
Gate/Up、Hp 和 GMM2 输出的 GM 地址按 packed 行定位。

### GMM2 → Combine

GMM2 将输出写到独立的 packed GM 区域。每个实际参与该 slice 的生产核发送一次完成通知：

```text
AIC 下发通知并更新 issued
→ FIX 真正完成后 FULL 到达
→ AIV1 接收 FULL，增加 slice ready
→ AIV1 归还 EMPTY，允许复用通知信用
```

`issued` 表示通知已下发。AIV1 必须收到 FULL 后才能发布 ready，不能只凭 issued 判断输出已写完。
参与核数为 `min(ceil(slice_rows/192) × ceil(H/256), AIC)`，空任务核不贡献 GMM2 完成计数。

Combine 仍按整个专家 slice 就绪触发，再逐行将结果发送回源 rank。发送循环每下发一行，
穿插调用 `PollDownCompletion()`，接收后续完成通知并归还 EMPTY，避免等当前 slice 全部发送完才服务通知。
GM 输出在本次前向内具有独立地址；归还 EMPTY 释放的是通知信用，不会让待发送数据被下一批覆盖。

## Dispatch 双槽预取与 UB 布局

Dispatch 保留原有的每槽行数预算。设 `S=align_up(H,32)`，当前计算为：

```text
rows_per_slot = floor((248 KiB − 192 KiB − 4 KiB − 4 KiB) / (4 × S))
```

式中的 192 KiB 用来保持既有批量预算，实际 Dispatch 缓冲起点由 `DispatchUbBase(H)` 计算。
H7168 时每槽一行，两个 BF16 槽各 14 KiB。提前预取改变命令下发顺序：

```text
读取行 0 → 读取行 1 → 下发行 0 写回 → 读取行 2 → 下发行 1 写回 → …
```

读取前等待目标槽上次写回完成，写回前等待对应读取完成。循环结束后写回最后一批并排空两个槽，
随后搬运 TopK weight。这里的行批量与 GMM 的 M192 块、Wave 的组数分别控制不同层次的粒度。

H7168 下 AIV1 的 UB 分区如下：

| 区域 | 字节范围 | 用途 |
| --- | --- | --- |
| Combine 两槽 | 0～28 KiB | 两个完整 token 行的发送中转 |
| Dispatch 两槽 | 28～56 KiB | 两个完整 token 行的接收中转 |
| TopK weight | 56～60 KiB | 按输入 dtype 搬运权重 |
| 路由元数据 | 244～248 KiB | 专家累计计数和源端行偏移 |

Dispatch 从两行 Combine 缓冲之后开始放置，避免沿用 GMM2 旧 UB 输出布局留下的空洞。
UB 搬运缓冲与 metadata 不重叠。

## 工作区与源码入口

Wave 控制区包括 Hp、Dispatch、GMM2 完成计数和各核 issued 字。
专用 GMM2 输出区按 `max_recv_tokens × H × 2` 字节计算，并对齐控制区与输出区边界。
`max_recv_tokens=24576、H7168` 时，该输出区为 336 MiB。Host 在原有工作区需求与 Wave 布局需求之间
取足够的大小；总占用还包括已有布局、保存状态和其他 Tensor。

| 文件 | 职责 |
| --- | --- |
| [tiling.cpp](csrc/bf16/tiling.cpp) | 计算每 Wave 的组数 |
| [training.cpp](csrc/bf16/training.cpp) | Host 输入检查与 transient 工作区分配 |
| [forward_wave_plan.hpp](csrc/kernels/bf16/schedule/forward_wave_plan.hpp) | Wave/slice 遍历、控制区和输出区布局、UB 预算 |
| [forward_wave_completion.hpp](csrc/kernels/bf16/schedule/forward_wave_completion.hpp) | Gate/Up 与 GMM2 的完成信用交接 |
| [forward_wave_matmul.hpp](csrc/kernels/bf16/catlass/forward_wave_matmul.hpp) | 持久矩阵乘对象、GMM1/GMM2 调度及 GM 卸载 |
| [mega_moe_bf16_forward.hpp](csrc/kernels/bf16/mega_moe_bf16_forward.hpp) | AIC/AIV0/AIV1 主循环、Dispatch、SwiGLU 与 Combine |
