# MegaMoE BF16代码组织

公开入口为 `SymmBuffer + bf16_mega_moe`。量化推理、BF16推理及BF16训练使用同一个包、Host扩展和运行时管理。

| 层 | 文件/目录 | 职责 |
| --- | --- | --- |
| Python | api.py | 共同buffer、输入字段与公开接口 |
| BF16调用 | _bf16.py、_autograd.py | 校验、Plan缓存及每次调用独立的反向状态 |
| Host | csrc/forward_bf16.cpp、backward.cpp | 前反向校验、分配和启动 |
| Plan | csrc/bf16_plan.hpp、bf16_tiling* | 借用根通信窗口，准备临时workspace |
| 前向 | mega_moe_bf16.hpp | 同一实现组织交错及分阶段流程 |
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

量化路径生成compact route表，BF16路径用InitRoutingV3生成原TopK slot映射。
两者共用peer配额、prefix cut和行区间工具。量化token/scale、BF16 token/p各自保留对应载荷布局。
BF16的保存、解码及反向状态由保存策略管理，激活数学和有序归并使用共同基础函数。

## 状态与生命周期

SHMEM窗口属于根Runtime，BF16 Plan只借用地址。prepare或首次调用准备临时workspace。
输入保留原Tensor的梯度连接。每次前向保存独立counts、dispatch数据及反向状态，后续前向可以复用临时区。
buffer锁及caller/runtime stream依赖对调用排序；异步Tensor登记record_stream。

反向保存行映射每route占两个uint32，描述物理行及连续段，不复制整份行序。
BF16整行路径一次读取dHp/Gate/Up，融合dP/Hp/dZ。归约计划使用完整rank的路由行数和全部AIV，不随局部行分工改变。
dHp覆盖为Hp前完成本行dP；dX覆盖dO前，等待dHp和dW2的全部读者退出。
