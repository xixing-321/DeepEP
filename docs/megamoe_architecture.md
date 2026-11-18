<!--
Copyright (c) 2026, Lu Lu
-->

# MegaMoE 架构说明

## 30 秒摘要

MegaMoE 是本仓的 MoE 融合算子工程：把 token 路由、跨卡 Dispatch、分组矩阵乘（GMM1 Gate/Up → SwiGLU → GMM2）、加权归并与 Combine/Unpermute 融合为单个 Ascend NPU kernel 流水，覆盖 W4A8/W8A8 量化推理与 BF16 推理/训练（含 autograd 反向）。产品代码集中在 `experiments/megamoe/`，Python 入口为 `deep_ep_experimental.megamoe`，与 DeepEP 根仓共享 SHMEM 运行时但独立持有对称窗口（SymmBuffer）。

本文件覆盖：MegaMoE 的组件职责、目录导航、一次调用的端到端数据流、训练保存/反向路径、与仓库其余部分的接口边界、构建宏与设计不变量。不覆盖：DeepEP 根仓 HT/LL 通信路径的内部设计（见 [架构与源码组织](architecture.md)）、量化推理的共享专家/小 BS 策略细节与多机测试操作（见 [MegaMoE README](../experiments/megamoe/README.md)）、BF16 共用机制的逐项对照（见 [BF16 设计说明](../experiments/megamoe/docs/bf16_design.md)）。使用方法与参数约束以 README 为准；本文件只描述代码现状的架构。

## 1. 定位

**解决什么问题**：MoE 层的通信（dispatch/combine）与计算（两层专家 GEMM + 激活）若分开执行，会产生多轮跨卡搬运与中间张量读写。MegaMoE 把整条链路融合进一个 `__mix__(1, 2)` kernel（AIC 做 GEMM、AIV 做路由/搬运/激活/归并），用对称内存窗口（SHMEM peermem）直接读写 peer 数据，一次调用完成 y 的原地写入。量化推理（W4A8/W8A8）与 BF16 训练共用同一套 Host 扩展、SymmBuffer 与设备 stage 基础函数。

**在本仓中的位置**：`experiments/megamoe/` 是自包含的产品子树（自带 Python 包、Host 扩展源码、设备 kernel、构建与测试），通过根仓的 `deep_ep` 运行时获得 SHMEM Session 与通信窗口；设备侧复用根仓 `csrc/kernels/common/` 的共享头。整体关系：

```text
                     模型侧 Tensor（x / topk_idx / topk_weights / W1 / W2）
                                       │
   ┌───────────────────────────────────▼───────────────────────────────────┐
   │ experiments/megamoe/python/deep_ep_experimental/megamoe/               │
   │   SymmBuffer（配置+输入字段+Plan 缓存）  fp8_fp4_mega_moe / bf16_mega_moe│
   │   BF16Function autograd（_autograd.py）     _MegaRuntime（专属 stream） │
   └───────────────┬───────────────────────────────────────┬───────────────┘
                   │ megamoe _C（csrc/bindings.cpp）        │ 继承/借用
                   ▼                                       ▼
   ┌───────────────────────────────┐        ┌──────────────────────────────┐
   │ Host：Plan（量化）/ BF16Plan   │        │ deep_ep 根仓                  │
   │  校验、tiling、workspace、启动 │        │  _runtime / _native / _store  │
   └───────────────┬───────────────┘        │  核心 _C：SHMEM Session、     │
                   │                        │  对称窗口地址                  │
                   ▼                        └──────────────────────────────┘
   ┌───────────────────────────────────────────────────────────────────────┐
   │ 设备库 libmegamoe_kernel.so / libdeepep_megamoe_train_bf16.so（Bisheng）│
   │   stage/（路由、Dispatch、Combine、Unpermute、就绪协议）                │
   │   catlass_gmm/（CATLASS GEMM）  epilogue/（SwiGLU、量化、保存）        │
   │   common/（tiling/layout/workspace/context 共用定义）                  │
   └───────────────┬───────────────────────────────────────────────────────┘
                   ▼
        SHMEM 对称窗口（peermem）＋ 本地临时 workspace（Plan 复用）
```

根仓视角下 MegaMoE 只是模块职责表中的一行（`experiments/megamoe/`，见 [架构说明](architecture.md)）；其 `SymmBuffer` 与 ElasticBuffer 分开管理，不能套用 HT/LL 的容量与初始化约束。

## 2. 目录与文件地图

### 顶层

| 路径 | 职责 |
| --- | --- |
| `experiments/megamoe/README.md` | 使用说明：接口签名、权重准备、构建/多机测试、BF16 训练语义 |
| `experiments/megamoe/docs/bf16_design.md` | BF16 代码组织与共用机制说明 |
| `experiments/megamoe/CMakeLists.txt` | MegaMoE 扩展构建：设备库、编译宏、pybind `_C`（`experiments/megamoe/CMakeLists.txt:41`） |
| `experiments/megamoe/cmake/Headers.cmake` | 量化路径 Tensor API 头（CANN 算子包 common 目录）定位 |
| `experiments/megamoe/scripts/` | `build.sh`/`build.py` 一键构建安装；`batch_launch.py`/`batch_node.py` 多机批量测试 |

### Python 包 `python/deep_ep_experimental/megamoe/`

| 文件 | 职责 |
| --- | --- |
| `__init__.py` | 公开导出：`SymmBuffer`、`bf16_mega_moe`、`fp8_fp4_mega_moe`、`mega_moe`、`transform_weights_for_mega_moe`、`StoreGroup` |
| `api.py` | SymmBuffer、量化推理入口、权重转换、Plan 缓存与流/锁编排 |
| `_bf16.py` | BF16 校验、`prepare`/`run`、按调用的保存配置与 autograd 分流 |
| `_autograd.py` | `BF16Function`：每次前向的反向状态持有与 backward 调用 |
| `validation.py` | 路由合法性扫描（`check_routing`）与跨 rank 一致性检查（`collective_check`） |
| `reference.py` / `golden.py` | CPU 参考实现与 Golden 缓存 |
| `benchmark.py` / `performance.py` / `example.py` | 基准口径、性能汇总、可运行示例 |
| `timer/` | 量化 timer 离线解析（`python -m deep_ep_experimental.megamoe.timer`） |
| `timer/bf16/` | BF16 训练计时：`TimerCapture` 采集 + `trace_parser` 导出 |
| `configs/` | 模型 JSON（`deepseek_v4_pro` 等） |

### Host 扩展 `csrc/`

| 文件 | 职责 |
| --- | --- |
| `bindings.cpp` | pybind11 模块 `_C`：`Plan`、`BF16Plan`、能力开关与 `api_protocol` |
| `forward.cpp` / `forward.hpp` | 量化推理 Host：`deepep::megamoe::Plan` 校验、分配与启动 |
| `forward_bf16.cpp` | BF16 前向 Host op（`MegaMoeBF16Forward`）+ `BF16Plan` 前向方法与 `buffer_size` |
| `backward.cpp` | BF16 反向 Host op（`MegaMoeBF16Backward`）+ `BF16Plan::backward` |
| `bf16_plan.hpp` | `BF16Plan` 与前/反向结果结构声明 |
| `bf16_tiling.cpp` / `bf16_tiling_impl.hpp` | BF16 Host tiling：Wave 预算、peer 字节、workspace 切分 |
| `bf16_launcher.cpp` / `bf16_kernel_launch.hpp` | 通信 context 初始化（peer 窗口地址表）与设备入口符号声明 |
| `bf16_host_utils.hpp` / `check.hpp` | 窗口控制区清理、保存状态/工作区分配、连续性检查 |

### 设备 kernel `csrc/kernels/mega_moe/`

| 路径 | 职责 |
| --- | --- |
| `mega_moe.cpp` | 量化推理 kernel 入口 `mega_moe_kernel_do`，按 `is_a8w4`/BS 特化 |
| `mega_moe_wave_a8w4.hpp` / `mega_moe_wave_a8w8.hpp` | W4A8 / W8A8 量化 Wave 实现（含共享专家策略） |
| `mega_moe_bf16.cpp` / `mega_moe_bf16.hpp` | BF16 前向 kernel 入口与唯一前向主体（Wave 调度模式 0/1） |
| `mega_moe_bf16_backward.cpp` / `mega_moe_bf16_backward.hpp` | BF16 反向 kernel（P0–P6 阶段流水） |
| `mega_moe_tiling.hpp` / `mega_moe_arch35.hpp` | 量化 tiling 数据结构 / 平台常量 |
| `stage/` | 数据移动与完成条件：路由压缩、token dispatch、GMM1 激活、GMM2 combine、unpermute、topk 权重 gather、就绪协议；`stage/moe_init_routing_v3/arch35/` 为 BF16 路由展开（InitRoutingV3） |
| `catlass_gmm/` | CATLASS GEMM 供数与实现（量化 A8W4、BF16 前/反向），见其 `README.md` |
| `epilogue/` | 激活（SwiGLU 系列）、MXFP8/MXFP4 量化、Gate/Up 保存与反向 epilogue |
| `common/` | Host/Device 共用的 tiling、layout（保存状态/Dispatch 输出/反向 workspace 布局）、peermem 窗口、Wave 规划、context、timer |
| `mc2_common/` | MC2 通信工具与异常 dump 策略 |
| `timer/` | 设备侧打点实现（`ascend_timer_device.hpp`） |

### 测试

| 路径 | 职责 |
| --- | --- |
| `experiments/megamoe/tests/` | pytest、不启动设备 kernel：公开 API、参考/Golden、打包、主机协议、计时解析 |
| `experiments/megamoe/tests/bf16/` | NPU 设备测试入口 `test_mega_moe.py`（CPU FP64 真值、精度/性能/Timer 选项） |
| `experiments/megamoe/tests/integration/` | BF16 生命周期集成脚本 |
| `tests/runtime/megamoe_peer_rows_test.cpp` + `test_megamoe_peer_rows.py` | peer 行映射往返的 C++ 单测（按编译宏参数化） |
| `tests/runtime/test_megamoe_train.py` / `test_megamoe_train_packaging.py` | 训练 Runtime owner 契约、双组件单扩展打包契约 |

## 3. 核心组件与职责

### 3.1 公开入口

- `bf16_mega_moe`：BF16 前向（训练/推理），原地写 `y` 返回 `None`（`experiments/megamoe/python/deep_ep_experimental/megamoe/api.py:677`）。签名不携带保存/调度/计时配置——这些由 SymmBuffer 与编译配置承载。
- `fp8_fp4_mega_moe`：W4A8 量化推理主入口，位置参数对齐 DeepGEMM 风格（`api.py:627`）。仅支持 `recipe=(1,1,32)`、SwiGLU、无 clamp（`api.py:650`）。
- `mega_moe`：显式输入的兼容入口，返回 `(y, per_call_counts)`（`api.py:489`）。
- `transform_weights_for_mega_moe`：原始 ND 权重一次性转 NZ 打包布局；routed FP4 GMM1 同时准备 Prefill/Decode 两套布局并以版本计数禁止原地修改（`api.py:418`、`_RoutedWeightPair` `api.py:403`）。
- 量化路径拒绝任何 `requires_grad` 输入（推理专用，`api.py:575`）；训练能力只属于 BF16 入口。

### 3.2 SymmBuffer：配置与资源归属

`SymmBuffer`（`api.py:96`）是集合资源 owner：持有输入字段（`x/x_sf/topk_idx/topk_weights`，构造时分配 `api.py:222`）、配置、Plan 缓存与 `_MegaRuntime`。关键配置（构造参数 `api.py:104`）：

| 配置 | 归属与作用 |
| --- | --- |
| `mma_type` | `"fp8xfp4"`/`"fp8xfp8"`/`"bf16xbf16"`，选择计算与输入格式（`api.py:144`） |
| `need_grad` | BF16 训练语义开关：乘权位置（前乘/后乘）与反向能力（`api.py:118`；语义表见 README「BF16语义与状态」） |
| `gmm1_save_type` | GMM1 Gate/Up 保存策略 `bf16`/`none`/`mxfp8`（`api.py:152`）；即任务计划中的 `gmm1_out_type`（计划里的 `auto` 在代码中不存在） |
| `max_recv_tokens` | 本 rank 接收 route 行数上限，参与 workspace/保存布局计算（`api.py:120`） |

配置在进入 SHMEM 前做跨 rank 一致性校验（`collective_check`，`api.py:179`）。构造按 `mma_type` 选择 native 容量函数 `buffer_size`/`bf16_buffer_size`（`api.py:184-198`）并创建 `_MegaRuntime`。生命周期约定：所有 rank 同序构造/调用/`destroy()`（`api.py:382`）。

### 3.3 根 Runtime 与调用编排

`_MegaRuntime` 继承根仓 `deep_ep._runtime.Runtime`（`api.py:39`），是唯一 SHMEM owner：持有对称窗口、进程级独占、专属 stream 与调用锁。工厂按组件选择 `create_megamoe_runtime`/`create_megamoe_train_runtime`（`deep_ep/_runtime.py:139`）。每次调用在 `buffer._lock` 下把调用方流切入 runtime 专属流串行执行，返回后 `record_stream`（量化 `api.py:610`、BF16 `_bf16.py:137`）——同一 owner 的对称空间禁止被不同流并发覆盖。窗口地址由核心 native 暴露（`megamoe_workspace_address` `api.py:259` / `megamoe_train_workspace_address` `_bf16.py:66`），Plan 只借用。

### 3.4 Host Plan

- 量化 `Plan`（`experiments/megamoe/csrc/forward.hpp:20`）：一个 BS 对应一份准备好的计划；持有 tiling、本地 workspace 与 context，`forward` 返回 `(y, counts)`。
- BF16 `BF16Plan`（`experiments/megamoe/csrc/bf16_plan.hpp:54`）：`prepare` 计算并分配临时 workspace（同 shape/保存配置复用，`forward_bf16.cpp:158`）；`forward_out`/`backward`/`*_with_timer` 各自调用 Host op（`forward_bf16.cpp:188`、`backward.cpp:171`）。构造参数 `weight_before_gmm2` 实际传入 `need_grad`（`_bf16.py:68`），反向要求该 Plan 为前乘语义（`backward.cpp:177`）。
- Python 侧 Plan 缓存：量化按 BS 单份缓存（`api.py:248`）；BF16 按 `(bs, shape, topk, max_recv, p dtype, need_grad, save_type, limit)` 键控（`_bf16.py:52`）。

### 3.5 前向 Wave 调度（模式 0 / 模式 1）

BF16 前向只有一份实现 `mega_moe_bf16.hpp` 的 `Process`（`mega_moe_bf16.hpp:705`）→ `ProcessWithSaveMode`（`:736`）→ `ProcessWave`。工作单元是 **Wave**：按专家行数把任务切成 `Slice`，以 `kGroupRows=256` 为组预算推进（`common/mega_moe_bf16_wave_plan.hpp:16`；`Next`/`Plan`/`PlanScheduled` `:139/:169/:195`）。

编译宏 `DEEPEP_MEGAMOE_BF16_FORWARD_SCHEDULE` 决定调度模式，在 tiling 中以 `forwardSchedule` 下发（`bf16_tiling.cpp:116`），同一 `ProcessWave` 主体内分支：

| 模式 | AIC（GEMM） | AIV1（通信/归并） |
| --- | --- | --- |
| 0 交错（默认） | Gmm1Wave 与 Gmm2Wave 按 Wave 交错；`m≥512` 时 GMM2 滞后一拍（`mega_moe_bf16.hpp:1248`、`UseGmm2Lag` `wave_plan.hpp:67`） | DispatchWave 与 CombineSlice 轮转推进（`mega_moe_bf16.hpp:1197`） |
| 1 分阶段 | 先提交全部 Gmm1Wave，再统一 Gmm2Wave（`mega_moe_bf16.hpp:1251-1267`） | Dispatch 排空后才 Combine（`:1197` 的 `!dispatch.valid` 分支） |

两模式共用同一套 Wave 基础函数、行映射与就绪协议（`wave_plan.hpp`），不是两份实现；构建侧仅改变编排顺序（`experiments/megamoe/CMakeLists.txt:91`）。AIV0 负责 SwigluWave（`mega_moe_bf16.hpp:1237`），AIC 用 CATLASS `WaveMatmul` 执行 GEMM（`catlass_gmm/mega_moe_bf16_catlass_impl.hpp`）。

### 3.6 保存策略（训练）

`gmm1_save_type` 决定前向为反向保留什么 Gate/Up（`ResolveSavedGateUpMode`，`mega_moe_bf16.hpp:709`）：

| 取值 | 行为 |
| --- | --- |
| `bf16` | 保存原始 BF16 Gate/Up 平面（`kSaveBf16`） |
| `mxfp8` | 量化保存（每 32 列一组 + 可选 clipping 掩码位平面，`mega_moe_bf16_layout.hpp:352-358`） |
| `none` | 不保存，反向 P0-C 重算 GMM1（`mega_moe_bf16_backward.hpp:18`） |

非梯度调用强制按 `none` 执行（`_bf16.py:133`）。保存布局由 Host/Device 共用的 `TrainingSavedStateLayout` 定义（`mega_moe_bf16_layout.hpp:295`），前向 kernel 在计数表快照稳定后落盘（`SaveForwardState`，`mega_moe_bf16.hpp:798`）。

### 3.7 反向 / autograd

`BF16Function`（`_autograd.py:12`）在 `forward` 中保存 `counts`、`saved_state`、`dispatch_output` 与权重引用（`_autograd.py:26`），`backward` 在同一锁/流约定下调用 `plan.backward`（`_autograd.py:45`），返回 `dx/dtopk_weights/dw1/dw2`（`_autograd.py:65`）。各次前向的反向状态相互独立，支持多前向后逆序反向（README「BF16语义与状态」）；`once_differentiable`（`_autograd.py:32`）禁用二阶反向。反向 kernel 阶段流水见 §4.2。

### 3.8 Timer

设备打点由编译宏 `DEEPEP_MEGAMOE_TIMER` 控制（默认 OFF），暴露为 `_C.timer_enabled`（`csrc/bindings.cpp:65`）。两条采集路径独立于常规性能测量：

- 量化：`sym_buffer.allocate_timer()` 生成零初始化 trace 张量，仅传给单独的诊断调用（`api.py:310`；`timer_enabled` 仅对 `fp8xfp4` 开放，`api.py:308`）。
- BF16：`TimerCapture` 上下文绑定一次 autograd 图（`timer/bf16/capture.py:12`），经 `forward_with_timer`/`backward_with_timer` 采集（`_bf16.py:148`、`_autograd.py:43`），`export()` 落盘原始 `.bin` 并可离线解析（`capture.py:47`）。

离线解析入口 `python -m deep_ep_experimental.megamoe.timer`（推理）与 `timer/bf16`（训练），见 README「测试结果」。

## 4. 关键数据流

### 4.1 一次 `bf16_mega_moe` 调用（前向）

```text
x[M,H]  topk_idx[M,K]  topk_weights[M,K]  W1[E,2F,H]  W2[E,H,F]
   │
   ├─ 1 入口校验 ─ 2 流/锁 ─ 3 Plan 准备 ─ 4 Host 分配/启动 ─ 5 kernel 流水
   ▼
y[M,H]（原地） + counts[E]（可选累加） + saved_state/dispatch_output（训练）
```

1. **入口校验**：`bf16_mega_moe` → `_bf16.run`（`api.py:677`、`_bf16.py:85`）。校验 y 形状/连续性/不重叠（`_bf16.py:102-119`）、权重布局 `[E,2F,H]/[E,H,F]` 或 `[E,H,2F]/[E,F,H]` 配套（`_bf16.py:112`）、输入字段（`validate_buffer_inputs` `_bf16.py:25`）。
2. **流与锁**：取 `buffer._lock`，runtime 专属流等待调用方流（`_bf16.py:137-143`）。
3. **Plan 准备**：`prepare` 命中缓存或新建 `BF16Plan` 并 `plan.prepare` 分配临时 workspace（`_bf16.py:47`、`forward_bf16.cpp:158`）。
4. **Host 执行**：梯度模式走 `BF16Function.apply`，否则 `plan.forward_out`（`_bf16.py:146/:150`）→ `MegaMoeBF16Forward`（`forward_bf16.cpp:8`）：解析权重布局、生成 tiling、分配 `saved_state`/`dispatch_output`（`forward_bf16.cpp:95-97`）、清理窗口控制区（`:105`）、启动设备 kernel（`:108`）。
5. **kernel 流水**（`mega_moe_bf16.cpp:20` → `mega_moe_bf16.hpp:705`）：
   1. **路由展开**：`moe_init_routing_v3` 把 x 按 TopK 槽位打包进 peermem（`mega_moe_bf16.hpp:750`），gather 路由权重（`:763`）；
   2. **计数交换**：全 EP 域交换每专家 token 计数、推导 prefix（`CrossRankSyncCntPerExpert` `:785`）；
   3. **保存快照**：稳定后保存路由元数据与计数表（`:798`，训练时）；
   4. **Wave 流水**（`ProcessWave`）：`DispatchWave` 把行搬到专家所在 rank → `Gmm1Wave`（Gate/Up）→ `SwigluWave`（激活；`need_grad` 时此处乘 FP32(p) 形成 Hp）→ `Gmm2Wave` → `CombineSlice` 把结果送回源 rank；
   5. **收尾**：`ResetTokenPerExpert` + `UnpermuteV2` 按 TopK 槽位 FP32 归并出 y 行（`FinishForward` `:1278-1292`），跨 rank 软同步后才允许下一次调用复用窗口。

乘权语义由 `need_grad` 决定：`need_grad=True` 为前乘（Hp=BF16(SiLU_FP32(Gate)×Up×FP32(p)) 进 GMM2）；`need_grad=False` 为后乘（无权 BF16 H 进 GMM2，源端 FP32 逐 route 乘 p 后归并），由 tiling 的 `routeWeightBeforeGmm2` 承载（`forward_bf16.cpp:65`）。`torch.no_grad()` 不改变乘权位置，只跳过构图与 Gate/Up 保存。

**量化路径对照**（`fp8_fp4_mega_moe`）：输入为 FP8 E4M3 + E8M0 scale（或 BF16 `x` 覆盖、内部量化），同样经路由 → Dispatch → GMM1 → SwiGLU → GMM2 → Combine，但按 W4A8/W8A8 量化 Wave（`mega_moe_wave_a8w4.hpp`/`a8w8`）执行，共享专家按 B≤256/>256 切换策略；无反向、无保存。详见 README 第 1 节。

### 4.2 训练反向数据流

前向为反向保留（`forward_bf16.cpp:73-95`、`mega_moe_bf16_layout.hpp:295/:276`）：

- `counts`：每本地专家 token 数；
- `saved_state`：路由元数据（cumsum、行映射 `expandedRowIdx`、rank 前缀、计数表快照）+ 按保存策略的 Gate/Up 平面（可选 MXFP8 scale 与 clipping 掩码）；
- `dispatch_output`：dispatch 后的打包 token（即反向所需的 Xpacked）与配对 TopK 权重。

反向（`BF16Function.backward` → `MegaMoeBF16Backward`，`mega_moe_bf16_backward.hpp:14-24`）：

| 阶段 | 动作 | 执行单元 |
| --- | --- | --- |
| P0-A/B | grad_y 路由 gather；恢复计数表并 Dispatch dO | AIV |
| P0-C | `save=none` 时重算 Gate/Up（Z = Xpacked @ W1.T） | AIC |
| P1 | dHp = dO @ W2；epilogue 形成 dh=FP32(p)·FP32(dHp) → dZ，并累计 dp 偏导 | AIC+AIV |
| P2 | packed dX = dZ @ W1，经 Combine 发回各源 rank | AIC+AIV |
| P3 | dW1 = dZ.T @ Xpacked | AIC |
| P4 | `save=none` 时 dW2 = dO.T @ Hp | AIC |
| P5 | Unpermute：各 TopK 槽位 FP32 求和 → BF16 grad_x | AIV |
| P6 | dp 偏导归约 → grad_topk_weights | AIV |

输出 `dx/dtopk_weights/dw1/dw2` 回传 PyTorch（`backward.cpp:165`）。dW2 在保存模式下紧接 P2 前按专家计算（`mega_moe_bf16_backward.hpp:14`）。反向 workspace 由 `mega_moe_workspace_backward.hpp` 同时供 Host 与 Device 寻址（文件头注释与 `backward.cpp:139`）。

## 5. 接口边界

**MegaMoE 依赖（仓库其余部分）**：

| 依赖 | 用途 | 证据 |
| --- | --- | --- |
| `deep_ep._runtime.Runtime` | 进程级资源 owner、流管理、`integer`/`warn_unclosed` 工具 | `api.py:13`、`_MegaRuntime` `api.py:39` |
| `deep_ep._native.require_native`/`npu_api` | 核心 `_C` 身份校验（`api_protocol`）与设备/流访问 | `api.py:14`、`api.py:27-35` |
| `deep_ep._store.StoreGroup` | 主机协调（TCPStore）+ 算子通信（SHMEM+MTE）的进程组 | `api.py:15`、README 第 1 节 |
| 核心 `deep_ep._C`（`csrc/bindings/module.cpp`） | SHMEM Session、对称窗口分配与地址暴露 | `api.py:259`、`_bf16.py:66`、[架构说明](architecture.md) |
| 根仓 `csrc/kernels/common/` | 设备共享头：Wave 预算 `mega_moe_wave_math.hpp`、peer 行映射 `mega_moe_peer_rows.hpp`、timer | `mega_moe_bf16_wave_plan.hpp:7-8`、`experiments/megamoe/CMakeLists.txt:68` |
| 第三方 SHMEM / CATLASS / CANN（Bisheng） | 对称窗口通信、GEMM 模板、编译与 runtime 库 | `experiments/megamoe/CMakeLists.txt:25-26,47` |

**MegaMoE 私有**：`experiments/megamoe/csrc/`（Host）、`csrc/kernels/mega_moe/`（设备实现）、`python/deep_ep_experimental/megamoe/`（Python 包）全部为 MegaMoE 私有；其 `SymmBuffer` 与 ElasticBuffer/HT/LL 资源、容量约束互不通用（[架构说明](architecture.md)）。

**测试布局**：见 §2 测试表。CPU 检查不启动设备 kernel；NPU 设备验证入口在 `experiments/megamoe/tests/bf16/test_mega_moe.py` 与多机脚本（README 第 3 节）。

**打包入口**：`setup.py` 以 `DEEPEP_BUILD_MEGAMOE` / `DEEPEP_BUILD_MEGAMOE_TRAIN` 开关把 `deep_ep_experimental.megamoe` 纳入 wheel，并打包设备 `*.so` 与 `configs/*.json`（`setup.py:24-41,58-60`）；一键入口 `experiments/megamoe/scripts/build.sh`（README 第 2 节）。Python/原生以 `api_protocol=5` 握手防版本错配（`bindings.cpp:19`、`api.py:34`）。

## 6. 构建与配置

| 宏/选项 | 定义处 | 作用 |
| --- | --- | --- |
| `DEEPEP_BUILD_MEGAMOE` | `setup.py:25`、`experiments/megamoe/CMakeLists.txt:5` | 构建量化推理组件（`libmegamoe_kernel.so` + Host `forward.cpp`） |
| `DEEPEP_BUILD_MEGAMOE_TRAIN` | `setup.py:27`、`CMakeLists.txt:87` | 构建 BF16 训练组件（`libdeepep_megamoe_train_bf16.so` + `forward_bf16/backward/bf16_*`）；两者可同时开启，共用一个 Host 扩展 `_C` |
| `DEEPEP_MEGAMOE_BF16_FORWARD_SCHEDULE` | `CMakeLists.txt:91-94` | BF16 前向调度：0=交错（默认）、1=分阶段；写入 tiling `forwardSchedule`（`bf16_tiling.cpp:116`） |
| `DEEPEP_MEGAMOE_BF16_BACKWARD_SLICES` / `DEEPEP_MEGAMOE_BF16_COMBINE_SLICES` | `CMakeLists.txt:88-89` | 反向 dHp/SwiGLU 切片就绪、dX 切片就绪 Combine 的编译开关 |
| `DEEPEP_BF16_FORCE_EXPERT_READY` | `CMakeLists.txt:90` | Dispatch 等待完整专家的构建策略（覆盖诊断拆分） |
| `DEEPEP_MEGAMOE_TIMER` | `CMakeLists.txt:52,132,177` | 设备打点编译开关（默认 OFF），经 `_C.timer_enabled` 暴露 |
| `DEEPEP_NPU_ARCH=Ascend950` | README「BF16构建与安装」 | 根仓构建的目标架构入口；设备编译固定 `--npu-arch=dav-3510`（`CMakeLists.txt:47,125`） |

运行期诊断环境变量：`MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS`（反向 Dispatch 专家拆分覆盖，`backward.cpp:125`）。构建缓存/依赖锁定见 `dependencies.lock.json`；SHMEM/CATLASS 可用 `SHMEM_ROOT`/`CATLASS_ROOT` 指定（README 第 2 节）。

## 7. 设计约定与不变量

1. **配置不进公开前向参数，由 SymmBuffer 承载**。`bf16_mega_moe` 签名无保存/调度/计时参数（`api.py:677-688`）；保存策略取自 `b.gmm1_save_type`（`_bf16.py:133`），调度是编译配置，Timer 走独立采集对象。量化入口保留历史 `x`/`timer` 关键字（`api.py:638-641`），不新增配置面。
2. **模式 0/1 共用同一份前向主体与 Wave 基础函数**。仅 `ProcessWave` 内按 `forwardSchedule` 分支推进顺序（`mega_moe_bf16.hpp:1197,1247`）；Wave 规划/行映射/就绪协议一份（`wave_plan.hpp`），不复制实现。
3. **配置跨 rank 一致**。构造/prepare/调用各阶段经 `collective_check` 集合校验（`api.py:179,286,294`）；TopK 权重 dtype 与前后向顺序一致性可开启 `debug_collectives` 跨 rank 诊断（`api.py:47`）。
4. **SymmBuffer 是唯一通信 owner，Plan 只借用窗口**。BF16 Plan 构造校验 SHMEM 身份并借地址（`forward_bf16.cpp:130-133`）；本地 workspace 归 Plan 复用，`workspace_bytes` 与 `symmetric_buffer_bytes` 分开计量（`api.py:297-304`）。
5. **调用串行于专属流**。锁 + 流切换保证同一对称空间不被并发覆盖；异步 Tensor 一律 `record_stream`（`api.py:610-623`、`forward_bf16.cpp:113-117`）。
6. **W4A8/W8A8 量化推理语义保持现状**。默认 recipe/激活不可改（`api.py:650`）、推理专用无反向（`api.py:575`）、路由权重在 GMM2 后施加的数值口径不变（README「兼容入口」段）；BF16 训练不改变量化路径行为。
7. **保存/反向状态按调用独立**。每次前向的 counts、dispatch 数据与保存张量由 autograd context 持有（`_autograd.py:26`）；临时 workspace 跨调用复用（`_bf16.py:52` 缓存键含 shape/保存配置）。
8. **各 rank 同序、同 B 调用**，`destroy()` 全体同序释放（README「输入、输出与公开接口」、`api.py:382`）；窗口复用边界由跨 rank 软同步发布（`mega_moe_bf16.hpp:1286-1292`）。
9. **Host/Device 布局公式单一事实源**。保存状态、Dispatch 输出、反向 workspace 的字节偏移由 `mega_moe_bf16_layout.hpp`/`mega_moe_workspace_backward.hpp` 的 constexpr 布局类同时供 Host 分配与 Device 寻址（`forward_bf16.cpp:73-88` 与 kernel 共用）。

## 8. 术语表

| 术语 | 含义 |
| --- | --- |
| GMM（Grouped Matmul） | 按专家分组的批量矩阵乘；GMM1 产出 Gate/Up，GMM2 产出隐藏输出 |
| SymmBuffer | MegaMoE 对称内存资源 owner：配置、输入字段、SHMEM 窗口与 Runtime 的归属对象 |
| Host Plan | Host 侧按 BS/shape 准备的执行计划（量化 `Plan` / BF16 `BF16Plan`）：tiling、workspace、启动封装 |
| Wave | BF16 前/反向的任务推进单元：按专家行数切 `Slice`，以 256 行为组预算（`mega_moe_bf16_wave_plan.hpp`） |
| 模式 0 / 模式 1 | BF16 前向调度：0=交错（GEMM 与通信互锁推进）、1=分阶段（先 GMM1 后 GMM2/先 Dispatch 后 Combine）；编译宏 `DEEPEP_MEGAMOE_BF16_FORWARD_SCHEDULE` |
| dispatch / combine | 跨卡把 token 行发往专家所在 rank / 把专家输出发回源 rank |
| Unpermute | 按 TopK 槽位把多路专家输出 FP32 归并回每 token 一行 |
| InitRoutingV3 | BF16 路由展开 stage：生成 TopK 槽位映射与打包输入（`stage/moe_init_routing_v3/`） |
| `need_grad` | SymmBuffer 训练开关：决定乘权位置（前乘/后乘）与反向能力 |
| `gmm1_save_type` | GMM1 Gate/Up 保存策略（`bf16`/`mxfp8`/`none`）；任务计划文档中称 `gmm1_out_type` |
| `max_recv_tokens` | 本 rank 接收 route 行数的显式上限，参与 workspace/保存布局 |
| peermem 窗口 | SHMEM 对称窗口（跨卡可直接寻址），布局以 `mega_moe_peermem.hpp` 为单一事实源 |
| saved_state / dispatch_output | 前向为反向保留的持久状态：路由元数据+保存的 Gate/Up / dispatch 后的打包 token 与权重 |
| Timer / TimerCapture | `DEEPEP_MEGAMOE_TIMER` 编译的设备打点；BF16 训练侧的一次性采集对象 |
| W4A8 / W8A8 | 量化推理权重/激活格式（FP4 权重+FP8 激活 / FP8+FP8），对应 `mma_type="fp8xfp4"`/`"fp8xfp8"` |
| AIC / AIV / MIX | Ascend 计算单元类型；MegaMoE kernel 为 `__mix__(1,2)`（1 AIC + 2 AIV 一组） |
