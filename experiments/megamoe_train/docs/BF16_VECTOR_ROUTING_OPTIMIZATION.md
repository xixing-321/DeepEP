# BF16前反向SwiGLU与路由搬运优化

## 前向：批量读权重、直接解包与整块计算

`epilogue/block_epilogue_swiglu_forward_bf16.hpp::ComputeSwigluDirect()`把BF16 Gate/Up直接解包到FP32寄存器，以固定小块完成向量计算，按tile批量读路由权重。原来每行读一个p再同步的管理开销减少。仍保留准确Div、clamp，以及SwiGLU结果先舍入BF16、转回FP32乘p、最后舍入BF16的顺序。不是把数学公式换成低精度近似。

普通BF16前向固定使用M192与上述直接整块计算，不再通过旧`DEEPEP_BF16_M192_FORWARD`宏选择实现。反向仍使用独立的M128计算块；通信fragment与计算tile的粒度分别定义。

## 前向：解耦选择与容量必须一致

`DEEPEP_BF16_SWIGLU_DECOUPLE`控制普通BF16前向中GMM1与SwiGLU之间的交接方式，保留该编译选项：

| 编译值 | 前向交接方式 |
| --- | --- |
| `0` | 强制UB握手路径：AIC把一对Gate/Up tile写入UB，AIV0等待后完成SwiGLU和路由权重乘法，再通知AIC复用UB。 |
| `1` | 强制GM解耦路径：AIC先把原始BF16 Gate/Up写入GM，当前专家的所有AIC发布完成后，AIV0按tile从GM读回UB并完成激活。 |
| `2` | auto：按hidden和实际保存模式选择上述两条路径。 |

两条路径均将加权后的SwiGLU结果写入GM，供GMM2读取；GM解耦并不表示SwiGLU直接在GM上计算。两者保留既有clipping、准确Div以及先将SwiGLU结果舍入BF16、再乘p并舍入BF16的数值边界。

正式设备构建在模块`CMakeLists.txt`中显式传入`-DDEEPEP_BF16_SWIGLU_DECOUPLE=2`。共享头文件在未定义宏时的回退值为`0`，用于没有该编译定义的Host/CPU等翻译单元，**不代表当前设备SO默认使用UB路径**。

auto策略由`layout/mega_moe_layout.hpp::ForwardUsesGmSwiglu(hidden,savedMode)`实现：

| hidden | none / bf16 | mxfp8 |
| --- | --- | --- |
| 小于3584 | GM解耦 | GM解耦 |
| 等于3584 | UB握手 | GM解耦 |
| 大于3584 | UB握手 | UB握手 |

表中的savedMode必须先经`ResolveSavedGateUpMode()`解析。公共保存模式`auto`在clipping关闭时解析为`mxfp8`，开启时解析为`bf16`；它与编译值`2`的交接策略auto不是同一个选项。例如H3584、保存模式auto且clipping关闭时，实际按mxfp8走GM解耦，不能把未解析的auto值直接带入上表。

Host在分配前向临时空间时，先解析保存模式，再通过`launcher.cpp::MegaMoeBF16SwigluDecoupled()`直接调用设备SO导出的`deepep_train_mega_moe_bf16_swiglu_decoupled_for_saved_mode(hidden,savedMode)`。该符号用设备构建中的同一策略返回结果，Host再调用`ForwardSwigluExtraBytes()`决定额外容量，避免误用Host头文件回退值推断设备路径。

非解耦路径不增加原始Gate/Up临时GM。GM路径中，`bf16`保存模式复用已有的原始Gate/Up持久平面，不增加这部分容量；`none/mxfp8`将Hp空间临时用作原始Gate，在既有控制区之后追加一份原始Up平面，额外字节数为`AlignUp(max_recv_tokens * (n / 2) * sizeof(bfloat16_t), 512)`，其中`n=2F`。既有workspace前缀偏移和saved-state格式保持不变。

SO中仍保留旧的hidden-only查询符号，并按保存模式策略的并集返回结果；**当前Host没有向旧查询回退的逻辑**，直接依赖上述带savedMode的查询。改变该宏后必须完整重建并重装同一构建的Host/Device wheel，不能只替换一个SO或混装新旧产物。

该宏仅控制前向交接。反向的`BackwardUsesGm(savedMode)`另按已解析的保存模式决定路径：`none`走UB，`bf16/mxfp8`走GM；不能用前向编译值`0/1/2`替代这一判断。

## Dispatch统一整专家回退与诊断清理

构建选项`DEEPEP_BF16_FORCE_EXPERT_READY=ON/OFF`默认`OFF`，可通过同名wheel构建环境变量或直接CMake选项配置，Host/Device一致获得C++值`1/0`。默认保留现有前向Ready384及反向saved模式行进度策略；`ON`关闭前向tile-ready及配套元数据，并在Host/Device均强制反向`splitExperts=0`，使用既有整专家Dispatch完成flag。它优先于运行时`MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS`，不改变上面的SwiGLU `0/1/2`选择、保存模式或数值边界。

更改选项后需完整重建、重装Host/Device wheel，所有EP rank使用同一构建；SO名字和公开API不变。反向调试打印及诊断阶段码写入已删除，saved-state原1KiB阶段码区域仅保留为reserved以固定后续布局；它不是算法使用的Dispatch行进度后缀。

## InitRouting：多行成组搬运

`moe_v3_gather_out.h`的普通BF16 X Gather使用4/2行成组读入，按组同步，连续目标一次写出。双缓冲槽和尾组仍分别受MTE完成事件保护；UB容量或DMA长度不满足时回退原路径。

`routing/mega_moe_topk_weight_gather.hpp`对65537～131072条路由提供固定两个65536窗口。每次先生成安全的窗口内偏移，再Gather，以原始BF16位选择对应窗口结果。短第二窗只读有效GM字节。不会把越界地址先Gather后再掩码。≤65536保留原快路径，超过131072回退；UDMA/MGroup默认不启用两窗。临时UB约152KiB，在Dispatch分配UB之前释放，没有新增GM或运行态遍历窗口搜索。

## 反向：满tile单遍计算，保留数值语义

`epilogue/block_epilogue_swiglu_backward_bf16.hpp::RunFusedFullTile()`在固定寄存器小块内读入Gate/Up/dHp，完成Hp、dTopK局部乘积与dZ所需运算。路由权重按tile预取，减少逐行同步和两遍重新读Gate/Up。仅128×256满tile走此路径，尾块保留原正确路径。

保留准确Div、clamp导数及MXFP8剪裁掩码、最终BF16张量边界和dTopK四组64元素的原归约顺序。SiLU及dH×up内部采用FP32中间量，详见[数值契约](BF16_SWIGLU_NUMERICS.md)。原来的GM/UB布局、双AIV分工、ready发布和等待不变。`none`仍走UB握手，`bf16/mxfp8`仍走已有GM解耦；不带旧none-GM实验、实验控制尾部或额外完整GM缓冲。

`mega_moe_bf16_backward.hpp::GatherRoutes()`在P1/P2三个工作tile上方的固定剩余UB中分两槽组织dY搬运：K3为4行，H7168为2行，容量不足回退1行。仍读取原始dY，不提前乘p；所有原有跨核/跨卡同步保持。

## 反向Dispatch稳定方案

在上述优化基础上接入全部专家细分、配套分核、组内3宽和跨组尾块后置；单M专家回退整专家ready，none保持原路径。详见[BF16反向Dispatch细分与配套分核](BF16_BACKWARD_DISPATCH_READY.md)。设备性能需要按实际配套、shape 和路由独立测量。
