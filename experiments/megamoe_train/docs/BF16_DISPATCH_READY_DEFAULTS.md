# BF16前反向Dispatch细分的当前默认


普通BF16使用默认构建`adaptive-auto`。前向采用M192／Ready384和静态producer元数据，反向saved路径采用M128／256行进度和静态元数据。单行搬运是DMA粒度，与是否等待整专家ready分开选择。

## 统一整专家回退选项

`DEEPEP_BF16_FORCE_EXPERT_READY`是唯一的前反向Dispatch就绪回退构建选项，默认`OFF`。wheel构建可设置同名环境变量，直接CMake构建可传`-DDEEPEP_BF16_FORCE_EXPERT_READY=ON/OFF`；构建系统向Host和Device一致传入C++值`0/1`。

| 构建值 | 前向 | 反向 |
| --- | --- | --- |
| `OFF`（默认） | 保留下文M192／Ready384与静态producer元数据策略 | 保留下文saved模式的256行进度及逐专家回退 |
| `ON` | 关闭tile-ready和配套元数据，消费既有完整专家Dispatch完成flag | Host与Device均固定`splitExperts=0`，不分配行进度后缀、不准备配套元数据，消费既有完整专家完成flag |

`ON`优先于`MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS`，该环境变量不能重新开启细分。默认`OFF`时才使用原专家数选择及显式排障覆盖。整专家回退保留必要同步、DMA搬运、GMM1/GMM2计算及保存模式，也不控制SwiGLU的UB／GM交接选择；`DEEPEP_BF16_SWIGLU_DECOUPLE`仍保留`0/1/2`且设备默认`2`。

更改选项后必须完整重建、重装同一次构建的Host/Device wheel；所有EP rank必须使用同一构建。SO名称、公开接口及saved-state布局不变。不要只替换设备SO，也不要在各rank混用默认构建和整专家构建。以下细分说明均针对默认`OFF`。

## 前向

新路径对**所有符合条件的非空专家**分别启用细分，不再只选首专家。正常形状要求每卡专家数≤7、`M × TopK ≥ 每卡专家数 × 768`，该专家接收容量截断后的实际行数≥768。none、bf16、mxfp8保存模式共用此选择，SwiGLU解耦的hidden阈值是另一套独立策略。

静态元数据在原`GetCumsumForMMAIV()`的source前缀循环内生成，沿用原pre-Dispatch同步；计算侧按真实producer发布阶段组织M块，AIC与消费GMM1结果的AIV使用相同坐标。没有payload相关的metadata arrival等待，实际行ready及整专家完成flags仍保留。

元数据必须能放入既有GM空闲word、UB与staging空间。当前支持28／32／36 AIC、EP≤256等既有容量条件；不足时回退原首专家选择，不能把它描述成所有shape无条件全细分。容量截断的专家退回原计算映射，实际行ready检查继续保留。小专家仍等整专家。

默认判定可概括为：

```text
静态元数据容量可用且shape满足候选条件：
    每个实际非空专家独立判断：行数>=768用Ready384，否则整专家ready
静态元数据容量不可用：
    回原首个非空专家选择，其余整专家ready
shape不满足候选条件：
    全部整专家ready
```

## 反向

| 保存模式与实际行数 | 默认交付 |
| --- | --- |
| bf16／mxfp8，每卡专家数≤7，专家实际有效行数>128 | 全部符合条件的非空专家细分，256行进度＋原N3配套分核 |
| bf16／mxfp8，每卡专家数≥8 | 整专家ready，保留原3_1分核，不分配细分进度后缀 |
| bf16／mxfp8，专家实际有效行数≤128 | 整专家ready，保留单M保护 |
| none | 原重算＋UB流程，不启用本细分 |

反向元数据在原`RestoreTokenPerExpert()` source循环内分散准备，复用原AIV同步；AIC首次使用时经过一处准备完成等待点，点内仍会轮询，后续不再逐组等待metadata arrival。真实dO行ready、dW完整专家归约、dO复用同步不变。

正常使用不用新增参数。排障可用`MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS=0/1/.../all`选择前k个非空专家。默认每卡专家数≤7时为all、≥8时为0；该专家数边界复用前向常量，但实际行数门槛仍各自保留。显式参数是排障覆盖，可在≥8专家时强制all；再逐专家应用单M保护。none始终强制按0处理。

默认前向为M192／Ready384、自适应N条带，反向细分为M128／256行producer进度、固定N3和跨组尾块后置，反向整专家为3_1。专家数边界及统一整专家回退不改梯度数学、舍入或既有workspace前缀偏移。≥8专家默认以及强制整专家构建均不追加细分控制区。

## 诊断预留区

saved-state中原来的`256 × int32`阶段码区域现仅保留为1KiB reserved空间，不再写阶段码或在Host逐次清零，以固定后续Gate/Up、scale和clipping mask偏移。它与上述真正用于等待数据行到达的`BackwardDispatchProgressLayout`不是同一块内存；真实行进度发布和等待仍由当前就绪策略控制，不能作为诊断残留删除。

## 支持边界

容量、必要同步和真实数据等待均保留。更大 EP、跨节点及不同路由分布需要独立设备验证；
离线调度覆盖不等于设备功能、性能或训练收敛保证。
