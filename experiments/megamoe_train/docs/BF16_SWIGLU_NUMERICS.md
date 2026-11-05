# BF16 SwiGLU数值与梯度契约

## 前向计算

`R`表示舍入到BF16，`p`表示BF16路由权重转换后的FP32值。Gate/Up进入计算时由BF16转换为FP32；启用正的有限裁剪上限`L`时，Gate执行`min(g, L)`，Up执行`clamp(u, -L, L)`。以下`g/u`表示裁剪后的值。

```text
g = FP32(gate_BF16)，u = FP32(up_BF16)
按需clamp g、u
silu = g / (1 + exp(-g))             # FP32
H  = silu * u                       # FP32
Hp = R(H * p)
```

SiLU、H 和乘 p 连续使用 FP32 中间量，最后一次舍入形成 BF16 Hp。H 保留在寄存器中，前向和反向重建均采用同一规则。计算使用准确 Div。

## 反向重建与导数

反向重建H与前向使用相同的中间量边界：

```text
H  = silu * u                       # FP32
Hp = R(H * p)
dH = FP32(dHp) * p                  # FP32
s  = 1 / (1 + exp(-g))              # FP32
dUp   = R(dH * silu * up_mask)
dGate = R((dH * u) * (s * (1 + g * (1 - s))) * gate_mask)
```

SiLU、H、dH 及导数中间量使用 FP32；Hp 和 dGate/dUp 写出时舍入到 BF16。`gate_mask/up_mask`由裁剪前的输入决定，裁剪边界点传递梯度，超出对应裁剪区间时梯度为零。

该契约覆盖`ComputeForwardAux`的尾块H、`ComputeFusedFullTile`的满tile H及导数，以及`ComputeSwigluGradImpl`的尾块导数。`saved-none/bf16/mxfp8`共用反向epilogue：`none`使用GMM1重算值，`bf16`使用保存的BF16值，`mxfp8`使用解码后的压缩值及clipping掩码。MXFP8压缩具有独立误差，不能据此保证重建H与未压缩前向逐位一致。

dTopK 使用 FP32(dHp) 与重建的 FP32 H 相乘，每行每 256 列生成一个 FP32 partial。跨 tile 用 FP32 归约，最终路由权重梯度舍入到 BF16。路由权重在GMM2之前参与计算，不能将完整算子等同于在最终GMM输出上后乘p的外部计算图。

## 梯度约定与参考

导数按融合FP32子图计算，并在上述BF16张量边界收口，遵循浮点Cast的恒等梯度约定。该约定不是离散BF16舍入函数的严格数学导数；对包含BF16台阶的完整函数直接使用微小有限差分，不能验证该梯度契约。

测试参考[`reference_packed_intermediates()`](../tests/bf16/swiglu_reference.py)采用相同的 FP32 H 与 `dH=FP32(p)*FP32(dHp)` 规则。内部平滑SiLU/Up子图可用有限差分核对，clamp掩码和BF16边界需分别检查；算子精度验收使用相应测试入口的独立参考与误差判据。内部数值检查不构成训练收敛或外部计算图逐位一致的保证。
