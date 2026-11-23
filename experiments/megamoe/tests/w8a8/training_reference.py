# Copyright (c) 2026, Lu Lu

"""W8A8 training oracle using independent Torch NPU quantization and GEMMs.

Inputs are the global token snapshot and the weights belonging to this rank.
Expert rows follow expert, source rank, token, TopK slot order.
"""

import torch

from experiments.megamoe.tests.bf16.npu_baseline import (
    _mxfp8_saved_roundtrip,
    _swiglu_backward_fp32,
    _swiglu_fp32,
)


def feature_quant(value):
    """Group each row along its feature axis."""
    import torch_npu

    return torch_npu.npu_dynamic_mx_quant(value.contiguous(), axis=-1, round_mode="rint", dst_type=torch.float8_e4m3fn)


def decode_feature(pair):
    data, scale = pair
    width = data.size(1)
    # E8M0 is stored as exponent bytes. Take its bit pattern, then expand
    # each scale over the corresponding 32 columns.
    exponents = scale.view(torch.uint8).flatten(1).float() - 127.0
    return (data.float() * torch.exp2(exponents).repeat_interleave(32, dim=1)[:, :width]).to(torch.bfloat16)


def token_quant(value):
    """Group along the token axis of a single expert, zero-padding its tail."""
    padded_rows = (value.size(0) + 63) // 64 * 64
    packed = value.new_zeros((value.size(1), padded_rows))
    packed[:, : value.size(0)].copy_(value.T)
    return feature_quant(packed)


def mx_matmul(left, right):
    """Inputs are independent E4M3/E8M0 pairs in [M,K] and [N,K] order."""
    import torch_npu

    a, sa = left
    b, sb = right
    return torch_npu.npu_quant_matmul(
        a,
        b.T,
        sb.transpose(0, 1),
        pertoken_scale=sa,
        output_dtype=torch.bfloat16,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        group_sizes=[1, 1, 32],
    )


@torch.no_grad()
def run_reference(inputs, device, expert_start, save_mode="mxfp8", swiglu_limit=0.0, *, x_wire="bf16"):
    """Return route results and rank-local weight gradients on CPU."""
    if save_mode not in ("mxfp8", "none"):
        raise ValueError("W8A8 training supports mxfp8 and none")
    if x_wire not in ("mxfp8", "bf16"):
        raise ValueError("The X communication dtype must be mxfp8 or bf16")
    ids, x, dy, p, w1, w2 = (inputs[key] for key in ("ids", "x", "dy", "p", "w1", "w2"))
    # Use the original BF16 inputs for both independent directions.
    if any(t.dtype != torch.bfloat16 for t in (x, dy, w1, w2)):
        raise TypeError("X, dY, W1 and W2 must be BF16")
    indices = [(ids == expert_start + i).nonzero(as_tuple=True) for i in range(w1.size(0))]
    result = dict(
        ids=ids,
        counts=torch.tensor([rows.numel() for rows, _ in indices], dtype=torch.int64),
        routes=[],
        dw1=torch.zeros_like(w1, device="cpu"),
        dw2=torch.zeros_like(w2, device="cpu"),
        reference="w8a8",
        topk_weight_dtype=p.dtype,
    )
    products = []
    for expert, (rows, slots) in enumerate(indices):
        if rows.numel() == 0:
            continue
        xr, do, prob = (value.to(device) for value in (x[rows], dy[rows], p[rows, slots, None]))
        w1e, w2e = w1[expert].to(device), w2[expert].to(device)
        x_feature, w1_forward = feature_quant(xr), feature_quant(w1e)
        raw = mx_matmul(x_feature, w1_forward)
        forward_hp = (_swiglu_fp32(raw, swiglu_limit) * prob.float()).to(raw.dtype)
        y = mx_matmul(feature_quant(forward_hp), feature_quant(w2e))

        clip_source = None
        if save_mode == "mxfp8":
            clip_source = raw
            z = _mxfp8_saved_roundtrip(raw)
        else:
            z = mx_matmul(x_feature, w1_forward)

        dhp = mx_matmul(feature_quant(do), feature_quant(w2e.T))
        h = _swiglu_fp32(z, swiglu_limit, clip_source)
        hp = (h * prob.float()).to(z.dtype)
        dz = _swiglu_backward_fp32(dhp.float() * prob.float(), z, swiglu_limit, clip_source)
        products.append(dhp.float() * h)
        result["dw2"][expert].copy_(mx_matmul(token_quant(do), token_quant(hp)).cpu())
        # Scheme A derives the wgrad input from the actual FP8 payload;
        # scheme B takes both directions directly from the dispatched BF16.
        x_wgrad = decode_feature(x_feature) if x_wire == "mxfp8" else xr
        result["dw1"][expert].copy_(mx_matmul(token_quant(dz), token_quant(x_wgrad)).cpu())
        dx = mx_matmul(feature_quant(dz), feature_quant(w1e.T))
        result["routes"].append(dict(rows=rows, slots=slots, y=y.cpu(), dx=dx.cpu()))

    if products:
        # The small-op ReduceSum chooses its tree using the rank's routed row
        # count. Launch it once across all experts before splitting dP by route.
        dp = torch.cat(products).sum(dim=-1).to(p.dtype).cpu()
        for route, values in zip(result["routes"], dp.split([v.size(0) for v in products])):
            route["dp"] = values
    return result
