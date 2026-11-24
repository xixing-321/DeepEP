# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026

"""Autograd and per-call state for the W8A8 training path."""

import math

import torch
from torch.autograd.function import once_differentiable

from deep_ep._native import npu_api
from .api import _extension


def _quantize(value):
    import torch_npu

    width = value.shape[-1]
    data, scale = torch_npu.npu_dynamic_mx_quant(
        value.contiguous().view(-1, width), axis=-1, round_mode="rint", dst_type=torch.float8_e4m3fn
    )
    return data.reshape(value.shape), scale.view(torch.float8_e8m0fnu).reshape(*value.shape[:-1], width // 64, 2)


def _launch_forward(owner, x, ids, p, w1, w2, out, limit, save_gate_up, save_dispatch, capture):
    allocations = _extension()._StreamAllocations()
    q1, q2 = _quantize(w1), _quantize(w2)
    p_fp32 = p.float()
    allocations.remember((*q1, *q2) if p.dtype == torch.float32 else (*q1, *q2, p_fp32))
    state = owner._plan.training_forward(
        x,
        ids,
        p_fp32,
        q1[0],
        q2[0],
        q1[1],
        q2[1],
        out,
        limit,
        save_gate_up,
        save_dispatch,
        capture,
        p.dtype == torch.float32,
        allocations,
    )
    return state, q1, allocations


def _saved_gate_up_planes(raw, rows, ffn, limit):
    # Planes are capacity-strided; autograd keeps their active row prefix.
    capacity = raw.shape[0]
    widths = (ffn, ffn, ffn // 32, ffn // 32) + ((ffn // 8,) * 3 if limit else ())
    pieces = raw.view(-1).split([capacity * width for width in widths])
    gate, up, gs, us, *masks = [plane.view(capacity, width)[:rows] for plane, width in zip(pieces, widths)]
    return (
        gate.view(torch.float8_e4m3fn),
        gs.view(torch.float8_e8m0fnu),
        up.view(torch.float8_e4m3fn),
        us.view(torch.float8_e8m0fnu),
        *masks,
    )


def _validate_rebuilt_dispatch(result, owner, rows, p_dtype):
    h = owner.hidden
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise ValueError("W8A8 dispatch_rebuilder must return (x, scale, topk_weight)")
    for name, value, shape, dtype in zip(
        ("x", "scale", "topk_weight"),
        result,
        ((rows, h), (rows, h // 32), (rows,)),
        (torch.float8_e4m3fn, torch.float8_e8m0fnu, p_dtype),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device != owner._device
            or not value.is_contiguous()
        ):
            raise ValueError(f"Rebuilt W8A8 {name} must be contiguous {dtype} {shape} on the buffer's NPU")
    return result


class W8A8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owner, limit, x, ids, p, w1, w2, out):
        ctx.rebuild_dispatch = None if owner.save_dispatched_x else owner.dispatch_rebuilder
        if not owner.save_dispatched_x and not callable(ctx.rebuild_dispatch):
            raise ValueError("Set sym_buffer.dispatch_rebuilder for this microbatch when save_dispatched_x=False")
        capture = getattr(owner, "_timer_capture", None)
        ctx.capture = capture
        if capture is not None:
            capture.claim()
        recompute = owner.gmm1_save_type == "none"
        state, w1_forward, allocations = _launch_forward(
            owner, x, ids, p, w1, w2, out, limit, not recompute, owner.save_dispatched_x, capture is not None
        )
        if capture is not None:
            capture.forward = state.timer
        offsets = torch.cat((state.counts.new_zeros(1), state.counts.cumsum(0, dtype=torch.int32)))
        rows = int(offsets[-1].item())
        # Dispatch rows are already grouped by local expert and source rank.
        maps = state.routes[:rows, :3].contiguous()
        packed_p = None
        if owner.save_dispatched_x:
            packed_p = state.routes[:rows, 3].contiguous().view(torch.float32).to(p.dtype)
        saved = () if recompute else _saved_gate_up_planes(state.gate_up, rows, owner.intermediate_hidden, limit)
        # Only remember storage allocated by this forward. Unpack hooks and framework re-dispatch may replace it.
        buffers = [maps, offsets]
        if owner.save_dispatched_x:
            buffers.extend((state.dispatch, state.scales, packed_p))
        if not recompute:
            buffers.append(state.gate_up)
        allocations.remember(buffers)
        ctx.allocations = allocations
        ctx.owner, ctx.limit, ctx.recompute = owner, limit, recompute
        ctx.rows, ctx.p_dtype = rows, p.dtype
        ctx.save_for_backward(
            w1,
            w2,
            state.dispatch[:rows] if owner.save_dispatched_x else None,
            state.scales[:rows] if owner.save_dispatched_x else None,
            maps,
            offsets,
            packed_p,
            *(w1_forward if recompute else saved),
        )
        ctx.mark_non_differentiable(state.counts)
        ctx.mark_dirty(out)
        return out, state.counts

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, _counts=None):
        w1, w2, dispatch, scales, maps, offsets, p, *saved = ctx.saved_tensors
        owner = ctx.owner
        capture = ctx.capture
        runtime = owner._runtime
        with owner._lock:
            runtime.ensure_alive()
            caller = npu_api().current_stream(runtime.device)
            if ctx.rebuild_dispatch is not None:
                dispatch, scales, p = _validate_rebuilt_dispatch(ctx.rebuild_dispatch(), owner, ctx.rows, ctx.p_dtype)
            runtime.stream.wait_stream(caller)
            with npu_api().stream(runtime.stream):
                plan = owner._prepare_w8a8_backward(dy)
                # Hooks can restore the BF16 source weights on a different stream.
                w1.record_stream(runtime.stream)
                w2.record_stream(runtime.stream)
                contiguous_dy = dy.contiguous()
                if contiguous_dy is not dy:
                    dy.record_stream(runtime.stream)
                weights = (*_quantize(w1.transpose(-1, -2)), *_quantize(w2.transpose(-1, -2)))
                ctx.allocations.remember(weights if contiguous_dy is dy else (*weights, contiguous_dy))
                if capture is not None:
                    capture.begin_backward()
                dx, dp, dw1, dw2, timer = plan.backward(
                    contiguous_dy,
                    p,
                    dispatch,
                    scales,
                    maps,
                    offsets,
                    () if ctx.recompute else saved,
                    (*weights, *saved) if ctx.recompute else weights,
                    ctx.limit,
                    capture is not None,
                    ctx.allocations,
                )
                if capture is not None:
                    capture.backward = timer
            caller.wait_stream(runtime.stream)
            if caller != runtime.stream:
                for t in (dx, dp, dw1, dw2):
                    t.record_stream(caller)
        return None, None, dx, None, dp, dw1, dw2, None


def validate_inputs(buffer, bs, x, ids, p):
    def match(value, shape, dtype, name):
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype not in dtype
            or value.device != buffer._device
            or not value.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous {dtype} tensor with shape {shape} on the buffer's NPU")

    if not 0 < bs <= buffer.num_max_tokens_per_rank:
        raise ValueError("BS is outside the W8A8 buffer capacity")
    match(x, (bs, buffer.hidden), (torch.bfloat16,), "x")
    match(ids, (bs, buffer.num_topk), (torch.int32,), "topk_idx")
    match(p, (bs, buffer.num_topk), (torch.bfloat16, torch.float32), "topk_weights")
    return match


def run(y, x, ids, p, w1, w2, buffer, stats, activation_clamp):
    limit = 0.0 if activation_clamp is None else float(activation_clamp)
    if not math.isfinite(limit) or not 0 <= limit <= torch.finfo(torch.float32).max:
        raise ValueError("activation_clamp must be a finite, nonnegative FP32 value")
    bs, h, f = y.shape[0], buffer.hidden, buffer.intermediate_hidden
    runtime = buffer._runtime
    e = buffer.num_experts // runtime.world_size
    match = validate_inputs(buffer, bs, x, ids, p)
    match(y, (bs, h), (torch.bfloat16,), "y")
    match(w1, (e, 2 * f, h), (torch.bfloat16,), "l1_weights")
    match(w2, (e, h, f), (torch.bfloat16,), "l2_weights")
    inputs = (x, ids, p, w1, w2)
    input_storage = {t.untyped_storage().data_ptr() for t in inputs}
    y_storage = y.untyped_storage().data_ptr()
    if y.requires_grad or y_storage in input_storage:
        raise ValueError("y must be separate storage without an existing autograd history")
    if stats is not None:
        match(stats, (e,), (torch.int32,), "cumulative_local_expert_recv_stats")
        if stats.untyped_storage().data_ptr() in (input_storage | {y_storage}):
            raise ValueError("Cumulative stats must not alias inputs or output")
    gradients = torch.is_grad_enabled() and any(t.requires_grad for t in (x, p, w1, w2))
    with buffer._lock:
        runtime.ensure_alive()
        caller = npu_api().current_stream(runtime.device)
        runtime.stream.wait_stream(caller)
        with npu_api().stream(runtime.stream):
            buffer._prepare(bs)
            # Native forward records the tensors it reads; these originals are converted in Python.
            w1.record_stream(runtime.stream)
            w2.record_stream(runtime.stream)
            if p.dtype != torch.float32:
                p.record_stream(runtime.stream)
            if gradients:
                y, counts = W8A8Function.apply(buffer, limit, x, ids, p, w1, w2, y)
            else:
                counts = _launch_forward(buffer, x, ids, p, w1, w2, y, limit, False, False, False)[0].counts
            if stats is not None:
                stats.add_(counts)
                stats.record_stream(runtime.stream)
        caller.wait_stream(runtime.stream)
        if caller != runtime.stream:
            counts.record_stream(caller)
            y.record_stream(caller)
    return y, counts
