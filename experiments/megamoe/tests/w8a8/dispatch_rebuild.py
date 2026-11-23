# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026

"""Device regression for the W8A8 framework re-dispatch contract. Run with torchrun."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

from deep_ep_experimental.megamoe import SymmBuffer, mega_moe


def digest(tensor):
    data = tensor.contiguous().reshape(-1).view(torch.uint8).cpu()
    return hashlib.sha256(data.numpy()).hexdigest()


def make_rebuilder(group, x, ids, p, local_experts):
    """Rebuild from source tensors using HCCL, independently of the kernel route table."""
    flat_ids = ids.cpu().flatten().long()
    ep, rank = group.size(), group.rank()
    local = flat_ids.bincount(minlength=ep * local_experts).view(ep, local_experts)
    counts = local.to(device=x.device, dtype=torch.int32)
    gathered = [torch.empty_like(counts) for _ in range(ep)]
    dist.all_gather(gathered, counts, group=group)
    recv_counts = torch.stack([value[rank].cpu() for value in gathered])
    splits_in = local.sum(1).tolist()
    splits_out = recv_counts.sum(1).tolist()
    flat_counts = recv_counts.flatten()
    starts = torch.cat((flat_counts.new_zeros(1), flat_counts.cumsum(0))).tolist()
    order = flat_ids.argsort(stable=True).to(x.device)
    row_order = torch.tensor(
        [
            row
            for e in range(local_experts)
            for peer in range(ep)
            for row in range(starts[peer * local_experts + e], starts[peer * local_experts + e + 1])
        ],
        device=x.device,
        dtype=torch.long,
    )

    def exchange(value):
        recv = value.new_empty((row_order.numel(), *value.shape[1:]))
        dist.all_to_all_single(recv, value, splits_out, splits_in, group=group)
        return recv.index_select(0, row_order)

    def rebuild():
        quant, scales = torch_npu.npu_dynamic_mx_quant(
            x.detach(), axis=-1, round_mode="rint", dst_type=torch.float8_e4m3fn
        )
        h = x.shape[1]
        wire = torch.cat((quant.view(torch.uint8), scales.reshape(-1, h // 32).view(torch.uint8)), dim=1)
        rows = order // ids.shape[1]
        packed = exchange(wire.index_select(0, rows))
        data = packed[:, :h].contiguous().view(torch.float8_e4m3fn)
        scale = packed[:, h:].contiguous().view(torch.float8_e8m0fnu)
        weights = exchange(p.detach().reshape(-1).index_select(0, order))
        return data, scale, weights

    return rebuild, recv_counts.sum(0).tolist()


def compare_dispatch(group, config, tensors, ids, rebuild, expected_counts, output):
    x, p, w1, w2, dy = tensors
    m, h = x.shape
    f, e, k = w2.shape[-1], w2.shape[0], ids.shape[1]
    grad_inputs = (x, p, w1, w2)
    rows = sum(expected_counts)
    saved = {}
    buffer = SymmBuffer(
        group,
        group.size() * e,
        m,
        k,
        h,
        f,
        mma_type="fp8xfp8",
        need_grad=True,
        max_recv_tokens=config.capacity,
        gmm1_save_type=config.mode,
    )

    def pack(value):
        key = (tuple(value.shape), value.dtype)
        for name, spec in (
            ("x", ((rows, h), torch.float8_e4m3fn)),
            ("scale", ((rows, h // 32), torch.float8_e8m0fnu)),
            ("p", ((rows,), p.dtype)),
        ):
            if key == spec:
                saved[name] = value
        return value

    def measure(keep):
        buffer.save_dispatched_x = keep
        buffer.dispatch_rebuilder = None if keep else rebuild
        torch.npu.synchronize()
        dist.barrier(group=group)
        start = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            y, counts = mega_moe(x, ids, p, w1, w2, buffer)
        torch.npu.synchronize()
        end_forward = torch.npu.memory_allocated()
        peak_forward = torch.npu.max_memory_allocated()
        assert counts.cpu().tolist() == expected_counts
        if keep:
            assert set(saved) == {"x", "scale", "p"}
            rebuilt = rebuild()
            raw = {name: digest(saved[name]) == digest(value) for name, value in zip(("x", "scale", "p"), rebuilt)}
            saved.clear()
            del rebuilt
            assert all(raw.values()), raw
            output["wire_exact"] = raw
            print(f"rank={group.rank()} wire exact: {raw}", flush=True)
        else:
            assert not saved, list(saved)
            # Backward must use the callback captured by this forward.
            buffer.dispatch_rebuilder = None
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        grads = torch.autograd.grad(y, grad_inputs, grad_outputs=dy)
        torch.npu.synchronize()
        peak_backward = torch.npu.max_memory_allocated()
        print(f"rank={group.rank()} backward done, save_dispatched_x={keep}", flush=True)
        result = {
            name: digest(value) for name, value in zip(("y", "counts", "dx", "dp", "dw1", "dw2"), (y, counts, *grads))
        }
        return result, dict(
            before_forward=start,
            live_after_forward=end_forward - start,
            peak_forward=peak_forward - start,
            peak_backward=peak_backward - start,
            workspace=buffer.workspace_bytes,
        )

    try:
        buffer.prepare(m)
        with torch.npu.stream(buffer._runtime.stream):
            buffer._prepare_w8a8_backward(dy)
        if config.check_overflow:
            hot = torch.arange(k, device=x.device, dtype=torch.int32).expand_as(ids).contiguous()
            try:
                mega_moe(x, hot, p, w1, w2, buffer)
            except ValueError as error:
                assert "EP rank 0" in str(error) and "exceeding max_recv_tokens" in str(error), str(error)
                print(f"rank={group.rank()} {error}", flush=True)
            else:
                raise AssertionError("over-capacity forward did not raise")
        stored, output["memory_saved"] = measure(True)
        recreated, output["memory_rebuilt"] = measure(False)
        output["outputs_exact"] = {name: stored[name] == recreated[name] for name in stored}
        assert all(output["outputs_exact"].values()), output["outputs_exact"]
    finally:
        buffer.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("m", 4096), ("h", 7168), ("f", 3072), ("k", 6), ("e", 4), ("capacity", 24576)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--mode", choices=("mxfp8", "none"), required=True)
    parser.add_argument("--p-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--check-overflow", action="store_true")
    parser.add_argument("--random-routing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    group = dist.group.WORLD
    rank = group.rank()
    try:
        m, h, f, k, e = args.m, args.h, args.f, args.k, args.e
        gen = torch.Generator().manual_seed(10652 + rank)

        def normal(shape, scale):
            return (torch.randn(shape, generator=gen) * scale).bfloat16().npu()

        if args.random_routing:
            ids = torch.rand((m, e * group.size()), generator=gen).topk(k, dim=1).indices
        else:
            ids = (torch.arange(m * k).view(m, k) + rank * k) % (e * group.size())
        ids = ids.int().npu()
        p = torch.rand((m, k), generator=gen)
        p = (p / p.sum(dim=-1, keepdim=True)).to(getattr(torch, args.p_dtype)).npu().requires_grad_()
        x = normal((m, h), 0.25).requires_grad_()
        w1 = normal((e, 2 * f, h), h**-0.5).requires_grad_()
        w2 = normal((e, h, f), f**-0.5).requires_grad_()
        dy = normal((m, h), 0.25)
        rebuild, counts = make_rebuilder(group, x, ids, p, e)
        output = {"rank": rank, "counts": counts, "overflow_checked": args.check_overflow}
        compare_dispatch(group, args, (x, p, w1, w2, dy), ids, rebuild, counts, output)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank-{rank}.json").write_text(json.dumps(output, indent=2) + "\n")
        print(output, flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
