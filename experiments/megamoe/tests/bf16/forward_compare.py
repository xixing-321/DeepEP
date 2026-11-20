"""Same-input BF16 forward comparison of MegaMoE, BigOp and DeepEP MC."""

import csv
from contextlib import ExitStack
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
from uuid import uuid4

from perf_compare import baseline_plan, bench_scope, device_group, exchanged_rows, fused_forward, reorder_chunks


BACKENDS = ("megamoe", "bigop", "mc")


def kernel_rounds(directory, iterations):
    """Union of device task intervals per profiler step; includes compute and communication."""
    from deep_ep._timer import merge_intervals

    steps = {}
    devices = set()
    for path in Path(directory).rglob("kernel_details*.csv"):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                step = int(row["Step Id"])
                start, duration = Decimal(row["Start Time(us)"].strip()), Decimal(row["Duration(us)"].strip())
                if not start.is_finite() or not duration.is_finite() or duration < 0:
                    raise ValueError(f"Invalid kernel timing in {path}: {start}, {duration}")
                if duration:
                    steps.setdefault(step, []).append((start, start + duration))
                    devices.add(row["Device_id"])
    if len(devices) != 1 or len(steps) != iterations or max(steps) - min(steps) + 1 != iterations:
        raise ValueError(
            f"Expected {iterations} consecutive device steps on one rank under {directory}, got {sorted(steps)}"
        )
    return [float(sum(end - start for start, end in merge_intervals(steps[i]))) / 1000 for i in sorted(steps)]


def profile_forward(call, directory, loops):
    import torch
    import torch.distributed as dist
    from torch_npu import profiler

    config = profiler._ExperimentalConfig(
        export_type=profiler.ExportType.Text, profiler_level=profiler.ProfilerLevel.Level1
    )
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=0, active=loops, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
        experimental_config=config,
    ) as prof:
        dist.barrier()
        for _ in range(loops):
            call()
            torch.npu.synchronize()
            prof.step()
    return kernel_rounds(directory, loops)


def reference_forward(backend, tensors, args, rank, world, resources):
    import torch
    import torch.distributed as dist

    from npu_baseline import _grouped_matmul, _swiglu_fp32

    x, ids, p, w1, w2 = tensors
    group, reason = device_group()
    if group is None:
        raise RuntimeError(reason)
    plan, reason = baseline_plan(ids, w1, w2, group, args, rank, world, args.save_gmm1_output)
    if plan is None:
        raise RuntimeError(reason)
    inverse = torch.argsort(plan.order)

    def dispatch(values):
        rows = exchanged_rows(group, values, plan.output_splits, plan.input_splits)
        return reorder_chunks(rows, plan.received_sizes, plan.sort_index, plan.local_experts)

    if backend == "mc":
        from deep_ep_experimental.ep_gmm_fused import alltoallv_gmm, gmm_alltoallv, udma_group

        resources.enter_context(udma_group(group=dist.group.WORLD, local_rank=rank))
        opts = dict(ep_size=world, expert_num=world * plan.local_experts)
        local_counts = torch.bincount(ids.flatten().long(), minlength=opts["expert_num"]).to(torch.int64)
        received_counts = torch.tensor(plan.received_sizes, dtype=torch.int64, device=x.device)
        received = sum(plan.received_sizes)
        gmm1_out = x.new_empty((received, 2 * args.ffn))
        gmm2_out = x.new_empty((ids.numel(), args.hidden))

        def gmm1(packed):
            return alltoallv_gmm(
                packed, plan.weight1, local_counts, received_counts, received_rows=received, out=gmm1_out, **opts
            )

        def gmm2(hp):
            return gmm_alltoallv(hp, plan.weight2, local_counts, received_counts, out=gmm2_out, **opts)
    else:

        def gmm1(packed):
            return _grouped_matmul(dispatch(packed), plan.weight1, plan.counts)

        def gmm2(hp):
            rows = _grouped_matmul(hp, plan.weight2, plan.counts)
            rows = reorder_chunks(rows, plan.return_sizes, plan.restore_index, plan.local_experts)
            return exchanged_rows(group, rows, plan.input_splits, plan.output_splits)

    def call():
        packed = x.index_select(0, plan.route_row)
        gate_up = gmm1(packed)
        weights = dispatch(p.reshape(-1).index_select(0, plan.order))
        hp = (_swiglu_fp32(gate_up, args.swiglu_limit) * weights.float().unsqueeze(1)).to(x.dtype)
        routes = gmm2(hp).index_select(0, inverse).reshape(plan.tokens, plan.topk, plan.hidden)
        y = torch.zeros_like(x, dtype=torch.float32)
        for slot in range(plan.topk):
            y.add_(routes[:, slot])
        return y.to(x.dtype)

    return call


def worker(rank, args, backend, directory):
    import torch
    import torch.distributed as dist

    from test_mega_moe import make_buffer, make_inputs, prepare_worker
    from precision_metrics import write_report

    world = len(args.devices)
    buffer_type = prepare_worker(rank, args, rank, world, {})
    os.environ["DEEPEP_SHMEM_ENDPOINT"] = args.shmem_ip_port
    try:
        inputs = make_inputs(args, rank, world, global_tokens=False)
        names = ("x", "ids", "p", "w1", "w2")
        digest = hashlib.sha256()
        for name in names:
            digest.update(memoryview(inputs[name].contiguous().view(torch.uint8).numpy()).cast("B"))
        tensors = [inputs[name].to(f"npu:{rank}").contiguous() for name in names]
        rows = int(inputs["all_counts"].reshape(world, -1).sum(1).max())
        capacity = math.ceil(rows / 128) * 128
        del inputs
        with ExitStack() as resources:
            if backend == "megamoe":
                op = make_buffer(buffer_type, dist.group.WORLD, args, world)
                resources.callback(op.destroy)
                # Exercise the training forward with its requested save policy.
                for t in (tensors[0], *tensors[2:]):
                    t.requires_grad_(True)

                def call():
                    y, _ = fused_forward(
                        op,
                        *tensors,
                        capacity,
                        save_gmm1_output=args.save_gmm1_output,
                        swiglu_limit=args.swiglu_limit,
                        api=args.api,
                        collect_counts=False,
                    )
                    return y
            else:
                call = reference_forward(backend, tensors, args, rank, world, resources)
            y = call()
            if y.shape != tensors[0].shape or y.dtype != torch.bfloat16 or not torch.isfinite(y).all().item():
                raise ValueError(f"{backend}: invalid BF16 forward output on rank {rank}")
            del y
            dist.barrier()
            samples = bench_scope(lambda _: call(), None, args)
            kernel = profile_forward(call, directory / f"rank_{rank}_profile", args.loops)
            dist.barrier()
            write_report(
                directory / f"rank_{rank}.json",
                {
                    "rank": rank,
                    "input_sha256": digest.hexdigest(),
                    "event_ms": samples,
                    "kernel_ms": kernel,
                },
            )
    finally:
        dist.destroy_process_group()


def compare_forward(args):
    import torch.multiprocessing as mp
    from deep_ep_experimental.megamoe.performance import summarize_rounds
    from precision_metrics import write_report

    output = args.output or Path("output") / f"forward_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    digests = None
    timings = {}
    for backend in BACKENDS:
        directory = output / backend
        directory.mkdir()
        # Each backend owns its runtime and releases it before the next one starts.
        mp.spawn(worker, args=(args, backend, directory), nprocs=len(args.devices), join=True)
        ranks = [json.loads((directory / f"rank_{i}.json").read_text()) for i in range(len(args.devices))]
        fingerprints = [r["input_sha256"] for r in ranks]
        if digests is not None and fingerprints != digests:
            raise ValueError(f"{backend}: inputs differ from MegaMoE")
        digests = fingerprints
        timings[backend] = {}
        for metric in ("event_ms", "kernel_ms"):
            us = [[v * 1000 for v in r[metric]] for r in ranks]
            timings[backend][metric] = summarize_rounds(us)["rank_max_mean_us"] / 1000
        print(f"[FORWARD] {backend} {timings[backend]}", flush=True)
    write_report(
        output / "summary.json",
        {
            "scope": "forward",
            "aggregation": "tail-half mean of per-round slowest rank",
            "kernel_metric": "union of device computation and communication task intervals per local profiler step",
            "config": {k: v for k, v in vars(args).items() if k != "output"},
            "rank_input_sha256": digests,
            "timings": timings,
        },
    )
    return 0
