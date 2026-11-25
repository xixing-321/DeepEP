# Copyright (c) 2026, Lu Lu

"""Run W8A8 training precision, timing and optional Timer capture with torchrun."""

import argparse
import json
import os
from pathlib import Path
import sys


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for field, default in (
        ("seq", 4096),
        ("hidden", 7168),
        ("ffn", 3072),
        ("topk", 6),
        ("expert-per-rank", 4),
        ("seed", 20260825),
        ("warmup", 5),
        ("loops", 20),
    ):
        parser.add_argument("--" + field, type=int, default=default)
    parser.add_argument("--routing-mode", choices=("balanced", "random", "imbalanced"), default="balanced")
    parser.add_argument("--data-profile", choices=("random", "normal", "equal_weights"), default="random")
    parser.add_argument("--routed-scaling-factor", type=float, default=1.0)
    parser.add_argument("--save-gmm1-output", choices=("mxfp8", "none"), default="mxfp8")
    parser.add_argument("--save-dispatched-x", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--swiglu-limit", type=float, default=0.0)
    parser.add_argument("--max-recv-tokens", type=int)
    parser.add_argument("--topk-weight-dtype", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument("--topk-weight-values", choices=("bf16-rounded", "full"))
    parser.add_argument("--skip", choices=("precision", "performance"))
    parser.add_argument("--timer", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.topk_weight_values is None:
        args.topk_weight_values = "bf16-rounded" if args.topk_weight_dtype == "bf16" else "full"
    if args.topk_weight_dtype == "bf16" and args.topk_weight_values != "bf16-rounded":
        parser.error("BF16 TopK weights require bf16-rounded values")
    if min(args.loops, args.seq, args.topk, args.expert_per_rank) < 1 or args.warmup < 0:
        parser.error("Shape and sample counts must be positive; warmup may be zero")
    return args


def kernel_samples(step, warmup, loops, path):
    import torch
    import torch.distributed as dist
    from torch_npu import profiler

    with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]) as prof:
        dist.barrier()
        for _ in range(warmup + loops):
            step()
            torch.npu.synchronize()
    prof.export_chrome_trace(str(path))
    data = json.loads(path.read_bytes())
    events = data if isinstance(data, list) else data["traceEvents"]
    samples = {}
    for scope, name in (("forward", "mega_moe_w8a8_training_forward"), ("backward", "mega_moe_w8a8_backward")):
        launches = [
            ev
            for ev in events
            if ev.get("ph") == "X"
            and name in ev.get("name", "")
            and ev.get("args", {}).get("Task Type", "").startswith("KERNEL")
        ]
        if len(launches) != warmup + loops:
            raise ValueError(f"{path}: expected {warmup + loops} {name} launches, found {len(launches)}")
        ordered = sorted(launches, key=lambda ev: float(ev["ts"]))
        samples[scope] = [float(ev["dur"]) / 1000 for ev in ordered[warmup:]]
    samples["complete"] = [a + b for a, b in zip(samples["forward"], samples["backward"], strict=True)]
    return samples


def main():
    args = arguments()
    # Run the public wheel with the same CPU input snapshots and route combine as the BF16 device suite.
    test_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(test_dir.parents[2]))
    sys.path.insert(0, str(test_dir / "bf16"))
    import torch
    import torch.distributed as dist
    from deep_ep_experimental.megamoe import SymmBuffer, fp8_fp4_mega_moe
    from experiments.megamoe.tests.bf16.test_mega_moe import make_inputs
    from experiments.megamoe.tests.bf16.accuracy import combine_baseline
    from experiments.megamoe.tests.bf16.perf_compare import event_stats
    from experiments.megamoe.tests.w8a8.dispatch_rebuild import digest, make_rebuilder
    from experiments.megamoe.tests.w8a8.training_reference import run_reference

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    group = dist.group.WORLD
    report_group = dist.new_group(backend="gloo")
    rank, world = group.rank(), group.size()
    args.output.mkdir(exist_ok=True, parents=True)
    args.inference = False
    precision = args.skip != "precision"
    op = None
    try:
        inputs = make_inputs(args, rank, world, global_tokens=precision)
        selection = slice(rank * args.seq, (rank + 1) * args.seq) if precision else slice(None)
        x, ids, p, dy = (inputs[name][selection].npu() for name in ("x", "ids", "p", "dy"))
        w1, w2 = (inputs[name].npu() for name in ("w1", "w2"))
        leaves = (x, p, w1, w2)
        for tensor in leaves:
            tensor.requires_grad_()
        op = SymmBuffer(
            group,
            world * args.expert_per_rank,
            args.seq,
            args.topk,
            args.hidden,
            args.ffn,
            mma_type="fp8xfp8",
            ranks_per_node=int(os.environ.get("LOCAL_WORLD_SIZE", 0)),
            need_grad=True,
            max_recv_tokens=args.max_recv_tokens,
            gmm1_save_type=args.save_gmm1_output,
            save_dispatched_x=args.save_dispatched_x,
        )
        op.x, op.topk_idx, op.topk_weights = x, ids, p
        if not args.save_dispatched_x:
            op.dispatch_rebuilder, _ = make_rebuilder(group, x, ids, p, args.expert_per_rank)
        op.prepare(args.seq)

        def forward(with_counts=False):
            y = torch.empty_like(x)
            counts = torch.zeros(args.expert_per_rank, dtype=torch.int32, device=x.device) if with_counts else None
            fp8_fp4_mega_moe(
                y, w1, w2, op, cumulative_local_expert_recv_stats=counts, activation_clamp=args.swiglu_limit
            )
            return y, counts

        def align_capture():
            # Each rank finishes its own profiler export and reference hashes before capture.
            torch.npu.synchronize()
            dist.barrier(group=report_group)

        def step(with_counts=False, align_phases=False):
            if align_phases:
                align_capture()
            y, counts = forward(with_counts)
            if align_phases:
                align_capture()
            result = dict(
                zip(("y", "dx", "dtopk_weights", "dw1", "dw2"), (y, *torch.autograd.grad(y, leaves, dy)), strict=True)
            )
            if with_counts:
                result["counts"] = counts
            return result

        report = {"rank": rank}
        if precision:
            partial = run_reference(
                inputs, x.device, rank * args.expert_per_rank, args.save_gmm1_output, args.swiglu_limit, x_wire="mxfp8"
            )
            reference = combine_baseline(partial, args, rank, world, x.device, group=report_group)
            reference["counts"] = partial["counts"].int()
            values = step(with_counts=True)
            report["bitwise_equal"] = {
                name: value.dtype == reference[name].dtype
                and value.shape == reference[name].shape
                and digest(value) == digest(reference[name])
                for name, value in values.items()
            }
            del values, reference, partial
        del inputs
        if args.skip != "performance":
            for _ in range(args.warmup):
                step()
                torch.npu.synchronize()
            dist.barrier()
            samples = {scope: [] for scope in ("forward", "backward", "complete")}
            for _ in range(args.loops):
                start, mid, end = (torch.npu.Event(enable_timing=True) for _ in range(3))
                start.record()
                y, _ = forward()
                mid.record()
                torch.autograd.grad(y, leaves, dy)
                end.record()
                end.synchronize()
                for scope, a, b in (("forward", start, mid), ("backward", mid, end), ("complete", start, end)):
                    samples[scope].append(a.elapsed_time(b))
            report["event_ms"] = samples
            report["kernel_ms"] = kernel_samples(
                step, args.warmup, args.loops, args.output / f"rank_{rank}_profiler.json"
            )
        if args.timer:
            from deep_ep_experimental.megamoe.timer.bf16.capture import TimerCapture

            plain = {name: digest(t) for name, t in step(with_counts=True).items()}
            with TimerCapture(op) as capture:
                timed = {name: digest(t) for name, t in step(with_counts=True, align_phases=True).items()}
            capture.export(args.output / "timer")
            report["timer_exact"] = plain == timed
        reports = [None] * world
        dist.all_gather_object(reports, report, group=report_group)
        passed = all(all(r.get("bitwise_equal", {}).values()) and r.get("timer_exact", True) for r in reports)
        if rank == 0:
            summary = {}
            for kind in ("event_ms", "kernel_ms"):
                if kind in report:
                    summary[kind] = {
                        scope: event_stats([max(v) for v in zip(*(r[kind][scope] for r in reports), strict=True)])
                        for scope in report[kind]
                    }
            output = dict(passed=passed, ep=world, config=vars(args), rank_max_rounds=summary, ranks=reports)
            (args.output / "report.json").write_text(json.dumps(output, default=str, indent=2) + "\n")
            print(
                json.dumps(
                    dict(
                        passed=passed,
                        rank_max_p50_ms={
                            key: {scope: st["median"] for scope, st in scopes.items()}
                            for key, scopes in summary.items()
                        },
                    )
                ),
                flush=True,
            )
        if not passed:
            raise AssertionError(f"W8A8 validation failed: see {args.output / 'report.json'}")
    finally:
        if op is not None:
            op.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
