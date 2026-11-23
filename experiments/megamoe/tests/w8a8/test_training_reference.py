# Copyright (c) 2026, Lu Lu

from collections import Counter

import importlib
import pytest

torch = pytest.importorskip("torch")

reference = importlib.import_module("experiments.megamoe.tests.w8a8.training_reference")


def make_inputs(p_dtype):
    g = torch.Generator().manual_seed(39)
    x = torch.randn(33, 256, generator=g).to(torch.bfloat16) / 16
    return dict(
        x=x,
        dy=torch.randn(x.shape, generator=g).to(torch.bfloat16) / 16,
        p=torch.rand(33, 2, generator=g).to(p_dtype),
        w1=torch.randn(3, 512, 256, generator=g).to(torch.bfloat16) / 16,
        w2=torch.randn(3, 256, 256, generator=g).to(torch.bfloat16) / 16,
        ids=torch.tensor([[0, 3]] * 32 + [[1, 0]], dtype=torch.int32),
    )


@pytest.mark.parametrize("save_mode", ["mxfp8", "none"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_reference_quantization_boundaries(monkeypatch, save_mode, dtype):
    calls = []
    inputs = make_inputs(getattr(torch, dtype))

    def quant(value):
        assert value.dtype == torch.bfloat16
        calls.append(value.detach().clone())
        return value, None

    def mm(left, right):
        return (left[0].float() @ right[0].float().T).to(torch.bfloat16)

    monkeypatch.setattr(reference, "feature_quant", quant)
    monkeypatch.setattr(reference, "mx_matmul", mm)
    result = reference.run_reference(inputs, "cpu", 0, save_mode=save_mode, swiglu_limit=0.015)
    assert result["counts"].tolist() == [33, 1, 0]
    assert torch.count_nonzero(result["dw1"][2]) == torch.count_nonzero(result["dw2"][2]) == 0
    assert result["routes"][0]["slots"].tolist() == [0] * 32 + [1]
    assert result["routes"][1]["slots"].tolist() == [0]
    for route in result["routes"]:
        assert route["dp"].dtype == inputs["p"].dtype
        raw_do = inputs["dy"][route["rows"]]
        assert any(torch.equal(value, raw_do) for value in calls)
        shape = (256, (len(raw_do) + 63) // 64 * 64)
        assert any(
            value.shape == shape
            and torch.equal(value[:, : len(raw_do)], raw_do.T)
            and not torch.count_nonzero(value[:, len(raw_do) :])
            for value in calls
        )

    # Every nonempty expert uses K-feature groups for feature GEMMs and
    # independently padded K-token groups for the weight gradient GEMMs.
    shapes = Counter(tuple(t.shape) for t in calls)
    assert shapes[(512, 64)] == 2  # dZ.T in dW1, one per expert.
    assert (256, 512) in shapes  # W1.T for dX.


@pytest.mark.parametrize("save_mode", ["mxfp8", "none"])
def test_npu_training_reference(save_mode):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is required")
    result = reference.run_reference(make_inputs(torch.float32), "npu", 0, save_mode, 0.015)
    assert result["counts"].tolist() == [33, 1, 0]
    for value in (result["dw1"], result["dw2"], *(r[k] for r in result["routes"] for k in ("y", "dx", "dp"))):
        assert torch.isfinite(value).all()
        assert torch.count_nonzero(value)
