# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026
"""One extension for quantized/BF16 payloads and component switching."""

import importlib.util
import json
from pathlib import Path
import zipfile
import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPONENT = "deep_ep_experimental/megamoe/"


def inspect_wheel(path):
    spec = importlib.util.spec_from_file_location("validate_package", ROOT / "scripts/validate_package.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.inspect_wheel(path)


def wheel(tmp, payload, train, beta, *, missing=None, extra=()):
    path = tmp / "sample.whl"
    names = set()
    if train or beta:
        names.update(("__init__.py", "api.py", "_C.fake.so"))
    if train:
        names.update(("_bf16.py", "_autograd.py", "libdeepep_megamoe_train_bf16.so"))
    if beta:
        names.update(("benchmark.py", "configs/smoke.json", "libmegamoe_kernel.so"))
    if missing:
        names.discard(missing)
    with zipfile.ZipFile(path, "w") as a:
        manifest = payload(a)
        a.writestr("deep_ep/_C.fake.so", "fixture")
        a.writestr(
            "deep_ep/_build_info.json",
            json.dumps(
                dict(
                    communication_operations=[],
                    megamoe=dict(enabled=beta),
                    megamoe_train=dict(enabled=train),
                    shmem_runtime=manifest,
                )
            ),
        )
        a.writestr("licenses/deepseek-deepep.LICENSE", "fixture")
        for n in names:
            a.writestr(COMPONENT + n, "fixture")
        for n in extra:
            a.writestr(n, "fixture")
    return path


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("beta", [False, True])
def test_one_extension_supports_all_capability_combinations(tmp_path, shmem_wheel_payload, train, beta):
    info = inspect_wheel(wheel(tmp_path, shmem_wheel_payload, train, beta))
    assert info["megamoe_train"]["enabled"] is train and info["megamoe"]["enabled"] is beta


@pytest.mark.parametrize("missing", ["_C.fake.so", "_bf16.py", "_autograd.py", "libdeepep_megamoe_train_bf16.so"])
def test_training_requires_matching_host_python_and_device(tmp_path, shmem_wheel_payload, missing):
    with pytest.raises(RuntimeError, match="Incomplete MegaMoE"):
        inspect_wheel(wheel(tmp_path, shmem_wheel_payload, True, False, missing=missing))


@pytest.mark.parametrize(
    "extra",
    [
        "deep_ep_experimental/megamoe_train/_C.old.so",
        "build/stale/deep_ep_experimental/megamoe/_C.old.so",
        COMPONENT + "_C.old.so",
    ],
)
def test_old_or_duplicate_extension_is_rejected(tmp_path, shmem_wheel_payload, extra):
    with pytest.raises(RuntimeError):
        inspect_wheel(wheel(tmp_path, shmem_wheel_payload, True, True, extra=[extra]))


@pytest.mark.parametrize(
    "train,beta,extra", [(False, True, "libdeepep_megamoe_train_bf16.so"), (True, False, "libmegamoe_kernel.so")]
)
def test_device_payload_must_match_capability(tmp_path, shmem_wheel_payload, train, beta, extra):
    with pytest.raises(RuntimeError, match="OFF wheel"):
        inspect_wheel(wheel(tmp_path, shmem_wheel_payload, train, beta, extra=[COMPONENT + extra]))
