# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026
"""Wheel contains the joint quantized and BF16 MegaMoE payload."""

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


def wheel(tmp, payload, enabled, *, missing=None, extra=()):
    path = tmp / "sample.whl"
    names = set()
    if enabled:
        names.update(
            (
                "__init__.py",
                "api.py",
                "_C.fake.so",
                "_bf16.py",
                "_autograd.py",
                "benchmark.py",
                "configs/smoke.json",
                "libmegamoe_kernel.so",
            )
        )
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
                    megamoe=dict(enabled=enabled),
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


@pytest.mark.parametrize("enabled", [False, True])
def test_single_megamoe_payload_matches_build_option(tmp_path, shmem_wheel_payload, enabled):
    info = inspect_wheel(wheel(tmp_path, shmem_wheel_payload, enabled))
    assert info["megamoe"]["enabled"] is enabled


@pytest.mark.parametrize("missing", ["_C.fake.so", "_bf16.py", "_autograd.py", "benchmark.py", "libmegamoe_kernel.so"])
def test_megamoe_requires_matching_host_python_and_device(tmp_path, shmem_wheel_payload, missing):
    with pytest.raises(RuntimeError, match="Incomplete MegaMoE"):
        inspect_wheel(wheel(tmp_path, shmem_wheel_payload, True, missing=missing))


@pytest.mark.parametrize(
    "extra",
    [
        "build/stale/deep_ep_experimental/megamoe/_C.old.so",
        COMPONENT + "_C.old.so",
    ],
)
def test_nested_or_duplicate_extension_is_rejected(tmp_path, shmem_wheel_payload, extra):
    with pytest.raises(RuntimeError):
        inspect_wheel(wheel(tmp_path, shmem_wheel_payload, True, extra=[extra]))


def test_second_device_library_is_rejected(tmp_path, shmem_wheel_payload):
    with pytest.raises(RuntimeError, match="one device library"):
        inspect_wheel(wheel(tmp_path, shmem_wheel_payload, True, extra=[COMPONENT + "libextra_kernel.so"]))
