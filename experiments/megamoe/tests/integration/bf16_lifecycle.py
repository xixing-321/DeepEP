# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026

"""Run the BF16 lifecycle regression using its original torchrun path."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "bf16" / "bf16_lifecycle.py"), run_name="__main__")
