"""Optional dashboard dependencies must not prevent file-only training logs."""

import subprocess
import sys

import pytest


def test_file_monitor_without_wandb_overview(tmp_path):
    pytest.importorskip("wandb", reason="W&B ships with the GPU extra")
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
import sys
from pathlib import Path

sys.modules["prime_rl.monitors.wandb.overview"] = None
from prime_rl import monitors
from prime_rl.configs.monitors import FileMonitorConfig

async def check():
    await monitors.setup(file=FileMonitorConfig(), output_dir=Path(sys.argv[1]))
    await monitors.log({"test/value": 1.0}, step=0)
    await monitors.finalize()

asyncio.run(check())
""",
            str(tmp_path),
        ],
        check=True,
    )
    assert '"test/value": 1.0' in (tmp_path / "monitors/file/metrics.jsonl").read_text()
