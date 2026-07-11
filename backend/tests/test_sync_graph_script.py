from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_sync_graph_script_can_run_directly() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(backend_root / "scripts/sync_graph.py"), "--help"],
        cwd=backend_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--paper-id" in result.stdout
