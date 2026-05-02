"""
Portable CPU / GPU compute % for GET /anymatix/host_compute_metrics.

- CPU: psutil (Linux, macOS, Windows; already a ComfyUI dependency).
- GPU: NVIDIA via nvidia-smi when available; optional AMD on Linux via rocm-smi.
  Apple Silicon and other stacks return gpu=null when no probe applies.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from typing import Any


def sample_host_compute_metrics() -> dict[str, Any]:
    return {"cpu": _sample_cpu_percent(), "gpu": _sample_gpu_percent()}


def _clamp_pct(x: int | None) -> int | None:
    if x is None:
        return None
    return max(0, min(100, int(x)))


def _subprocess_kw() -> dict[str, Any]:
    if sys.platform == "win32":
        f = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if f:
            return {"creationflags": f}
    return {}


def _sample_cpu_percent() -> int | None:
    try:
        import psutil

        v = int(psutil.cpu_percent(interval=0.2))
        return _clamp_pct(v)
    except Exception:
        return None


def _sample_gpu_percent() -> int | None:
    v = _gpu_nvidia_smi()
    if v is not None:
        return v
    return _gpu_rocm_smi()


def _gpu_nvidia_smi() -> int | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [
                exe,
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            **_subprocess_kw(),
        )
        if r.returncode != 0 or not (r.stdout or "").strip():
            return None
        line = (r.stdout or "").strip().splitlines()[0]
        nums = re.findall(r"\d+", line)
        if not nums:
            return None
        return _clamp_pct(int(nums[0]))
    except Exception:
        return None


def _gpu_rocm_smi() -> int | None:
    if sys.platform == "win32":
        return None
    exe = shutil.which("rocm-smi")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "--showuse"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if r.returncode != 0:
            return None
        m = re.search(r"(\d+)\s*%", r.stdout or "")
        if m:
            return _clamp_pct(int(m.group(1)))
    except Exception:
        pass
    return None
