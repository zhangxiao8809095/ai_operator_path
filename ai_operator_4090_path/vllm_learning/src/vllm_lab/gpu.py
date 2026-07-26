"""Small nvidia-smi wrapper for educational memory snapshots."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    name: str
    memory_used_mib: int
    memory_total_mib: int
    utilization_percent: int

    @property
    def memory_free_mib(self) -> int:
        return self.memory_total_mib - self.memory_used_mib


def take_gpu_snapshots() -> list[GpuSnapshot]:
    """Return one snapshot per visible NVIDIA GPU, or an empty list without nvidia-smi."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []

    query = (
        "index,name,memory.used,memory.total,utilization.gpu"
    )
    completed = subprocess.run(
        [
            executable,
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    snapshots: list[GpuSnapshot] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        index, name, used, total, utilization = (part.strip() for part in line.split(",", 4))
        snapshots.append(
            GpuSnapshot(
                index=int(index),
                name=name,
                memory_used_mib=int(used),
                memory_total_mib=int(total),
                utilization_percent=int(utilization),
            )
        )
    return snapshots


def format_snapshots(label: str, snapshots: list[GpuSnapshot]) -> str:
    if not snapshots:
        return f"[{label}] nvidia-smi unavailable or no NVIDIA GPU visible"
    lines = [f"[{label}]"]
    for item in snapshots:
        lines.append(
            f"  GPU {item.index} {item.name}: "
            f"{item.memory_used_mib}/{item.memory_total_mib} MiB used, "
            f"{item.memory_free_mib} MiB free, utilization {item.utilization_percent}%"
        )
    return "\n".join(lines)
