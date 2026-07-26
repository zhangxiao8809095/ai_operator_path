"""Environment-driven settings shared by offline and online examples."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _parse_bool(value: str, variable: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{variable} must be true/false, got {value!r}")


@dataclass(frozen=True)
class LabConfig:
    """Validated settings suitable for a 24GB single-GPU learning machine."""

    model: str = DEFAULT_MODEL
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    seed: int = 42
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be at least 1")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in the range (0, 1]")
        if self.max_model_len < 1:
            raise ValueError("max_model_len must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> LabConfig:
        """Create settings from ``VLLM_*`` environment variables."""

        env = os.environ if environ is None else environ
        return cls(
            model=env.get("VLLM_MODEL", DEFAULT_MODEL),
            dtype=env.get("VLLM_DTYPE", "auto"),
            tensor_parallel_size=int(env.get("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            gpu_memory_utilization=float(env.get("VLLM_GPU_MEMORY_UTILIZATION", "0.85")),
            max_model_len=int(env.get("VLLM_MAX_MODEL_LEN", "4096")),
            seed=int(env.get("VLLM_SEED", "42")),
            trust_remote_code=_parse_bool(
                env.get("VLLM_TRUST_REMOTE_CODE", "false"),
                "VLLM_TRUST_REMOTE_CODE",
            ),
        )

    def llm_kwargs(self) -> dict[str, object]:
        """Return the common keyword arguments accepted by ``vllm.LLM``."""

        return {
            "model": self.model,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "seed": self.seed,
            "trust_remote_code": self.trust_remote_code,
        }
