from __future__ import annotations

import unittest

from vllm_lab import DEFAULT_MODEL, LabConfig


class LabConfigTests(unittest.TestCase):
    def test_defaults_are_single_gpu_4090_friendly(self) -> None:
        config = LabConfig.from_env({})
        self.assertEqual(config.model, DEFAULT_MODEL)
        self.assertEqual(config.tensor_parallel_size, 1)
        self.assertEqual(config.max_model_len, 4096)
        self.assertLess(config.gpu_memory_utilization, 1)

    def test_environment_overrides(self) -> None:
        config = LabConfig.from_env(
            {
                "VLLM_MODEL": "/models/local",
                "VLLM_DTYPE": "float16",
                "VLLM_TENSOR_PARALLEL_SIZE": "2",
                "VLLM_GPU_MEMORY_UTILIZATION": "0.75",
                "VLLM_MAX_MODEL_LEN": "2048",
                "VLLM_SEED": "7",
                "VLLM_TRUST_REMOTE_CODE": "yes",
            }
        )
        self.assertEqual(config.model, "/models/local")
        self.assertEqual(config.tensor_parallel_size, 2)
        self.assertEqual(config.gpu_memory_utilization, 0.75)
        self.assertTrue(config.trust_remote_code)
        self.assertEqual(config.llm_kwargs()["seed"], 7)

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "tensor_parallel_size"):
            LabConfig.from_env({"VLLM_TENSOR_PARALLEL_SIZE": "0"})
        with self.assertRaisesRegex(ValueError, "gpu_memory_utilization"):
            LabConfig.from_env({"VLLM_GPU_MEMORY_UTILIZATION": "1.1"})
        with self.assertRaisesRegex(ValueError, "true/false"):
            LabConfig.from_env({"VLLM_TRUST_REMOTE_CODE": "maybe"})


if __name__ == "__main__":
    unittest.main()
