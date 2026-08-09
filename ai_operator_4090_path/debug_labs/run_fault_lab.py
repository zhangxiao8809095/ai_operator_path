#!/usr/bin/env python3
"""Run one isolated intentional CUDA fault; use one process per case."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


FAULT_DIR = Path(__file__).resolve().parent / "fault_extension"
sys.path.insert(0, str(FAULT_DIR))

import aiop4090_faults as faults  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["identity", "invalid-launch", "illegal-address",
                                           "oob", "race", "init"],
                        required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    x = torch.arange(1, 257, device="cuda", dtype=torch.float32)

    if args.case == "identity":
        out = faults.identity(x)
        torch.cuda.synchronize()
        if not torch.equal(out, x):
            raise RuntimeError("safe identity baseline failed")
        print("PASS safe fault-extension baseline")
    elif args.case == "invalid-launch":
        try:
            faults.invalid_launch(x)
        except RuntimeError as error:
            print("PASS invalid launch was reported at the launcher check")
            print(f"evidence: {error}")
        else:
            raise RuntimeError("invalid launch was not reported")
    elif args.case == "oob":
        faults.out_of_bounds(x)
        torch.cuda.synchronize()
        print("The process survived; memcheck must still report the intentional OOB write.")
    elif args.case == "illegal-address":
        faults.illegal_address(x)
        # With CUDA_LAUNCH_BLOCKING=0 the error normally surfaces here. With
        # CUDA_LAUNCH_BLOCKING=1 it normally surfaces in the extension call.
        torch.cuda.synchronize()
        raise RuntimeError("illegal-address case unexpectedly completed without a CUDA error")
    elif args.case == "race":
        faults.shared_race(x)
        torch.cuda.synchronize()
        print("racecheck must report intentional WAW/RAW hazards in shared_race_kernel.")
    elif args.case == "init":
        faults.uninitialized_read(x)
        torch.cuda.synchronize()
        print("initcheck must report reads from the raw uninitialized CUDA allocation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
