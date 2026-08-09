#!/usr/bin/env python3
"""Safe build/integration fault exercises; does not modify the working tree."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def missing_export() -> None:
    extension = importlib.import_module("aiop4090._C")
    symbol = "debug_symbol_that_is_not_bound"
    try:
        getattr(extension, symbol)
    except AttributeError as error:
        print("PASS missing-export: classified at Python/binding export layer")
        print(f"evidence: {error}")
        return
    raise RuntimeError("missing-export lab did not produce AttributeError")


def stale_binary() -> None:
    extension = importlib.import_module("aiop4090._C")
    binary = Path(extension.__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="aiop4090-stale-") as tmp:
        marker = Path(tmp) / "changed_source.cu"
        marker.write_text("// simulated source edit in an isolated directory\n", encoding="utf-8")
        future = max(time.time(), binary.stat().st_mtime + 2.0)
        os.utime(marker, (future, future))
        stale = marker.stat().st_mtime > binary.stat().st_mtime + 1.0
        if not stale:
            raise RuntimeError("stale-binary lab failed to create the timestamp condition")
        print("PASS stale-binary: source mtime is newer than loaded extension")
        print(f"loaded extension: {binary}")
        print(f"isolated changed source: {marker}")


def source_omission() -> None:
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    production_sources = sorted((ROOT / "src/aiop4090/csrc").glob("*.cu"))
    omitted = [str(path.relative_to(ROOT)) for path in production_sources
               if str(path.relative_to(ROOT)) not in setup_text]
    if omitted:
        raise RuntimeError(f"real build source omission detected: {omitted}")
    with tempfile.TemporaryDirectory(prefix="aiop4090-source-") as tmp:
        extra = Path(tmp) / "forgotten_kernel.cu"
        extra.write_text("// intentionally absent from setup.py\n", encoding="utf-8")
        listed = extra.name in setup_text
        if listed:
            raise RuntimeError("source-omission simulation unexpectedly appeared in setup.py")
        print("PASS source-omission: detected a CUDA file absent from the declared source list")
        print("production setup.py currently lists every production .cu file")


def undefined_symbol() -> None:
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("a C++ compiler is required for undefined-symbol lab")
    with tempfile.TemporaryDirectory(prefix="aiop4090-symbol-") as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "broken.cpp"
        library = tmp_path / ("libbroken.dylib" if sys.platform == "darwin" else "libbroken.so")
        source.write_text(
            'extern "C" void aiop4090_missing_dependency();\n'
            'extern "C" int aiop4090_entry() { aiop4090_missing_dependency(); return 0; }\n',
            encoding="utf-8",
        )
        command = [compiler, "-shared", "-fPIC", str(source), "-o", str(library)]
        if sys.platform == "darwin":
            command.extend(["-Wl,-undefined,dynamic_lookup"])
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True)
        try:
            ctypes.CDLL(str(library), mode=getattr(os, "RTLD_NOW", 2))
        except OSError as error:
            if "symbol" not in str(error).lower():
                raise
            print("PASS undefined-symbol: dynamic loader rejected an unresolved dependency")
            print(f"evidence: {error}")
            return
        raise RuntimeError("undefined-symbol lab unexpectedly loaded the broken library")


def arch_config() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for arch-config lab")
    capability = torch.cuda.get_device_capability()
    configured = os.environ.get("TORCH_CUDA_ARCH_LIST", "<unset>")
    expected = f"{capability[0]}.{capability[1]}"
    print(f"device capability: {expected}")
    print(f"TORCH_CUDA_ARCH_LIST: {configured}")
    if capability != (8, 9):
        raise RuntimeError("this acceptance path expects RTX 4090 capability 8.9")
    configured_arches = configured.replace(";", " ").replace(",", " ").split()
    normalized_arches = {arch.removesuffix("+PTX") for arch in configured_arches}
    if expected not in normalized_arches:
        raise RuntimeError(
            f"TORCH_CUDA_ARCH_LIST must include {expected} for this build path; got {configured}"
        )
    print("PASS arch-config: runtime device is sm_89; build diagnostics must verify sm_89 code")


CASES = {
    "missing-export": missing_export,
    "stale-binary": stale_binary,
    "source-omission": source_omission,
    "undefined-symbol": undefined_symbol,
    "arch-config": arch_config,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    args = parser.parse_args()
    selected = CASES.values() if args.case == "all" else [CASES[args.case]]
    for run in selected:
        run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
