"""Read CUDA readiness; optional tiny allocation probe, never model training."""

import argparse
import json
import sys

import torch
import torchvision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    available = torch.cuda.is_available()
    result = {
        "python_executable": sys.executable,
        "torch_path": torch.__file__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": available,
    }
    if available:
        result.update(
            gpu=torch.cuda.get_device_name(0),
            memory_mb=torch.cuda.get_device_properties(0).total_memory / 2**20,
        )
    if args.probe and available:
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            value = torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda")
            result["probe_passed"] = bool(torch.all(value == 32).item())
        del value
    print(json.dumps(result, indent=2))
    if args.require_cuda and not available:
        if torch.version.cuda is None:
            raise SystemExit(
                "CPU wheel detected. Install torch==2.9.1+cu126 and "
                "torchvision==0.24.1+cu126 in .venv-cuda; see POC_REAL_AI.md."
            )
        raise SystemExit(
            "CUDA wheel present but GPU unavailable. Check driver/nvidia-smi before training."
        )


if __name__ == "__main__":
    main()
