"""Export a tensor-only model state from a trusted local trainer checkpoint.

The input checkpoint must be produced by this repository's trainer.  Loading a
pickle with ``weights_only=False`` is intentionally isolated here; inference
adapters continue to accept only tensor-only checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def export_safe_checkpoint(input_path: Path, output_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise TypeError("trusted trainer checkpoint must contain a state_dict")
    state_dict = checkpoint["state_dict"]
    if not state_dict or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise TypeError("state_dict must be a non-empty string-to-tensor mapping")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict}, output_path)

    verified = torch.load(output_path, map_location="cpu", weights_only=True)
    if set(verified.get("state_dict", {})) != set(state_dict):
        raise RuntimeError("safe checkpoint verification failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_safe_checkpoint(args.input.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
