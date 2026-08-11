#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VenusFold weights from Hugging Face.")
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("VENUSFOLD_HF_REPO", "AI4Protein/VenusFold"),
        help="Hugging Face model repository (default: AI4Protein/VenusFold).",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, default=Path("weights"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("model.safetensors", "model.json", "config.yaml"):
        path = hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            revision=args.revision,
            local_dir=args.output_dir,
            token=os.environ.get("HF_TOKEN"),
        )
        print(path)


if __name__ == "__main__":
    main()
