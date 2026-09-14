#!/usr/bin/env python3
"""Add searched protein MSA and optional template paths to inference JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
repository_root = str(REPOSITORY_ROOT)
if repository_root in sys.path:
    sys.path.remove(repository_root)
sys.path.insert(0, repository_root)

from venusfold.data.msa.pipeline import prepare_input_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search protein MSAs and templates for a VenusFold input JSON."
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--msa-dir", type=Path, required=True)
    parser.add_argument("--msa-server-url")
    parser.add_argument("--email", default="")
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument(
        "--max-wait-seconds",
        type=int,
        default=1800,
        help="Maximum time to wait for one MSA service request (default: 1800).",
    )
    parser.add_argument("--templates", action="store_true")
    parser.add_argument("--hmmsearch-binary")
    parser.add_argument("--hmmbuild-binary")
    parser.add_argument("--seqres-database", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = prepare_input_json(
        args.input_json,
        args.output_json,
        args.msa_dir,
        host_url=args.msa_server_url,
        email=args.email,
        poll_interval=args.poll_interval,
        max_wait_seconds=args.max_wait_seconds,
        include_templates=args.templates,
        hmmsearch_binary=args.hmmsearch_binary,
        hmmbuild_binary=args.hmmbuild_binary,
        seqres_database=args.seqres_database,
    )
    print(f"Prepared input: {output}")


if __name__ == "__main__":
    main()
