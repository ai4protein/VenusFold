#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"
export VENUSFOLD_ROOT_DIR="${VENUSFOLD_ROOT_DIR:-${repo_dir}/data}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-fast_layernorm}"

python_bin="${VENUSFOLD_PYTHON:-python3}"
exec "${python_bin}" -m runner.inference "$@"

