#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${VENUSFOLD_ROOT_DIR:-${repo_dir}/data}"
data_url="${VENUSFOLD_DATA_URL:-https://huggingface.co/AI4Protein/VenusFold/resolve/main/inference-data/common.tar.gz}"
archive="${data_root}/common.tar.gz"

mkdir -p "${data_root}" "${data_root}/mmcif"
if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 3 -C - -o "${archive}" "${data_url}"
elif command -v wget >/dev/null 2>&1; then
  wget -c -O "${archive}" "${data_url}"
else
  echo "curl or wget is required" >&2
  exit 1
fi
tar -xzf "${archive}" -C "${data_root}"
rm -f "${archive}"
echo "Inference data installed in ${data_root}"
