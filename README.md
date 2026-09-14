# VenusFold

VenusFold predicts structures of biomolecular complexes containing proteins,
DNA, RNA, and small molecules. This repository contains the inference code;
model weights are hosted on
[Hugging Face](https://huggingface.co/AI4Protein/VenusFold).

## Technical Report

Read the [VenusFold Technical Report](docs/VenusFold_Technical_Report.pdf).

## Requirements

- Linux with Python 3.11
- NVIDIA GPU with a CUDA 12.x toolkit, including `nvcc`
- PyTorch 2.7.1
- `kalign` when template alignment is enabled

VenusFold compiles its fused layer-normalization CUDA extension on first use,
so a working C++ compiler is also required.

The released model has 464,442,431 parameters. It was tested on NVIDIA A800
80 GB GPUs; memory use depends on token count and sampling settings.

## Installation

```bash
git clone https://github.com/ai4protein/VenusFold.git
cd VenusFold

conda create -n venusfold python=3.11 -y
conda activate venusfold
pip install -r requirements.txt
pip install -e . --no-deps
```

Download the model weights and required CCD reference data:

```bash
python scripts/download_weights.py
export VENUSFOLD_ROOT_DIR="$PWD/data"
bash scripts/download_inference_data.sh
```

The weight downloader writes `model.safetensors`, `model.json`, and
`config.yaml` to `weights/`. The reference-data downloader writes to
`data/`.

When `use_msa: true` and a protein chain has no valid MSA input, inference
automatically searches the sequence before loading the model. Results are
cached under `<dump_dir>/input_features/<input-hash>/` and reused by later
runs. Set `auto_search_msa: false` to use the query-only fallback. When
`use_template: true`, automatic preparation also runs template search.

To prepare protein MSAs ahead of inference, or to also search templates,
install the HMMER command-line tools (`hmmbuild` and `hmmsearch`) and run:

```bash
export VENUSFOLD_ROOT_DIR="$PWD/data"
python scripts/prepare_msa_templates.py \
  --input-json examples/input.json \
  --output-json output/example/input_msa_template.json \
  --msa-dir output/example/msa \
  --templates
```

Protein MSAs are obtained from the MMseqs2 service configured by
`VENUSFOLD_MSA_SERVER_URL`. The default is the public service used by
VenusFold. Template search uses a local PDB SeqRes database; when it is not
present under `$VENUSFOLD_ROOT_DIR/search_database`, the preparation command
downloads it automatically. Use `--seqres-database`, `--hmmsearch-binary`,
and `--hmmbuild-binary` to provide explicit locations. The command validates
that the first sequence in every generated A3M matches the input query and
writes an adjacent `*_msa_template_audit.json` report.
The MSA request times out after 30 minutes by default; use
`--max-wait-seconds` to change this limit. Existing downloaded archives are
reused when the command is restarted. Set `VENUSFOLD_MSA_USE_ENV_PROXY=true`
only when the MSA service and template database must be reached through the
environment proxy.

## Inference

Run the included example with automatic MSA search and without templates:

```bash
export VENUSFOLD_ROOT_DIR="$PWD/data"
bash run_inference.sh \
  --input_json_path examples/input.json \
  --config_path configs/venusfold_inference.yaml \
  --checkpoint_path weights/model.safetensors \
  --dump_dir output/example \
  --use_msa true \
  --use_template false \
  --use_rna_msa false \
  --sample_diffusion.N_sample 1
```

Predicted structures and confidence files are written under the selected
`--dump_dir`.

The default config uses five samples, 200 diffusion steps, and ten recycling
cycles. See [the input-format guide](docs/input_format.md) for supported
molecule types and optional MSA/template fields.

## Acknowledgments

We gratefully acknowledge [Protenix by ByteDance](https://github.com/bytedance/Protenix),
which inspired and informed substantial portions of the code in this project.
The applicable upstream license is preserved in
[`licenses/Protenix-Apache-2.0.txt`](licenses/Protenix-Apache-2.0.txt).

## License

The code is released under the MIT License. See [LICENSE](LICENSE). The model
is intended for research use and is not validated for clinical or diagnostic
use.
