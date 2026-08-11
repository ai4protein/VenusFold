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

## Inference

Run the included example without MSA or templates:

```bash
export VENUSFOLD_ROOT_DIR="$PWD/data"
bash run_inference.sh \
  --input_json_path examples/input.json \
  --config_path configs/venusfold_inference.yaml \
  --checkpoint_path weights/model.safetensors \
  --dump_dir output/example \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --sample_diffusion.N_sample 1
```

Predicted structures and confidence files are written under the selected
`--dump_dir`.

The default config uses five samples, 200 diffusion steps, and ten recycling
cycles. This repository reads precomputed MSA and template files but does not
generate them. See [the input-format guide](docs/input_format.md) for supported
molecule types and optional MSA/template fields.

## License

The code is released under the MIT License. See [LICENSE](LICENSE). The model
is intended for research use and is not validated for clinical or diagnostic
use.
