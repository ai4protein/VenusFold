import os
from pathlib import Path

from venusfold.config.extend_types import ListValue, RequiredValue

VENUSFOLD_ROOT_DIR = os.environ.get("VENUSFOLD_ROOT_DIR", str(Path.home()))

inference_configs = {
    "model_name": "venusfold",
    "seeds": ListValue([101]),
    "dump_dir": "./output/inference",
    "need_atom_confidence": False,
    "sorted_by_ranking_score": True,
    "input_json_path": RequiredValue(str),
    "config_path": "",
    "load_checkpoint_dir": os.path.join(VENUSFOLD_ROOT_DIR, "checkpoint"),
    "checkpoint_path": "",
    "dry_run": False,
    "num_workers": 0,
    "use_msa": True,
    "enable_tf32": True,
    "enable_efficient_fusion": True,
    "enable_diffusion_shared_vars_cache": True,
    "msa_pair_as_unpair": True,
    "use_template": False,
    "use_rna_msa": False,
    "use_seeds_in_json": False,
}
