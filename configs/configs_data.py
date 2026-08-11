import os
from pathlib import Path

from venusfold.config.extend_types import ListValue


VENUSFOLD_ROOT_DIR = os.environ.get("VENUSFOLD_ROOT_DIR", str(Path.home()))

data_configs = {
    "msa": {
        "enable_prot_msa": True,
        "prot_seq_or_filename_to_msadir_jsons": ListValue(
            [os.path.join(VENUSFOLD_ROOT_DIR, "common/seq_to_pdb_index.json")]
        ),
        "prot_msadir_raw_paths": ListValue(
            [os.path.join(VENUSFOLD_ROOT_DIR, "mmcif_msa_template")]
        ),
        "prot_pairing_dbs": ListValue(["pairing"]),
        "prot_non_pairing_dbs": ListValue(["pairing-non_pairing"]),
        "prot_indexing_methods": ListValue(["sequence"]),
        "enable_rna_msa": True,
        "rna_seq_or_filename_to_msadir_jsons": ListValue(
            [
                os.path.join(
                    VENUSFOLD_ROOT_DIR,
                    "rna_msa/rna_sequence_to_pdb_chains.json",
                )
            ]
        ),
        "rna_msadir_raw_paths": ListValue(
            [os.path.join(VENUSFOLD_ROOT_DIR, "rna_msa/msas")]
        ),
        "rna_indexing_methods": ListValue(["sequence"]),
        "min_size": {"train": 1, "test": 1},
        "max_size": {"train": 16384, "test": 16384},
        "sample_cutoff": {"train": 16384, "test": 16384},
    },
    "template": {
        "enable_prot_template": True,
        "template_dropout_rate": 0.0,
        "prot_template_mmcif_dir": os.path.join(VENUSFOLD_ROOT_DIR, "mmcif"),
        "prot_template_cache_dir": "",
        "prot_template_raw_paths": ListValue(
            [os.path.join(VENUSFOLD_ROOT_DIR, "mmcif_msa_template")]
        ),
        "prot_seq_or_filename_to_templatedir_jsons": ListValue(
            [os.path.join(VENUSFOLD_ROOT_DIR, "common/seq_to_pdb_index.json")]
        ),
        "prot_indexing_methods": ListValue(["sequence"]),
        "release_dates_path": os.path.join(
            VENUSFOLD_ROOT_DIR, "common/release_date_cache.json"
        ),
        "obsolete_pdbs_path": os.path.join(
            VENUSFOLD_ROOT_DIR, "common/obsolete_to_successor.json"
        ),
        "kalign_binary_path": os.environ.get("KALIGN_BINARY_PATH", "/usr/bin/kalign"),
        "fetch_remote": True,
    },
    "ccd_components_file": os.path.join(
        VENUSFOLD_ROOT_DIR, "common/components.cif"
    ),
    "ccd_components_rdkit_mol_file": os.path.join(
        VENUSFOLD_ROOT_DIR, "common/components.cif.rdkit_mol.pkl"
    ),
}

