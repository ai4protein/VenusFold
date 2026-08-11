"""Architecture settings for the released VenusFold checkpoint."""

model_configs = {
    "venusfold": {
        "c_z": 256,
        "model": {
            "N_cycle": 10,
            "local_activation": "silu",
            "relative_position_encoding": {"c_z": 256},
            "template_embedder": {
                "c_z": 256,
                "n_blocks": 2,
                "hidden_scale_up": True,
            },
            "msa_module": {
                "c_m": 128,
                "c_z": 256,
                "hidden_scale_up": True,
            },
            "pairformer": {
                "c_z": 256,
                "hidden_scale_up": True,
            },
            "diffusion_module": {"c_z": 256},
            "confidence_head": {
                "c_z": 256,
                "hidden_scale_up": True,
            },
            "distogram_head": {"c_z": 256},
        },
        "sample_diffusion": {"N_step": 200},
    }
}

