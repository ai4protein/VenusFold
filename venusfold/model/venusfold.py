import copy
import random
import time
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from venusfold.model import sample_confidence
from venusfold.model.generator import (
    InferenceNoiseScheduler,
    sample_diffusion,
)
from venusfold.model.modules.confidence import ConfidenceHead
from venusfold.model.modules.diffusion import DiffusionModule
from venusfold.model.modules.embedders import (
    ConstraintEmbedder,
    InputFeatureEmbedder,
    RelativePositionEncoding,
)
from venusfold.model.modules.head import DistogramHead
from venusfold.model.modules.pairformer import (
    MSAModule,
    PairformerStack,
    TemplateEmbedder,
)
from venusfold.model.modules.primitives import LinearNoBias
from venusfold.model.triangular.layers import LayerNorm
from venusfold.model.utils import simple_merge_dict_list
from venusfold.utils.logger import get_logger
from venusfold.utils.permutation.permutation import SymmetricPermutation
from venusfold.utils.torch_utils import autocasting_disable_decorator

logger = get_logger(__name__)


def update_input_feature_dict(input_feature_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Lines 1-3 of Algorithm 5 compute d_lm, v_lm, and pad_info utilized in the AtomAttentionEncoder.
    Args:
            input_feature_dict (dict[str, Any]): input features
    Returns:
            input_feature_dict (dict[str, Any]): input features
    """
    from venusfold.model.modules.transformer import rearrange_qk_to_dense_trunk

    with torch.no_grad():
        # Prepare tensors in dense trunks for local operations
        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
            q=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            k=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            dim_q=[-2, -1],
            dim_k=[-2, -1],
            n_queries=32,
            n_keys=128,
            compute_mask=True,
        )
        # Compute atom pair feature
        d_lm = (
            q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
        )  # [..., n_blocks, n_queries, n_keys, 3]
        v_lm = (
            q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()
        ).unsqueeze(
            dim=-1
        )  # [..., n_blocks, n_queries, n_keys, 1]
        input_feature_dict["d_lm"] = d_lm
        input_feature_dict["v_lm"] = v_lm
        input_feature_dict["pad_info"] = pad_info
        return input_feature_dict


class VenusFold(nn.Module):
    """
    Implements the AF3-style inference loop.
    """

    def __init__(self, configs: Any) -> None:
        super(VenusFold, self).__init__()
        self.configs = configs
        torch.backends.cuda.matmul.allow_tf32 = self.configs.enable_tf32
        # Some constants
        self.enable_diffusion_shared_vars_cache = (
            self.configs.enable_diffusion_shared_vars_cache
        )
        self.enable_efficient_fusion = self.configs.enable_efficient_fusion
        self.N_cycle = self.configs.model.N_cycle
        self.N_model_seed = self.configs.model.N_model_seed
        # Diffusion scheduler
        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )

        # Model
        local_activation = configs.model.local_activation
        esm_configs = configs.get("esm", {})  # This is used in InputFeatureEmbedder
        input_embedder_config = copy.deepcopy(configs.model.input_embedder)
        input_embedder_config.local_activation = local_activation
        template_embedder_config = copy.deepcopy(configs.model.template_embedder)
        template_embedder_config.local_activation = local_activation
        constraint_embedder_config = copy.deepcopy(configs.model.constraint_embedder)
        constraint_embedder_config.local_activation = local_activation
        diffusion_module_config = copy.deepcopy(configs.model.diffusion_module)
        diffusion_module_config.local_activation = local_activation
        self.input_embedder = InputFeatureEmbedder(
            **input_embedder_config, esm_configs=esm_configs
        )
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(**template_embedder_config)
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data.get("msa", {}),
        )
        self.constraint_embedder = ConstraintEmbedder(
            **constraint_embedder_config
        )
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)
        self.diffusion_module = DiffusionModule(**diffusion_module_config)
        self.distogram_head = DistogramHead(**configs.model.distogram_head)
        self.confidence_head = ConfidenceHead(**configs.model.confidence_head)

        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )
        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)

        # Zero init the recycling layer
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

    def get_pairformer_output(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        mc_dropout: bool = False,
        mc_dropout_rate: float = 0.4,
    ) -> tuple[torch.Tensor, ...]:
        """
        The forward pass from the input to pairformer output

        Args:
            input_feature_dict (dict[str, Any]): input features
            N_cycle (int): number of cycles
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            Tuple[torch.Tensor, ...]: s_inputs, s, z
        """
        # Line 1-5
        s_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 449]
        z_constraint = None

        if "constraint_feature" in input_feature_dict:
            z_constraint = self.constraint_embedder(
                input_feature_dict["constraint_feature"]
            )

        s_init = self.linear_no_bias_sinit(s_inputs)  # [..., N_token, c_s]
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  # [..., N_token, N_token, c_z]
        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict["relp"])
            z_init += self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init += z_constraint
        else:
            z_init = z_init + self.relative_position_encoding(
                input_feature_dict["relp"]
            )
            z_init = z_init + self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init = z_init + z_constraint
        # Line 6
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        # Line 7-13 recycling
        for cycle_no in range(N_cycle):
            with torch.no_grad():
                if mc_dropout:
                    z = z_init + F.dropout(
                        self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z)),
                        p=self.configs.mc_dropout_rate,
                    )
                else:
                    z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                if inplace_safe:
                    if self.template_embedder.n_blocks > 0:
                        z += self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                else:
                    if self.template_embedder.n_blocks > 0:
                        z = z + self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                s, z = self.pairformer_stack(
                    s,
                    z,
                    pair_mask=None,
                    triangle_multiplicative=self.configs.triangle_multiplicative,
                    triangle_attention=self.configs.triangle_attention,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

        return s_inputs, s, z

    def sample_diffusion(self, **kwargs: Any) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )
        _configs.update(
            {
                "guidance_configs": self.configs.sample_diffusion.to_dict().get(
                    "guidance"
                )
            }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def run_confidence_head(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs the confidence head with optional automatic mixed precision (AMP) disabled.

        Returns:
            Any: The output of the confidence head.
        """
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.confidence_head
        )(*args, **kwargs)

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        N_model_seed: int = 1,
        symmetric_permutation: SymmetricPermutation = None,
        mc_dropout_apply_rate: float = 0.4,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (multiple model seeds) for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_dict (dict[str, Any]): Label dictionary.
            N_cycle (int): Number of cycles.
            mode (str): Mode of operation (e.g., 'inference').
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to 4.
            N_model_seed (int): Number of model seeds. Defaults to 1.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.
            mc_dropout_apply_rate (float): Only for inference mode

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        # For backward compatibility, if N_model_seed > 1, process multiple seeds here
        # But in evaluation mode, this should be handled externally
        if N_model_seed > 1 and mode in ["inference"]:
            pred_dicts = []
            log_dicts = []
            time_trackers = []
            for _ in range(N_model_seed):
                pred_dict, log_dict, time_tracker = self._main_inference_loop(
                    input_feature_dict=(
                        copy.deepcopy(input_feature_dict)
                        if (N_model_seed > 1 and mode == "inference")
                        else input_feature_dict
                    ),  # the input_feature_dict is modified when mode is "inference"
                    label_dict=label_dict,
                    N_cycle=N_cycle,
                    mode=mode,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                    symmetric_permutation=symmetric_permutation,
                    mc_dropout=random.random() < mc_dropout_apply_rate,
                )
                pred_dicts.append(pred_dict)
                log_dicts.append(log_dict)
                time_trackers.append(time_tracker)

            # Combine outputs of multiple models
            def _cat(dict_list, key):
                return torch.cat([x[key] for x in dict_list], dim=0)

            def _list_join(dict_list, key):
                return sum([x[key] for x in dict_list], [])

            all_pred_dict = {
                "coordinate": _cat(pred_dicts, "coordinate"),
                "summary_confidence": _list_join(pred_dicts, "summary_confidence"),
                "full_data": _list_join(pred_dicts, "full_data"),
                "plddt": _cat(pred_dicts, "plddt"),
                "pae": _cat(pred_dicts, "pae"),
                "pde": _cat(pred_dicts, "pde"),
                "resolved": _cat(pred_dicts, "resolved"),
            }

            all_log_dict = simple_merge_dict_list(log_dicts)
            all_time_dict = simple_merge_dict_list(time_trackers)
            return all_pred_dict, all_log_dict, all_time_dict
        else:
            # Single seed inference - delegate to _main_inference_loop
            return self._main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                symmetric_permutation=symmetric_permutation,
                mc_dropout=random.random() < mc_dropout_apply_rate,
            )

    def _get_dynamic_chunk_size(self, N_token: int) -> Optional[int]:
        """
        Get dynamic chunk_size based on token count

        Args:
            N_token (int): Number of tokens

        Returns:
            Optional[int]: Optimal chunk_size for the given token count
        """
        if not hasattr(self.configs.infer_setting, "chunk_size_thresholds"):
            return self.configs.infer_setting.chunk_size

        thresholds = self.configs.infer_setting.chunk_size_thresholds

        # Convert string keys to integers and sort in ascending order
        threshold_pairs = [(int(k), v) for k, v in thresholds.items()]
        sorted_thresholds = sorted(threshold_pairs, key=lambda x: x[0])

        # Find the appropriate chunk_size for the given token count
        for threshold, chunk_size in sorted_thresholds:
            if N_token <= threshold:
                return None if chunk_size == -1 else chunk_size

        # For token counts larger than the largest threshold, use smallest chunk_size
        return 32  # extreme case for very large proteins

    def _main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        symmetric_permutation: SymmetricPermutation = None,
        mc_dropout: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (single model seed) for the Alphafold3 model.
        mc_dropout: do not use by default

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        step_st = time.time()
        N_token = input_feature_dict["residue_index"].shape[-1]

        # Apply dynamic chunk_size if enabled (otherwise keep the passed chunk_size)
        if (
            hasattr(self.configs.infer_setting, "dynamic_chunk_size")
            and self.configs.infer_setting.dynamic_chunk_size
        ):
            chunk_size = self._get_dynamic_chunk_size(N_token)
        # If dynamic chunking is disabled, chunk_size keeps its original value from the function parameter

        log_dict = {}
        pred_dict = {}
        time_tracker = {}

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            mc_dropout=mc_dropout,
        )

        keys_to_delete = []
        for key in input_feature_dict.keys():
            if "template_" in key or key in [
                "msa",
                "has_deletion",
                "deletion_value",
                "profile",
                "deletion_mean",
                # "token_bonds",
            ]:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del input_feature_dict[key]
        step_trunk = time.time()
        time_tracker.update({"pairformer": step_trunk - step_st})
        # Sample diffusion
        # [..., N_sample, N_atom, 3]
        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]

        noise_schedule = self.inference_noise_scheduler(
            N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype
        )
        cache = dict()
        if self.enable_diffusion_shared_vars_cache:
            # line 1-5 of algorithm 21 calculate z in diffusion conditioning
            cache["pair_z"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.diffusion_conditioning.prepare_cache)(
                input_feature_dict["relp"], z, False
            )
            cache["p_lm/c_l"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.atom_attention_encoder.prepare_cache)(
                ref_pos=input_feature_dict["ref_pos"],
                ref_charge=input_feature_dict["ref_charge"],
                ref_mask=input_feature_dict["ref_mask"],
                ref_element=input_feature_dict["ref_element"],
                ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                d_lm=input_feature_dict["d_lm"],
                v_lm=input_feature_dict["v_lm"],
                pad_info=input_feature_dict["pad_info"],
                r_l=True,
                z=cache["pair_z"],
                inplace_safe=False,
            )
        else:
            cache["pair_z"] = None
            cache["p_lm/c_l"] = [None, None]
        pred_dict["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=None if cache["pair_z"] is not None else z,
            pair_z=cache["pair_z"],
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            N_sample=N_sample,
            noise_schedule=noise_schedule,
            inplace_safe=inplace_safe,
            enable_efficient_fusion=self.enable_efficient_fusion,
        )

        step_diffusion = time.time()
        time_tracker.update({"diffusion": step_diffusion - step_trunk})
        # Distogram logits: log contact_probs only, to reduce the dimension
        pred_dict["contact_probs"] = autocasting_disable_decorator(True)(
            sample_confidence.compute_contact_prob
        )(
            distogram_logits=self.distogram_head(z),
            **sample_confidence.get_bin_params(self.configs.loss.distogram),
        )  # [N_token, N_token]

        # Confidence logits
        (
            pred_dict["plddt"],
            pred_dict["pae"],
            pred_dict["pde"],
            pred_dict["resolved"],
        ) = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=pred_dict["coordinate"],
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        step_confidence = time.time()
        time_tracker.update({"confidence": step_confidence - step_diffusion})
        time_tracker.update({"model_forward": time.time() - step_st})

        # Permutation: when label is given, permute coordinates and other heads
        if label_dict is not None and symmetric_permutation is not None:
            pred_dict, log_dict = symmetric_permutation.permute_inference_pred_dict(
                input_feature_dict=input_feature_dict,
                pred_dict=pred_dict,
                label_dict=label_dict,
                permute_by_pocket=("pocket_mask" in label_dict)
                and ("interested_ligand_mask" in label_dict),
            )
            last_step_seconds = step_confidence
            time_tracker.update({"permutation": time.time() - last_step_seconds})

        # Summary Confidence & Full Data
        # Computed after coordinates and logits are permuted
        if label_dict is None:
            interested_atom_mask = None
        else:
            interested_atom_mask = label_dict.get("interested_ligand_mask", None)
        (
            pred_dict["summary_confidence"],
            pred_dict["full_data"],
        ) = autocasting_disable_decorator(True)(
            sample_confidence.compute_full_data_and_summary
        )(
            configs=self.configs,
            pae_logits=pred_dict["pae"],
            plddt_logits=pred_dict["plddt"],
            pde_logits=pred_dict["pde"],
            contact_probs=pred_dict.get(
                "per_sample_contact_probs", pred_dict["contact_probs"]
            ),
            token_asym_id=input_feature_dict["asym_id"],
            token_has_frame=input_feature_dict["has_frame"],
            atom_coordinate=pred_dict["coordinate"],
            atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            atom_is_polymer=1 - input_feature_dict["is_ligand"],
            N_recycle=N_cycle,
            interested_atom_mask=interested_atom_mask,
            return_full_data=True,
            mol_id=(input_feature_dict["mol_id"] if mode != "inference" else None),
            elements_one_hot=(
                input_feature_dict["ref_element"] if mode != "inference" else None
            ),
        )

        return pred_dict, log_dict, time_tracker

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        mode: str = "inference",
        symmetric_permutation: SymmetricPermutation = None,
        disable_inplace: bool = False,
        mc_dropout_apply_rate: float = 0.4,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Run structure prediction or label-aware evaluation.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict[str, Any]): Label dictionary (cropped).
            mode (str): Either 'inference' or 'eval'. Defaults to 'inference'.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        if mode not in ["eval", "inference"]:
            raise ValueError(f"Inference-only release does not support mode={mode!r}")
        not_use_gradient = not (self.training or torch.is_grad_enabled())
        inplace_safe = not_use_gradient and (not disable_inplace)

        input_feature_dict = self.relative_position_encoding.generate_relp(
            input_feature_dict
        )
        input_feature_dict = update_input_feature_dict(input_feature_dict)

        if mode == "inference":
            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=None,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=self.configs.infer_setting.chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=None,
                mc_dropout_apply_rate=mc_dropout_apply_rate,
            )
            log_dict.update({"time": time_tracker})
        elif mode == "eval":
            if label_dict is not None:
                assert (
                    label_dict["coordinate"].size()
                    == label_full_dict["coordinate"].size()
                )
                label_dict.update(label_full_dict)

            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=self.configs.infer_setting.chunk_size,
                N_model_seed=1,
                symmetric_permutation=symmetric_permutation,
                mc_dropout_apply_rate=mc_dropout_apply_rate,
            )
            log_dict.update({"time": time_tracker})

        return pred_dict, label_dict, log_dict
