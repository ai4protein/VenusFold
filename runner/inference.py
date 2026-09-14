import hashlib
import json
import logging
import os
import time
import traceback
from argparse import Namespace
from contextlib import nullcontext
from os.path import exists as opexists, join as opjoin
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from venusfold.config.config import load_config, parse_configs, parse_sys_args
from venusfold.data.inference.infer_dataloader import get_inference_dataloader
from venusfold.utils.distributed import DIST_WRAPPER
from venusfold.utils.seed import seed_everything
from venusfold.utils.torch_utils import to_device

from runner.dumper import DataDumper

logger = logging.getLogger(__name__)

torch.serialization.add_safe_globals([Namespace])


def deep_update(dst: dict, src: Mapping[str, Any]) -> dict:
    for key, value in src.items():
        if isinstance(value, Mapping) and key in dst and isinstance(dst[key], Mapping):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def cli_has_key(arg_str: str, key: str) -> bool:
    return f"--{key} " in f"{arg_str} "


def resolve_checkpoint_path(configs: Any) -> str:
    if configs.checkpoint_path:
        return configs.checkpoint_path
    return opjoin(configs.load_checkpoint_dir, f"{configs.model_name}.pt")


def prepare_missing_msas(configs: Any) -> None:
    if not configs.use_msa or not configs.auto_search_msa:
        return

    import fcntl

    from venusfold.data.msa.pipeline import prepare_input_json

    input_path = Path(configs.input_json_path).resolve()
    fingerprint = hashlib.sha256(input_path.read_bytes()).hexdigest()[:16]
    cache_root = (
        Path(configs.msa_search_dir).resolve()
        if configs.msa_search_dir
        else Path(configs.dump_dir).resolve() / "input_features"
    )
    work_dir = cache_root / fingerprint
    work_dir.mkdir(parents=True, exist_ok=True)
    output_json = work_dir / f"{input_path.stem}_with_msa.json"

    with (work_dir / ".prepare_msa.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepared_path = prepare_input_json(
            input_path,
            output_json,
            work_dir / "msa",
            host_url=configs.msa_server_url or None,
            email=configs.msa_search_email,
            poll_interval=configs.msa_poll_interval,
            max_wait_seconds=configs.msa_max_wait_seconds,
            include_templates=configs.use_template,
        )
    configs.input_json_path = str(prepared_path)
    logger.info("Inference input with MSA: %s", prepared_path)


class InferenceRunner:
    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper(
            need_atom_confidence=configs.need_atom_confidence,
            sorted_by_ranking_score=configs.sorted_by_ranking_score,
        )

    def init_env(self) -> None:
        self.print(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device(f"cuda:{DIST_WRAPPER.local_rank}")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if DIST_WRAPPER.world_size > 1:
            dist.init_process_group(backend="nccl")

        if self.configs.triangle_attention == "deepspeed":
            cutlass_path = os.getenv("CUTLASS_PATH")
            assert cutlass_path is not None, (
                "CUTLASS_PATH must be set when triangle_attention=deepspeed."
            )

        if os.getenv("LAYERNORM_TYPE", "fast_layernorm") == "fast_layernorm":
            logging.info("fast_layernorm kernels may compile on first use.")

        logging.info("Finished environment initialization.")

    def init_basics(self) -> None:
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def init_model(self) -> None:
        from venusfold.model.venusfold import VenusFold

        self.model = VenusFold(self.configs).to(self.device)

    def load_checkpoint(self) -> None:
        checkpoint_path = resolve_checkpoint_path(self.configs)
        if not opexists(checkpoint_path):
            raise FileNotFoundError(f"Given checkpoint path does not exist: {checkpoint_path}")

        self.print(f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}")
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(checkpoint_path, device=str(self.device))
        else:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            state_dict = checkpoint.get("model", checkpoint)

        sample_key = list(state_dict.keys())[0]
        self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict=state_dict, strict=self.configs.load_strict)
        self.model.eval()
        self.print("Finished loading checkpoint.")
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        self.print(f"Model parameters: {total_params:.2f}M")

    def init_dumper(
        self, need_atom_confidence: bool = False, sorted_by_ranking_score: bool = True
    ) -> None:
        self.dumper = DataDumper(
            base_dir=self.dump_dir,
            need_atom_confidence=need_atom_confidence,
            sorted_by_ranking_score=sorted_by_ranking_score,
        )

    @torch.no_grad()
    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]

        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )

        data = to_device(data, self.device)
        with enable_amp:
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
                mc_dropout_apply_rate=self.configs.mc_dropout_apply_rate,
            )
        return prediction

    def print(self, msg: str) -> None:
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)

    def update_model_configs(self, new_configs: Any) -> None:
        self.model.configs = new_configs


def update_inference_configs(configs: Any, n_token: int) -> Any:
    if n_token > 2560 and configs.model_name in ["venusfold", "venusfold-v2"]:
        raise AssertionError("venusfold scaled model does not support n_token > 2560.")

    if n_token > 3840:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = False
    elif n_token > 2560:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = True
    else:
        configs.skip_amp.confidence_head = False if configs.model_name in ["venusfold", "venusfold-v2"] else True
        configs.skip_amp.sample_diffusion = True
    return configs


def infer_predict(runner: InferenceRunner, configs: Any) -> None:
    logger.info(f"Loading data from {configs.input_json_path}")
    with open(configs.input_json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    if not isinstance(json_data, list) or len(json_data) == 0:
        raise ValueError(
            f"Input JSON must be a non-empty top-level list, got "
            f"{type(json_data).__name__} from {configs.input_json_path}"
        )

    seed_in_json = json_data[0].get("modelSeeds")
    if seed_in_json and configs.use_seeds_in_json:
        seeds = [int(i) for i in seed_in_json]
        logger.info(f"Using seeds from JSON: {seeds}")
    else:
        seeds = configs.seeds

    try:
        dataloader = get_inference_dataloader(configs=configs)
    except Exception as exc:
        error_message = f"Dataloader initialization failed: {exc}\n{traceback.format_exc()}"
        logger.error(error_message)
        with open(opjoin(runner.error_dir, "error.txt"), "a", encoding="utf-8") as f:
            f.write(error_message)
        return

    num_data = len(dataloader.dataset)
    t0_start = time.time()
    for seed in seeds:
        seed_everything(seed=seed, deterministic=configs.deterministic)
        t1_start = time.time()
        for batch in dataloader:
            sample_name = "unknown"
            try:
                t2_start = time.time()
                data, atom_array, data_error_message = batch[0]
                sample_name = data["sample_name"]

                if data_error_message:
                    logger.error(f"Data error for {sample_name}: {data_error_message}")
                    with open(
                        opjoin(runner.error_dir, f"{sample_name}.txt"),
                        "a",
                        encoding="utf-8",
                    ) as f:
                        f.write(data_error_message)
                    continue

                logger.info(
                    f"[Rank {DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{num_data})] "
                    f"{sample_name} [seed:{seed}]: "
                    f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                    f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
                )
                new_configs = update_inference_configs(configs, data["N_token"].item())
                runner.update_model_configs(new_configs)
                prediction = runner.predict(data)
                runner.dumper.dump(
                    dataset_name="",
                    pdb_id=sample_name,
                    seed=seed,
                    pred_dict=prediction,
                    atom_array=atom_array,
                    entity_poly_type={
                        k: v
                        for k, v in data["entity_poly_type"].items()
                        if v != "non-polymer"
                    },
                )
                t2_end = time.time()
                logger.info(
                    f"[Rank {DIST_WRAPPER.rank}] {sample_name} [seed:{seed}] succeeded. "
                    f"Model forward time: {t2_end - t2_start:.2f}s. "
                    f"Results saved to {configs.dump_dir}"
                )
                torch.cuda.empty_cache()
            except Exception as exc:
                error_message = (
                    f"[Rank {DIST_WRAPPER.rank}] {sample_name} failed: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                logger.error(error_message)
                with open(
                    opjoin(runner.error_dir, f"{sample_name}.txt"),
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(error_message)
                torch.cuda.empty_cache()
        t1_end = time.time()
        logger.info(f"[Rank {DIST_WRAPPER.rank}] Seed {seed} completed in {t1_end - t1_start:.2f}s.")

    if opexists(runner.error_dir):
        try:
            if not os.listdir(runner.error_dir):
                os.rmdir(runner.error_dir)
        except Exception:
            pass

    t0_end = time.time()
    logger.info(f"[Rank {DIST_WRAPPER.rank}] Job completed in {t0_end - t0_start:.2f}s.")


def update_gpu_compatible_configs(configs: Any) -> Any:
    if not torch.cuda.is_available():
        return configs
    major, minor = torch.cuda.get_device_capability()
    cc = major + minor / 10.0
    if 7.0 <= cc < 8.0:
        configs.dtype = "fp32"
        configs.triangle_attention = "torch"
        configs.triangle_multiplicative = "torch"
        logger.info("Using FP32 and torch kernels for compute capability 7.x GPU.")
    return configs


def build_configs(arg_str: str) -> Any:
    base_configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    first_pass = parse_configs(
        configs=base_configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )

    user_config = {}
    if first_pass.config_path:
        user_config = load_config(first_pass.config_path) or {}

    model_name = user_config.get("model_name", first_pass.model_name)
    if cli_has_key(arg_str, "model_name"):
        model_name = first_pass.model_name
    if model_name not in model_configs:
        raise KeyError(f"Unknown model_name={model_name}. Available: {sorted(model_configs)}")

    merged = {**configs_base, **{"data": data_configs}, **inference_configs}
    deep_update(merged, model_configs[model_name])
    deep_update(merged, user_config)
    configs = parse_configs(
        configs=merged,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    return update_gpu_compatible_configs(configs)


def run() -> None:
    log_format = (
        "%(asctime)s,%(msecs)-3d %(levelname)-8s "
        "[%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    )
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )

    configs = build_configs(parse_sys_args())
    logger.info(
        f"Inference by VenusFold: model_name={configs.model_name}, "
        f"checkpoint={resolve_checkpoint_path(configs)}, dtype={configs.dtype}, "
        f"cycle={configs.model.N_cycle}, diffusion_steps={configs.sample_diffusion.N_step}, "
        f"samples={configs.sample_diffusion.N_sample}"
    )
    logger.info(
        f"Triangle kernels: multiplicative={configs.triangle_multiplicative}, "
        f"attention={configs.triangle_attention}; "
        f"shared_vars_cache={configs.enable_diffusion_shared_vars_cache}, "
        f"efficient_fusion={configs.enable_efficient_fusion}, tf32={configs.enable_tf32}"
    )
    if configs.dry_run:
        logger.info("Dry run completed; model and checkpoint were not loaded.")
        return
    prepare_missing_msas(configs)
    runner = InferenceRunner(configs)
    infer_predict(runner, configs)


if __name__ == "__main__":
    run()
