from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import math
import os

import hydra
import pydantic
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

from dataset_new import V1Dataset, V1DatasetConfig, V1DatasetMeta
from models.adam_atan2 import AdamATan2
from models.common import wrap_tensor
from models.transformer import TransformerBlock
from models.lora import inject_lora, mark_only_lora_trainable, save_lora_adapter, trainable_lora_parameters
from pretrain import ArchConfig, DataConfig, apply_fsdp, load_model_class, reduce_metrics, update_lr


class LoraConfig(pydantic.BaseModel):
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_suffixes: list[str] = ["gqkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head"]


class LoraTrainConfig(pydantic.BaseModel):
    arch: ArchConfig
    data: DataConfig

    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    ema: Optional[float] = None
    fwd_bwd_dtype: str = "bfloat16"

    project_name: Optional[str] = "KoHRM-Text-LoRA"
    run_name: Optional[str] = None
    checkpoint_path: str

    resume_from: str
    resume_epoch: Optional[int] = None
    resume_step: Optional[int] = None
    weights_only_resume_from_ema: bool = True
    resume_step_offset: int = 0
    total_steps_override: Optional[int] = None
    skip_batches: int = 0

    seed: int = 0
    checkpoint_interval: int = 1
    checkpoint_step_interval: Optional[int] = None
    checkpoint_keep_last: Optional[int] = 2
    log_interval: int = 5

    lora: LoraConfig = LoraConfig()


@dataclass
class LoraTrainState:
    model: nn.Module
    carry: object
    optim: AdamATan2
    step: int
    total_steps: int


def create_dataloader(config: LoraTrainConfig, local_batch_size: int, rank: int, world_size: int):
    dataset = V1Dataset(V1DatasetConfig(
        seed=config.seed,
        dataset_path=config.data.path,
        drop_last_batch=True,
        target_only=config.data.target_only,
        batch_max_length=local_batch_size,
        skip_batches=config.skip_batches,
        rank=rank,
        num_replicas=world_size,
    ))
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
    )
    return loader, dataset.metadata


def checkpoint_id_from_config(config: LoraTrainConfig) -> str:
    if config.resume_step is not None:
        return os.path.join(config.resume_from, f"fsdp2_step_{config.resume_step}")
    epoch = config.resume_epoch
    if epoch is None:
        ckpts = sorted(Path(config.resume_from).glob("fsdp2_epoch_*"))
        if not ckpts:
            raise FileNotFoundError(f"No epoch checkpoint found in {config.resume_from}")
        epoch = max(int(p.name.rsplit("_", 1)[-1]) for p in ckpts)
    return os.path.join(config.resume_from, f"fsdp2_epoch_{epoch}")


def create_raw_model_and_carry(config: LoraTrainConfig, train_metadata: V1DatasetMeta, local_batch_size: int):
    model_cfg = config.arch.model_dump() | train_metadata.model_dump() | config.data.model_dump()
    fwd_bwd_dtype = getattr(torch, config.fwd_bwd_dtype)
    model_cls = load_model_class(config.arch.name)
    head_cls = load_model_class(config.arch.head)
    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        carry = model.initial_carry(local_batch_size, dtype=fwd_bwd_dtype)  # pyright: ignore[reportCallIssue]
        model = head_cls(model, model_cfg)
    return model, carry


def load_base_weights(config: LoraTrainConfig, model: nn.Module) -> None:
    checkpoint_id = checkpoint_id_from_config(config)
    print(f"[LoRA] Loading base model + EMA source from {checkpoint_id}", flush=True)
    dummy_optim = AdamATan2(
        model.parameters(),
        lr=torch.tensor(0.0, dtype=torch.get_default_dtype(), device="cpu"),
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        ema=0.9999,
    )
    optim_state = get_optimizer_state_dict(model, dummy_optim)
    dcp.load(
        {"model": model.state_dict(), "optim": optim_state},
        checkpoint_id=checkpoint_id,
        no_dist=True,
    )
    if config.weights_only_resume_from_ema:
        dummy_optim.swap_ema()
    del optim_state
    del dummy_optim
    torch.cuda.empty_cache()
    print("[LoRA] Base weights loaded.", flush=True)


def apply_fsdp_to_lora_model(model: nn.Module, param_dtype: torch.dtype):
    if dist.is_initialized():
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            apply_fsdp(module, param_dtype)
    apply_fsdp(model, param_dtype)


def init_lora_train(config: LoraTrainConfig, rank: int, world_size: int):
    assert config.global_batch_size % world_size == 0
    local_batch_size = config.global_batch_size // world_size
    train_loader, train_metadata = create_dataloader(config, local_batch_size, rank=rank, world_size=world_size)

    model, carry = create_raw_model_and_carry(config, train_metadata, local_batch_size)
    load_base_weights(config, model)

    matched_modules = inject_lora(
        model,
        target_suffixes=config.lora.target_suffixes,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
    )
    mark_only_lora_trainable(model)
    apply_fsdp_to_lora_model(model, getattr(torch, config.fwd_bwd_dtype))
    lora_params = trainable_lora_parameters(model)
    if not lora_params:
        raise ValueError("No trainable LoRA parameters found")

    optim = AdamATan2(
        lora_params,
        lr=torch.tensor(0.0, dtype=torch.get_default_dtype(), device="cpu"),
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        ema=config.ema,
    )

    stage_steps = int(config.epochs * train_metadata.total_length // config.global_batch_size)
    total_steps = config.total_steps_override if config.total_steps_override is not None else config.resume_step_offset + stage_steps
    state = LoraTrainState(model=model, carry=carry, optim=optim, step=config.resume_step_offset, total_steps=total_steps)
    return state, train_loader, train_metadata, matched_modules


def train_batch_lora(train_state: LoraTrainState, batch: dict[str, Tensor], **kwargs):
    train_state.carry, loss, metrics = train_state.model(batch=batch, carry=train_state.carry, **kwargs)
    loss.backward()
    train_state.optim.step()
    train_state.optim.zero_grad()
    return metrics


def save_lora_training_checkpoint(config: LoraTrainConfig, train_state: LoraTrainState, matched_modules: list[str], tag: str, rank: int):
    if rank != 0:
        return
    payload = config.model_dump()
    save_lora_adapter(
        train_state.model,
        output_dir=config.checkpoint_path,
        config=payload,
        matched_modules=matched_modules,
        step=train_state.step,
        tag=tag,
    )


def load_synced_config(hydra_config: DictConfig, rank: int) -> LoraTrainConfig:
    objects = [None]
    if rank == 0:
        config = LoraTrainConfig(**OmegaConf.to_container(hydra_config, resolve=True))  # type: ignore[arg-type]
        if config.run_name is None:
            config.run_name = Path(config.data.path).name
        objects = [config]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]  # type: ignore[return-value]


@hydra.main(config_path="config", config_name="cfg_lora", version_base=None)
def launch(hydra_config: DictConfig):
    world_size = 1
    rank = 0
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    config = load_synced_config(hydra_config, rank)
    torch.random.manual_seed(config.seed + rank)

    state, train_loader, train_metadata, matched_modules = init_lora_train(config, rank=rank, world_size=world_size)
    progress_bar = None
    if rank == 0:
        Path(config.checkpoint_path).mkdir(parents=True, exist_ok=True)
        progress_bar = tqdm.tqdm(total=state.total_steps, initial=state.step)
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump() | {"train_metadata": train_metadata.model_dump(), "matched_lora_modules": matched_modules},
            settings=wandb.Settings(_disable_stats=True),
        )
        trainable = sum(p.numel() for p in state.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in state.model.parameters())
        wandb.log({"num_params": total, "num_trainable_lora_params": trainable}, step=0)
        with open(Path(config.checkpoint_path) / "lora_train_config.json", "w", encoding="utf-8") as f:
            json.dump(config.model_dump() | {"matched_modules": matched_modules, "train_metadata": train_metadata.model_dump()}, f, indent=2)

    for epoch in range(1, config.epochs + 1):
        print(f"[Rank {rank}, World Size {world_size}]: LoRA Epoch {epoch}", flush=True)
        state.model.train()
        for batch, batch_info in train_loader:
            state.step += 1
            lr = update_lr(config, state)  # type: ignore[arg-type]
            train_extra_args = state.model.compute_train_extra_args(state)  # type: ignore[operator]
            metrics = train_batch_lora(
                state,
                batch | {k: wrap_tensor(torch.tensor(v, device="cpu")) for k, v in batch_info.items()},
                **train_extra_args,
            )

            if state.step % config.log_interval == 0:
                metrics = reduce_metrics(metrics, prefix="train/")
                if rank == 0:
                    progress_bar.update(state.step - progress_bar.n)  # type: ignore[union-attr]
                    wandb.log(metrics | train_extra_args | {"train/lr": lr}, step=state.step)

            if config.checkpoint_step_interval is not None and state.step % config.checkpoint_step_interval == 0:
                save_lora_training_checkpoint(config, state, matched_modules, f"step_{state.step}", rank)

        if (epoch % config.checkpoint_interval == 0) or (epoch == config.epochs):
            save_lora_training_checkpoint(config, state, matched_modules, f"epoch_{epoch}", rank)

    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
