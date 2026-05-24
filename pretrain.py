from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from glob import glob
import math
import os
import json
import yaml
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from torch import Tensor, nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig, OmegaConf

from models.layers import Carry
from models.common import wrap_tensor
from models.transformer import TransformerBlock
from models.adam_atan2 import AdamATan2
from utils.functions import load_model_class, get_model_source_path
from dataset_new import V1Dataset, V1DatasetConfig, V1DatasetMeta


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    head: str


class DataConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    path: str
    target_only: bool = True  # Only supervise Answer.


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    data: DataConfig

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float
    ema: Optional[float] = None
    fwd_bwd_dtype: str = "bfloat16"

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Resume / fine-tune from checkpoint
    resume_from: Optional[str] = None
    resume_epoch: Optional[int] = None
    resume_step: Optional[int] = None
    weights_only_resume_from_ema: bool = False  # Swap EMA into model + reset optim
    resume_step_offset: int = 0  # Global step offset for staged pretraining LR continuity.
    total_steps_override: Optional[int] = None  # Planned total global steps across stages.
    skip_batches: int = 0  # Skip dataset batches when resuming inside an epoch.

    # Extras
    seed: int = 0
    checkpoint_interval: int = 1
    checkpoint_step_interval: Optional[int] = None
    checkpoint_keep_last: Optional[int] = 2
    log_interval: int = 5


@dataclass
class TrainState:
    model: nn.Module
    carry: Optional[Carry]
    
    optim: AdamATan2

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, local_batch_size: int, drop_last_batch: bool, rank: int, world_size: int):
    dataset = V1Dataset(V1DatasetConfig(
        seed=config.seed,

        dataset_path=config.data.path,
        drop_last_batch=drop_last_batch,

        target_only=config.data.target_only,

        batch_max_length=local_batch_size,
        skip_batches=config.skip_batches,
        rank=rank,
        num_replicas=world_size,
    ))
    dataloader = DataLoader(
        dataset,
        batch_size=None,

        num_workers=1,
        prefetch_factor=8,

        pin_memory=True,
        persistent_workers=True  # NOTE: Required for correct epoch handling
    )
    return dataloader, dataset.metadata


def apply_fsdp(module: nn.Module, param_dtype: torch.dtype):
    fully_shard(module,
                mp_policy=MixedPrecisionPolicy(param_dtype=param_dtype,
                                               reduce_dtype=torch.get_default_dtype()),  # Use master dtype for reduction
                reshard_after_forward=False)  # Trade off VRAM for less comms
    
    assert isinstance(module, FSDPModule)
    # Disable gradient division. Adams is scale invariant.
    module.set_gradient_divide_factor(1.0)
    module.set_force_sum_reduction_for_comms(True)


def create_model_and_carry(config: PretrainConfig, train_metadata: V1DatasetMeta, local_batch_size: int):
    model_cfg = config.arch.model_dump() | train_metadata.model_dump() | config.data.model_dump()
    fwd_bwd_dtype = getattr(torch, config.fwd_bwd_dtype)

    # Instantiate model with head
    model_cls = load_model_class(config.arch.name)
    head_cls = load_model_class(config.arch.head)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        carry = model.initial_carry(local_batch_size, dtype=fwd_bwd_dtype)  # pyright: ignore[reportCallIssue]
        # Attach loss head
        model = head_cls(model, model_cfg)

    # ----FSDP----
    # Broadcast buffers
    for buffer in model.buffers():
        dist.broadcast(buffer, src=0)

    # Detect TransformerBlock recursively and apply FSDP
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            apply_fsdp(module, fwd_bwd_dtype)

    apply_fsdp(model, fwd_bwd_dtype)

    # ----Create optimizer----
    optim = AdamATan2(model.parameters(),
                      lr=torch.tensor(0.0, dtype=torch.get_default_dtype(), device="cpu"),
                      betas=(config.beta1, config.beta2),
                      weight_decay=config.weight_decay,
                      ema=config.ema)

    return model, carry, optim


def init_train(config: PretrainConfig, rank: int, world_size: int):
    assert config.global_batch_size % world_size == 0, f"Global batch size {config.global_batch_size} must be divisible by world size {world_size}."
    local_batch_size = config.global_batch_size // world_size

    # Dataset
    train_loader, train_metadata = create_dataloader(config, local_batch_size, drop_last_batch=True,  rank=rank, world_size=world_size)

    # Model
    model, carry, optim = create_model_and_carry(config, train_metadata, local_batch_size)

    # Train state
    # Estimated total training steps
    stage_steps = int(config.epochs * train_metadata.total_length // config.global_batch_size)
    if config.total_steps_override is not None:
        total_steps = config.total_steps_override
    else:
        total_steps = config.resume_step_offset + stage_steps
    train_state = TrainState(
        model=model,
        carry=carry,
        optim=optim,
        
        step=config.resume_step_offset,
        total_steps=total_steps
    )
    return train_state, train_loader, train_metadata


def update_lr(config: PretrainConfig, train_state: TrainState) -> float:
    # Linear warmup cosine schedule
    if train_state.step < config.lr_warmup_steps:
        lr = config.lr * min(1.0, train_state.step / config.lr_warmup_steps)
    else:
        progress = (train_state.step - config.lr_warmup_steps) / (train_state.total_steps - config.lr_warmup_steps)
        lr = config.lr * (config.lr_min_ratio + max(0.0, (1 - config.lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))

    tensor_lr = torch.tensor(lr, dtype=torch.get_default_dtype(), device="cpu")
    for param_group in train_state.optim.param_groups:
        param_group["lr"] = tensor_lr

    return lr


@torch.compile(dynamic=False)
def train_batch(train_state: TrainState, batch: dict[str, Tensor], **kwargs):
    train_state.carry, loss, metrics = train_state.model(batch=batch, carry=train_state.carry, **kwargs)
    loss.backward()
    train_state.optim.step()
    train_state.optim.zero_grad()
    return metrics


@torch.inference_mode()
def reduce_metrics(local_metrics: dict[str, Tensor], prefix: str):
    metric_keys = list(sorted(local_metrics.keys()))  # Sort keys to guarantee all processes use the same order.
    # Reduce and reconstruct
    metric_values = torch.stack([local_metrics[k][0] for k in metric_keys] + [local_metrics[k][1] for k in metric_keys])
    dist.reduce(metric_values, dst=0)
    # Split and normalize
    metrics, metrics_div = metric_values.chunk(2, dim=-1)
    metrics = (metrics / metrics_div).cpu().numpy().tolist()
    return {prefix + name: metrics[idx] for idx, name in enumerate(metric_keys)}


def load_checkpoint(config: PretrainConfig, train_state: TrainState):
    """Resume from a saved checkpoint.

    Loads both model weights and optimizer state (which carries EMA in
    AdamATan2). When weights_only_resume_from_ema=True, swaps the EMA buffer
    into the model and resets the optimizer state — typical for fine-tuning
    off a pretrain run with EMA-smoothed weights.
    """
    if config.resume_from is None:
        return

    if config.resume_step is not None:
        checkpoint_id = os.path.join(config.resume_from, f"fsdp2_step_{config.resume_step}")
    else:
        epoch = config.resume_epoch
        if epoch is None:
            ckpt_files = glob(os.path.join(config.resume_from, "fsdp2_epoch_*"))
            if not ckpt_files:
                raise FileNotFoundError(f"No checkpoint found in {config.resume_from}")
            epoch = max(int(Path(f).stem.split("_")[-1]) for f in ckpt_files)

        checkpoint_id = os.path.join(config.resume_from, f"fsdp2_epoch_{epoch}")

    print(f"[Resume] Loading model + optimizer from {checkpoint_id}")
    optim_state = get_optimizer_state_dict(train_state.model, train_state.optim)
    dcp.load(
        {"model": train_state.model.state_dict(), "optim": optim_state},
        checkpoint_id=checkpoint_id,
    )
    set_optimizer_state_dict(train_state.model, train_state.optim, optim_state)

    # set_optimizer_state_dict silently overwrites param_groups with the pretrain hyperparams
    # (lr, betas, weight_decay, ema). Restore the SFT cfg values so that overrides take effect.
    # (lr is also restored every step by update_lr() — these three are not.)
    for param_group in train_state.optim.param_groups:
        param_group["betas"] = (config.beta1, config.beta2)
        param_group["weight_decay"] = config.weight_decay
        param_group["ema"] = config.ema

    if config.weights_only_resume_from_ema:
        print("[Resume] Swapping EMA into model and resetting optimizer state")
        train_state.optim.swap_ema()
        train_state.optim._init_state()

    print(f"[Resume] Done.")


def save_training_checkpoint(config: PretrainConfig, train_state: TrainState, rank: int, tag: str):
    if config.checkpoint_path is None:
        return
    checkpoint_id = os.path.join(config.checkpoint_path, f"fsdp2_{tag}")
    dcp.save({"model": train_state.model.state_dict(), "optim": get_optimizer_state_dict(train_state.model, train_state.optim)},  # pyright: ignore[reportPrivateImportUsage]
             checkpoint_id=checkpoint_id)
    torch.save(train_state.carry, os.path.join(config.checkpoint_path, f"carry_{tag}.{rank}.pt"))
    if rank == 0:
        step_info = {
            "tag": tag,
            "global_step": train_state.step,
            "stage_start_step": config.resume_step_offset,
            "skip_batches_hint": max(0, train_state.step - config.resume_step_offset),
            "data_path": config.data.path,
            "global_batch_size": config.global_batch_size,
        }
        with open(os.path.join(config.checkpoint_path, f"{tag}_info.json"), "wt") as f:
            json.dump(step_info, f, indent=2)
        with open(os.path.join(config.checkpoint_path, "latest_checkpoint.txt"), "wt") as f:
            f.write(f"{tag}\n")

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        prune_old_checkpoints(config, keep_last=config.checkpoint_keep_last)
    if dist.is_initialized():
        dist.barrier()


def checkpoint_tag_order(root: Path, tag: str) -> tuple[int, int]:
    info_path = root / f"{tag}_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            return int(info.get("global_step", 0)), 1 if tag.startswith("epoch_") else 0
        except Exception:
            pass

    try:
        return int(tag.rsplit("_", 1)[-1]), 1 if tag.startswith("epoch_") else 0
    except ValueError:
        return 0, 0


def prune_old_checkpoints(config: PretrainConfig, keep_last: Optional[int]):
    if keep_last is None or keep_last <= 0 or config.checkpoint_path is None:
        return

    root = Path(config.checkpoint_path)
    if not root.exists():
        return

    tags = []
    for path in root.glob("fsdp2_*"):
        if path.is_dir():
            tags.append(path.name[len("fsdp2_"):])
    tags = sorted(set(tags), key=lambda tag: checkpoint_tag_order(root, tag))
    remove = tags[:-keep_last]
    if not remove:
        return

    for tag in remove:
        shutil.rmtree(root / f"fsdp2_{tag}", ignore_errors=True)
        for carry in root.glob(f"carry_{tag}.*.pt"):
            carry.unlink(missing_ok=True)
        (root / f"{tag}_info.json").unlink(missing_ok=True)
    print(f"[Checkpoint] Pruned {len(remove)} old checkpoints; kept latest {keep_last}: {tags[-keep_last:]}", flush=True)


def save_code_and_config(config: PretrainConfig, train_metadata: V1DatasetMeta):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    with open(os.path.join(config.checkpoint_path, "all_config.yaml"), "wt") as f:
        yaml.dump(config.model_dump(), f)
    with open(os.path.join(config.checkpoint_path, "train_metadata.yaml"), "wt") as f:
        yaml.dump(train_metadata.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**OmegaConf.to_container(hydra_config, resolve=True))  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{Path(config.data.path).stem.capitalize()} HLM-torch"
        if config.run_name is None:
            config.run_name = os.environ.get("MLP_TASK_NAME", f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}")  # pyright: ignore[reportPrivateImportUsage]
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    dist.broadcast_object_list(objects, src=0)
    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    WORLD_SIZE = 1
    RANK = 0
    DEVICE_ID = 0

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        WORLD_SIZE = dist.get_world_size()
        RANK = dist.get_rank()
        DEVICE_ID = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(DEVICE_ID)

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # --- Training
    train_state, train_loader, train_metadata = init_train(config, rank=RANK, world_size=WORLD_SIZE)
    load_checkpoint(config, train_state)

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps, initial=train_state.step)

        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump() | {"train_metadata": train_metadata.model_dump()},
                   settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config, train_metadata)

    # Training Loop
    for epoch in range(1, config.epochs + 1):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {epoch}")

        # ############ Train Iter
        train_state.model.train()
        for batch, batch_info in train_loader:
            train_state.step += 1            
            lr = update_lr(config, train_state)
            # Extra train arguments (such as BP warmup etc.)
            train_extra_args = train_state.model.compute_train_extra_args(train_state)  # pyright: ignore[reportCallIssue]
            
            metrics = train_batch(train_state, batch | {k: wrap_tensor(torch.tensor(v, device="cpu")) for k, v in batch_info.items()}, **train_extra_args)

            if train_state.step % config.log_interval == 0:
                metrics = reduce_metrics(metrics, prefix="train/")
                if RANK == 0:
                    progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
                    wandb.log(metrics | train_extra_args | {"train/lr": lr}, step=train_state.step)

            del metrics

            if config.checkpoint_step_interval is not None and train_state.step % config.checkpoint_step_interval == 0:
                save_training_checkpoint(config, train_state, RANK, f"step_{train_state.step}")

        ############ EVAL STACK: TBD TODO

        ############ Checkpointing
        if (epoch % config.checkpoint_interval == 0) or (epoch == config.epochs):
            save_training_checkpoint(config, train_state, RANK, f"epoch_{epoch}")

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
