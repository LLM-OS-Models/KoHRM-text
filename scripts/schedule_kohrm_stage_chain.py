"""Schedule KoHRM-Text staged training after the current stage-0 run.

This watcher keeps GPUs busy without racing the tokenizer:
- wait for a stable stage-0 checkpoint;
- snapshot currently completed HRM fast-cap tokenized tasks via symlinks;
- sample them into a V1Dataset;
- resume training as stage-1 with a larger batch, with fallback on OOM/failure;
- after stage-1, wait for tokenizer completion and train remaining tasks as stage-2.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


REQUIRED_TASK_FILES = {
    "tokens.npy",
    "inst_start.npy",
    "inst_len.npy",
    "resp_start.npy",
    "resp_len.npy",
    "metadata.json",
}


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def wait_stable(path: Path, seconds: int) -> None:
    last = -1
    stable_since = time.monotonic()
    while True:
        current = dir_size(path)
        if current == last:
            if time.monotonic() - stable_since >= seconds:
                return
        else:
            last = current
            stable_since = time.monotonic()
        time.sleep(min(30, max(5, seconds // 3)))


def wait_checkpoint(root: Path, epoch: int, stable_seconds: int, poll_seconds: int) -> Path:
    ckpt = root / f"fsdp2_epoch_{epoch}"
    while True:
        carries = sorted(root.glob(f"carry_epoch_{epoch}.*.pt"))
        if ckpt.exists() and len(carries) >= 8:
            log(f"checkpoint detected: {ckpt}; waiting {stable_seconds}s for stable files")
            wait_stable(root, stable_seconds)
            return ckpt
        log(f"waiting for checkpoint {ckpt}; carry files={len(carries)}/8")
        time.sleep(poll_seconds)


def completed_tasks(tokenized_root: Path) -> list[Path]:
    tasks: list[Path] = []
    for path in sorted(tokenized_root.iterdir()):
        if not path.is_dir():
            continue
        names = {item.name for item in path.iterdir() if item.is_file()}
        if REQUIRED_TASK_FILES.issubset(names):
            tasks.append(path)
    return tasks


def build_snapshot(tokenized_root: Path, snapshot_root: Path, exclude_names: set[str] | None = None) -> list[str]:
    exclude_names = exclude_names or set()
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    snapshot_root.mkdir(parents=True, exist_ok=True)

    tok_info = tokenized_root / "tokenizer_info.json"
    if not tok_info.exists():
        raise FileNotFoundError(tok_info)
    shutil.copy2(tok_info, snapshot_root / "tokenizer_info.json")

    names: list[str] = []
    for task in completed_tasks(tokenized_root):
        if task.name in exclude_names:
            continue
        os.symlink(task, snapshot_root / task.name, target_is_directory=True)
        names.append(task.name)

    (snapshot_root / "snapshot_manifest.json").write_text(
        json.dumps(
            {
                "source": str(tokenized_root),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tasks": names,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"snapshot built: {snapshot_root}; tasks={len(names)}")
    return names


def read_total_tokens(dataset_path: Path) -> int:
    meta = json.loads((dataset_path / "metadata.json").read_text(encoding="utf-8"))
    return int(meta["total_length"])


def run_checked(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> int:
    log(f"running: {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as f:
        f.write((f"\n\n===== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} =====\n").encode())
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
        f.write((f"\n===== exit {proc.returncode} =====\n").encode())
        return proc.returncode


def sample_dataset(data_io_root: Path, tokenized_snapshot: Path, output_path: Path, epochs: int, log_path: Path) -> None:
    if output_path.exists():
        shutil.rmtree(output_path)
    cmd = [
        "python",
        str(data_io_root / "sample_tokenized.py"),
        f"tokenized_path={tokenized_snapshot}",
        f"output_path={output_path}",
        f"prefix_config_path={data_io_root / 'prefix_config.yaml'}",
        f"epochs={epochs}",
        "context_size=4097",
    ]
    rc = run_checked(cmd, data_io_root, os.environ.copy(), log_path)
    if rc != 0:
        raise RuntimeError(f"sample_tokenized failed with exit {rc}; see {log_path}")


def start_upload_watcher(hrm_root: Path, checkpoint_root: Path, repo_id: str, env_file: Path, log_path: Path) -> subprocess.Popen:
    cmd = [
        "python",
        "scripts/watch_and_upload_hrm_checkpoints.py",
        "--checkpoint-root",
        str(checkpoint_root),
        "--repo-id",
        repo_id,
        "--stage-root",
        "/home/work/.data/hrm_text_hf_upload_stage",
        "--env-file",
        str(env_file),
        "--poll-seconds",
        "300",
        "--stable-seconds",
        "120",
        "--num-workers",
        "4",
    ]
    log(f"starting upload watcher: {' '.join(cmd)}")
    f = log_path.open("ab")
    return subprocess.Popen(cmd, cwd=hrm_root, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)


def train_stage(
    hrm_root: Path,
    dataset_path: Path,
    resume_from: Path,
    checkpoint_base: Path,
    run_base: str,
    resume_step_offset: int,
    total_steps_override: int,
    batches: list[int],
    env: dict[str, str],
) -> tuple[Path, int, int]:
    tokens = read_total_tokens(dataset_path)
    for batch in batches:
        stage_steps = tokens // batch
        ckpt = Path(f"{checkpoint_base}-gbs{batch}")
        log_path = Path(f"/home/work/.data/hrm_text_logs/{run_base}-gbs{batch}.log")
        cmd = [
            "taskset",
            "-c",
            "0-31",
            "torchrun",
            "--standalone",
            "--nproc_per_node=8",
            "pretrain.py",
            "arch/size@arch=XL",
            f"data.path={dataset_path}",
            f"resume_from={resume_from}",
            f"+checkpoint_path={ckpt}",
            "+project_name=KoHRM-Text",
            f"+run_name={run_base}-gbs{batch}",
            "epochs=1",
            f"global_batch_size={batch}",
            "lr_warmup_steps=2000",
            f"resume_step_offset={resume_step_offset}",
            f"total_steps_override={total_steps_override}",
            "+log_interval=5",
            "checkpoint_interval=1",
        ]
        rc = run_checked(cmd, hrm_root, env, log_path)
        if rc == 0:
            return ckpt, stage_steps, batch
        log(f"stage train failed at batch={batch}; trying fallback if available")
    raise RuntimeError(f"all batch attempts failed for {run_base}")


def tokenizer_running(match: str) -> bool:
    proc = subprocess.run(["pgrep", "-af", match], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return bool(proc.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hrm-root", default="/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text")
    ap.add_argument("--data-io-root", default="/home/work/.projects/LLM-OS-Models/Terminal/data_io")
    ap.add_argument("--tokenized-root", default="/home/work/.data/hrm_text_tokenized/koterm_hrm_cleaned_fastcap_v1")
    ap.add_argument("--stage0-ckpt", default="/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage0-available-mix-gbs172")
    ap.add_argument("--repo-id", default="LLM-OS-Models/KoHRM-Text-1.4B")
    ap.add_argument("--env-file", default="/home/work/.projects/LLM-OS-Models/Terminal/.env")
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--stable-seconds", type=int, default=180)
    ap.add_argument("--total-steps-override", type=int, default=290643)
    args = ap.parse_args()

    hrm_root = Path(args.hrm_root)
    data_io_root = Path(args.data_io_root)
    tokenized_root = Path(args.tokenized_root)
    stage0_ckpt = Path(args.stage0_ckpt)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "HYDRA_FULL_ERROR": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "WANDB_MODE": "offline",
            "WANDB_DIR": "/home/work/.data/wandb",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NCCL_DEBUG": "WARN",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        }
    )

    log("KoHRM staged scheduler started")
    wait_checkpoint(stage0_ckpt, epoch=1, stable_seconds=args.stable_seconds, poll_seconds=args.poll_seconds)

    stage1_snapshot = Path("/home/work/.data/hrm_text_tokenized/koterm_hrm_cleaned_fastcap_stage1_snapshot")
    stage1_names = build_snapshot(tokenized_root, stage1_snapshot)
    if not stage1_names:
        raise RuntimeError("no completed tokenized tasks for stage-1")

    stage1_data = Path("/home/work/.data/hrm_text_prepared/koterm_hrm_cleaned_fastcap_stage1_v1")
    sample_dataset(
        data_io_root,
        stage1_snapshot,
        stage1_data,
        epochs=1,
        log_path=Path("/home/work/.data/hrm_text_logs/koterm_hrm_cleaned_fastcap_stage1_sample.log"),
    )

    stage1_ckpt, stage1_steps, stage1_batch = train_stage(
        hrm_root=hrm_root,
        dataset_path=stage1_data,
        resume_from=stage0_ckpt,
        checkpoint_base=Path("/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage1-hrm-fastcap"),
        run_base="KoHRM-Text-1.4B-stage1-hrm-fastcap",
        resume_step_offset=4134,
        total_steps_override=args.total_steps_override,
        batches=[196608, 180224, 172032],
        env=env,
    )
    start_upload_watcher(
        hrm_root,
        stage1_ckpt,
        args.repo_id,
        Path(args.env_file),
        Path("/home/work/.data/hrm_text_logs/KoHRM-Text-1.4B-stage1-hf-upload-watcher.log"),
    )

    log(f"stage-1 complete: ckpt={stage1_ckpt}; steps={stage1_steps}; batch={stage1_batch}")
    log("waiting for tokenizer completion before stage-2 remainder")
    while tokenizer_running("target/release/tokenizer .*koterm_hrm_cleaned_fastcap_v1"):
        time.sleep(args.poll_seconds)

    stage2_snapshot = Path("/home/work/.data/hrm_text_tokenized/koterm_hrm_cleaned_fastcap_stage2_remainder_snapshot")
    stage2_names = build_snapshot(tokenized_root, stage2_snapshot, exclude_names=set(stage1_names))
    if not stage2_names:
        log("no new completed tasks for stage-2; scheduler done")
        return

    stage2_data = Path("/home/work/.data/hrm_text_prepared/koterm_hrm_cleaned_fastcap_stage2_remainder_v1")
    sample_dataset(
        data_io_root,
        stage2_snapshot,
        stage2_data,
        epochs=1,
        log_path=Path("/home/work/.data/hrm_text_logs/koterm_hrm_cleaned_fastcap_stage2_sample.log"),
    )

    stage2_ckpt, stage2_steps, stage2_batch = train_stage(
        hrm_root=hrm_root,
        dataset_path=stage2_data,
        resume_from=stage1_ckpt,
        checkpoint_base=Path("/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage2-hrm-fastcap-remainder"),
        run_base="KoHRM-Text-1.4B-stage2-hrm-fastcap-remainder",
        resume_step_offset=4134 + stage1_steps,
        total_steps_override=args.total_steps_override,
        batches=[196608, 180224, 172032],
        env=env,
    )
    start_upload_watcher(
        hrm_root,
        stage2_ckpt,
        args.repo_id,
        Path(args.env_file),
        Path("/home/work/.data/hrm_text_logs/KoHRM-Text-1.4B-stage2-hf-upload-watcher.log"),
    )
    log(f"stage-2 complete: ckpt={stage2_ckpt}; steps={stage2_steps}; batch={stage2_batch}")


if __name__ == "__main__":
    main()
