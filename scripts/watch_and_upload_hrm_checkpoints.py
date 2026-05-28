"""Watch HRM-Text checkpoints and upload epoch artifacts to Hugging Face.

This intentionally runs outside the training process so network failures do not
kill training. It uploads at epoch granularity, not every step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import yaml
from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[2]
EPOCH_RE = re.compile(r"fsdp2_epoch_(\d+)$")


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def get_token(env_file: Path | None) -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token
    candidates = []
    if env_file is not None:
        candidates.append(env_file)
    candidates.extend([ROOT / ".env", Path.cwd() / ".env", Path.home() / ".env"])
    for path in candidates:
        values = parse_env_file(path)
        token = values.get("HF_TOKEN") or values.get("HUGGINGFACE_TOKEN")
        if token:
            return token
    raise RuntimeError("HF_TOKEN is not available in environment or .env")


def dir_size(path: Path) -> int:
    total = 0
    if path.is_file():
        return path.stat().st_size
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
        time.sleep(min(10, max(1, seconds // 3)))


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        for item in src.rglob("*"):
            if item.is_file():
                rel = item.relative_to(src)
                link_or_copy(item, dst / rel)
        return
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def latest_tokenizer_dir(checkpoint_root: Path) -> Path | None:
    meta_path = checkpoint_root / "train_metadata.yaml"
    if not meta_path.exists():
        return None
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    info = metadata.get("tokenizer_info") or {}
    tok_path = info.get("tokenizer_path")
    if not tok_path:
        return None
    path = Path(tok_path)
    return path.parent if path.name == "tokenizer.json" else path


def build_readme(repo_id: str, checkpoint_root: Path, epoch: int) -> str:
    model_card = Path(__file__).resolve().parents[1] / "docs" / "MODEL_CARD_KoHRM-Text-1.4B.md"
    if model_card.exists():
        return model_card.read_text(encoding="utf-8")

    return f"""---
license: other
tags:
- hrm-text
- korean
- terminal
- tool-use
- checkpoint
---

# {repo_id.split("/", 1)[-1]}

Raw HRM-Text FSDP2 checkpoint artifact.

- Source checkpoint root: `{checkpoint_root}`
- Epoch: `{epoch}`
- Upload policy: epoch-level upload only, to avoid slowing training with frequent network syncs.
- Format: HRM-Text training checkpoint (`fsdp2_epoch_*`) plus carry/config/tokenizer metadata.

This is primarily for monitoring and recovery. Final model-only exports should be produced with
`HRM-Text/conversion/convert_to_hf.py` after a checkpoint is selected.
"""


def stage_epoch(checkpoint_root: Path, stage_root: Path, repo_id: str, epoch: int) -> Path:
    src_epoch = checkpoint_root / f"fsdp2_epoch_{epoch}"
    if not src_epoch.exists():
        raise FileNotFoundError(src_epoch)

    stage_dir = stage_root / f"epoch_{epoch:03d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    link_or_copy(src_epoch, stage_dir / f"fsdp2_epoch_{epoch}")

    for name in ("all_config.yaml", "train_metadata.yaml"):
        src = checkpoint_root / name
        if src.exists():
            link_or_copy(src, stage_dir / name)

    for carry in sorted(checkpoint_root.glob(f"carry_epoch_{epoch}.*.pt")):
        link_or_copy(carry, stage_dir / carry.name)

    tok_dir = latest_tokenizer_dir(checkpoint_root)
    if tok_dir and tok_dir.exists():
        link_or_copy(tok_dir, stage_dir / "tokenizer")

    (stage_dir / "README.md").write_text(build_readme(repo_id, checkpoint_root, epoch), encoding="utf-8")
    manifest = {
        "repo_id": repo_id,
        "checkpoint_root": str(checkpoint_root),
        "epoch": epoch,
        "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage_size_bytes": dir_size(stage_dir),
    }
    (stage_dir / "upload_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stage_dir


def find_epochs(checkpoint_root: Path) -> list[int]:
    epochs: list[int] = []
    for path in checkpoint_root.glob("fsdp2_epoch_*"):
        match = EPOCH_RE.search(path.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(set(epochs))


def upload_epoch(api: HfApi, repo_id: str, stage_dir: Path, private: bool, num_workers: int) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=stage_dir,
        private=private,
        num_workers=num_workers,
        print_report=True,
        print_report_every=120,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-root", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--stage-root", default="/home/work/.data/hrm_text_hf_upload_stage")
    ap.add_argument("--env-file", default=str(ROOT / ".env"))
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--stable-seconds", type=int, default=120)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    checkpoint_root = Path(args.checkpoint_root)
    stage_root = Path(args.stage_root) / args.repo_id.replace("/", "__")
    token = get_token(Path(args.env_file) if args.env_file else None)
    api = HfApi(token=token)

    while True:
        for epoch in find_epochs(checkpoint_root):
            marker = stage_root / f"epoch_{epoch:03d}" / ".upload_complete"
            if marker.exists():
                continue
            print(f"[watch] detected epoch {epoch}; waiting for stable files", flush=True)
            wait_stable(checkpoint_root / f"fsdp2_epoch_{epoch}", args.stable_seconds)
            stage_dir = stage_epoch(checkpoint_root, stage_root, args.repo_id, epoch)
            size_gib = dir_size(stage_dir) / 1024**3
            print(f"[watch] staged epoch {epoch}: {stage_dir} ({size_gib:.2f} GiB)", flush=True)
            if args.dry_run:
                continue
            upload_epoch(api, args.repo_id, stage_dir, args.private, args.num_workers)
            marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()), encoding="utf-8")
            print(f"[watch] uploaded epoch {epoch} -> {args.repo_id}", flush=True)

        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
