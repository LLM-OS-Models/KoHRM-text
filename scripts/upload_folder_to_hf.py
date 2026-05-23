"""Upload a local folder to Hugging Face Hub using HF_TOKEN from env or .env."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[2]


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
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_token(env_file: Path | None) -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token
    for path in [p for p in [env_file, ROOT / ".env", Path.cwd() / ".env", Path.home() / ".env"] if p]:
        values = parse_env_file(path)
        token = values.get("HF_TOKEN") or values.get("HUGGINGFACE_TOKEN")
        if token:
            return token
    raise RuntimeError("HF_TOKEN is not available in environment or .env")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    ap.add_argument("--path-in-repo", default=None)
    ap.add_argument("--commit-message", default="Upload HRM-Text artifacts")
    ap.add_argument("--env-file", default=str(ROOT / ".env"))
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--large", action="store_true", help="Use upload_large_folder. Does not support --path-in-repo.")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(folder)
    token = get_token(Path(args.env_file) if args.env_file else None)
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, exist_ok=True, private=args.private)

    if args.large:
        if args.path_in_repo:
            raise ValueError("--large does not support --path-in-repo")
        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=folder,
            private=args.private,
            num_workers=args.num_workers,
            print_report=True,
            print_report_every=120,
        )
    else:
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=folder,
            path_in_repo=args.path_in_repo,
            commit_message=args.commit_message,
        )

    print(f"uploaded {folder} -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
