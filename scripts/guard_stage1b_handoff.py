"""Guard the stage1b -> stage2b handoff.

This is a failsafe process, not the primary orchestrator. The normal handoff is
handled by `watch_stage1b_then_finish_chain.py`. This guard waits for stage1b
to finish and verifies that the next training stage starts shortly after. If
stage2b has not started within the grace window, it relaunches the handoff
watcher so the GPUs do not sit idle.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


HRM_ROOT = Path("/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text")
CKPT_ROOT = Path("/home/work/.data/hrm_text_checkpoints")
LOG_ROOT = Path("/home/work/.data/hrm_text_logs")

STAGE1B = CKPT_ROOT / "KoHRM-Text-1.4B-stage1b-hrm-fastcap-repeat-gbs180"
STAGE1B_RUN = "KoHRM-Text-1.4B-stage1b-hrm-fastcap-repeat"
NEXT_STAGE_RUNS = [
    "KoHRM-Text-1.4B-stage2b-hrm-full-nocap-extra-epoch1",
    "KoHRM-Text-1.4B-stage3b-local-terminal-repeat",
    "KoHRM-Text-1.4B-stage4b-korean-tool-finance-repeat",
]
HANDOFF_SCRIPT = "scripts/watch_stage1b_then_finish_chain.py"


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)


def process_rows() -> list[tuple[int, str, str]]:
    out = subprocess.run(
        ["ps", "-eo", "pid=,stat=,cmd="],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    rows: list[tuple[int, str, str]] = []
    for line in out.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if parts[1].startswith("Z"):
            continue
        rows.append((pid, parts[1], parts[2]))
    return rows


def training_pids(run_name: str) -> list[int]:
    pids: list[int] = []
    for pid, _stat, cmd in process_rows():
        if pid == os.getpid():
            continue
        if run_name in cmd and ("torchrun" in cmd or "pretrain.py" in cmd):
            pids.append(pid)
    return pids


def next_stage_active() -> bool:
    return any(training_pids(name) for name in NEXT_STAGE_RUNS)


def handoff_watcher_pids() -> list[int]:
    pids: list[int] = []
    for pid, _stat, cmd in process_rows():
        if pid == os.getpid():
            continue
        if HANDOFF_SCRIPT in cmd:
            pids.append(pid)
    return pids


def stage1b_final_step() -> int | None:
    epoch = STAGE1B / "fsdp2_epoch_1"
    info_path = STAGE1B / "epoch_1_info.json"
    carries = sorted(STAGE1B.glob("carry_epoch_1.*.pt"))
    if not (epoch.exists() and info_path.exists() and len(carries) >= 8):
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return int(info["global_step"])


def restart_handoff(retire_pid: int | None) -> None:
    for pid in handoff_watcher_pids():
        try:
            os.kill(pid, signal.SIGTERM)
            log(f"terminated stale handoff watcher pid={pid}")
        except ProcessLookupError:
            pass
    time.sleep(2)

    cmd = ["python", HANDOFF_SCRIPT]
    if retire_pid:
        cmd += ["--retire-pid", str(retire_pid)]
    subprocess.Popen(
        cmd,
        cwd=HRM_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=(LOG_ROOT / "watch_stage1b_then_finish_chain_20260526.log").open("ab"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log(f"restarted handoff watcher: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--grace-seconds", type=int, default=600)
    parser.add_argument("--retire-pid", type=int, default=1672885)
    args = parser.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log(
        "stage1b handoff guard started "
        f"poll={args.poll_seconds}s grace={args.grace_seconds}s"
    )

    final_seen_at: float | None = None
    stage1b_exit_seen_at: float | None = None
    restarted = False

    while True:
        if next_stage_active():
            log("next stage is active; guard exiting")
            return

        final_step = stage1b_final_step()
        if final_step is None:
            final_seen_at = None
            stage1b_exit_seen_at = None
            log("stage1b final checkpoint not ready yet")
            time.sleep(args.poll_seconds)
            continue

        if final_seen_at is None:
            final_seen_at = time.monotonic()
            log(f"stage1b final checkpoint detected: global_step={final_step}")

        pids = training_pids(STAGE1B_RUN)
        if pids:
            stage1b_exit_seen_at = None
            log(f"stage1b processes still running: pids={pids}")
            time.sleep(args.poll_seconds)
            continue

        if stage1b_exit_seen_at is None:
            stage1b_exit_seen_at = time.monotonic()
            log("stage1b processes have exited; waiting grace window for normal handoff")

        elapsed = time.monotonic() - stage1b_exit_seen_at
        if elapsed < args.grace_seconds:
            log(
                "normal handoff grace window active "
                f"elapsed={elapsed:.0f}s/{args.grace_seconds}s"
            )
            time.sleep(min(args.poll_seconds, max(1, int(args.grace_seconds - elapsed))))
            continue

        if not restarted:
            log("stage2b not active after grace window; restarting handoff watcher")
            restart_handoff(args.retire_pid)
            restarted = True
            time.sleep(args.poll_seconds)
            continue

        if next_stage_active():
            log("next stage became active after handoff restart; guard exiting")
            return

        log("handoff restart did not start next stage yet; continuing to monitor")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
