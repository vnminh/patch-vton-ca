#!/usr/bin/env python3
"""Register/start the fixed VTON experiment in Supervisor after the old run stops."""

import argparse
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Completed checkpoint, or its last.ckpt symlink")
    parser.add_argument("--python", type=Path, default=Path("/venv/ai/bin/python"))
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/high-resolution-viton-zalando-dataset"))
    parser.add_argument("--dry-run", action="store_true", help="Print the service configuration without starting training")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint.resolve(strict=True)
    python = args.python.absolute()
    if not python.is_file():
        parser.error(f"Python not found: {python}")
    for split in ("train", "test"):
        for folder in ("image", "cloth", "cloth-mask", "agnostic-mask", "image-parse-v3"):
            if not (args.data_root / split / folder).is_dir():
                parser.error(f"Missing dataset directory: {args.data_root / split / folder}")
    for path in (repo, checkpoint, python, args.data_root):
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(path)):
            parser.error(f"Unsupported path characters: {path}")
    name = "pft-vton-garment-fix"
    log = repo / "logs" / "supervisor-training.log"
    config = f"""[program:{name}]
command={python} -u {repo}/train.py experiment=viton-pft-xl-512x384-garment-fix resume_checkpoint={checkpoint}
directory={repo}
environment=VITONHD_ROOT="{args.data_root}",PFT_XL_CKPT="{repo}/checkpoints/pft-xl_step400k_ema.ckpt",PYTHONUNBUFFERED="1"
autostart=false
autorestart=false
startsecs=10
startretries=0
stopasgroup=true
killasgroup=true
stopsignal=INT
stopwaitsecs=120
stdout_logfile={log}
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=3
redirect_stderr=true
"""
    if args.dry_run:
        print(config)
        return
    if os.geteuid() != 0:
        parser.error("Run as root to register the Supervisor service")
    # Never launch alongside the old training run or kill it implicitly.
    running = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            arguments = (proc / "cmdline").read_bytes().split(b"\0")
            if any(Path(arg.decode(errors="replace")).name == "train.py" for arg in arguments if arg):
                running.append(proc.name)
        except (OSError, ValueError):
            continue
    if running:
        parser.error(f"Stop the current training process first (PIDs: {', '.join(running)}). No changes made.")
    # New runs write under a distinct experiment name and retain two checkpoints.
    log.parent.mkdir(parents=True, exist_ok=True)
    config_path = Path("/etc/supervisor/conf.d") / f"{name}.conf"
    config_path.write_text(config, encoding="utf-8")
    subprocess.run(["supervisorctl", "reread"], check=True)
    subprocess.run(["supervisorctl", "update", name], check=True)
    subprocess.run(["supervisorctl", "start", name], check=True)
    print(f"Status: supervisorctl status {name}\nLog: {log}")
    print("To restart later: stop this service and pass its newest completed last.ckpt to this script.")


if __name__ == "__main__":
    main()
