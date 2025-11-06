import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=False, help="原始DOTA路径")
    parser.add_argument("--dst", type=str, required=False, help="输出OBB路径")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "data_processing" / "doto2obb.py"
    cmd = [sys.executable, str(script_path)]
    if args.src:
        cmd += ["--src", args.src]
    if args.dst:
        cmd += ["--dst", args.dst]
    env = os.environ.copy()
    subprocess.run(cmd, env=env, check=False)


if __name__ == "__main__":
    main()


