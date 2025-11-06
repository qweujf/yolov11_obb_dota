import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    cmd = [sys.executable, "-m", "ultralytics", "val", f"cfg={args.config}", f"weights={args.ckpt}"]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()


