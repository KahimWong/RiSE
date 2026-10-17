"""Publish the demo and shared inference code to a free ZeroGPU Space."""

import argparse
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import HfApi, SpaceHardware, get_token
from huggingface_hub.constants import HF_TOKEN_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="HF username/space-name")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # Prefer the user's stored login if a stale token exists in the environment.
    token_path = Path(HF_TOKEN_PATH)
    token = token_path.read_text().strip() if token_path.exists() else get_token()
    if not token:
        raise SystemExit("Authenticate first with hf auth login.")
    api = HfApi(token=token)
    with tempfile.TemporaryDirectory(prefix="rise-space-") as directory:
        stage = Path(directory)
        shutil.copytree(
            root / "demo",
            stage,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copytree(
            root / "rise",
            stage / "rise",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copy2(root / "LICENSE", stage / "LICENSE")
        api.create_repo(
            args.repo_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=SpaceHardware.ZERO_A10G,
            exist_ok=True,
        )
        # Also select ZeroGPU for an existing Space; never fall back to paid GPUs.
        api.request_space_hardware(args.repo_id, hardware=SpaceHardware.ZERO_A10G)
        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=stage,
            commit_message="Deploy RiSE-9-DCT ZeroGPU demo",
        )
        print(commit.commit_url)
        print(f"https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
