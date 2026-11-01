"""Download complete pinned snapshots; this utility does not load or run models."""
import argparse
import json
from pathlib import Path

from production.runtime import ROOT, emit_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", "english", "multilingual", "typed-decisions"), default="all")
    parser.add_argument("--directory", type=Path, default=Path("models"))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    lock = json.loads((ROOT / "models.lock.json").read_text(encoding="utf-8"))
    names = lock["models"] if args.model == "all" else [args.model]
    snapshots = {}
    for name in names:
        spec = lock["models"][name]
        destination = args.directory.resolve() / name
        marker = destination / "snapshot_identity.json"
        expected = {"repo_id": spec["repo_id"], "revision": spec["revision"]}
        if destination.exists() and any(destination.iterdir()):
            if not marker.exists() or json.loads(marker.read_text(encoding="utf-8")) != expected:
                raise ValueError(f"Refusing to mix an unrecognized/existing snapshot with {destination}")
        path = snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"],
                                 local_dir=str(destination), allow_patterns=[
                                     "model.safetensors", "rl_agent_config.json", "config.json",
                                     "encoder/*", "tokenizer/*",
                                 ])
        marker.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        snapshots[spec["local_env"]] = path
    emit_json({"snapshots": snapshots, "instruction": "Set these environment variables and HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 on the target host"})


if __name__ == "__main__":
    main()
