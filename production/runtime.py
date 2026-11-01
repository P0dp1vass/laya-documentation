"""Shared, lazy Laya runtime. Importing this module never imports ML packages."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("auto", "english", "multilingual", "typed-decisions")


@dataclass(frozen=True)
class Settings:
    model: str = "auto"
    device: str | None = None
    max_len: int | None = None
    head_max_len: int | None = None
    batch_size: int = 8
    max_loaded: int = 2

    def __post_init__(self):
        if self.model not in MODEL_NAMES:
            raise ValueError(f"model must be one of {MODEL_NAMES}")
        for name in ("max_len", "head_max_len", "batch_size", "max_loaded"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if self.max_len and self.head_max_len and self.head_max_len >= self.max_len:
            raise ValueError("head_max_len must leave room within max_len for the state")


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=MODEL_NAMES, default=os.getenv("LAYA_MODEL", "auto"))
    parser.add_argument("--device", default=os.getenv("LAYA_DEVICE"), help="cpu, cuda, cuda:0, mps, xpu; omitted: automatic")
    parser.add_argument("--max-len", type=int, default=os.getenv("LAYA_MAX_LEN"))
    parser.add_argument("--head-max-len", type=int, default=os.getenv("LAYA_HEAD_MAX_LEN"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-loaded", type=int, default=2)


def settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings(**{key: getattr(args, key) for key in Settings.__dataclass_fields__})


def settings_from_env() -> Settings:
    def optional_int(key):
        value = os.getenv(key)
        return int(value) if value else None
    return Settings(
        model=os.getenv("LAYA_MODEL", "auto"), device=os.getenv("LAYA_DEVICE") or None,
        max_len=optional_int("LAYA_MAX_LEN"), head_max_len=optional_int("LAYA_HEAD_MAX_LEN"),
        batch_size=int(os.getenv("LAYA_BATCH_SIZE", "8")),
        max_loaded=int(os.getenv("LAYA_MAX_LOADED", "2")),
    )


def emit_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def schema_hash(questions: dict) -> str:
    # Do not sort keys: option order is part of the model prompt.
    encoded = json.dumps(questions, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class Runtime:
    """Own one Router per process; no tokenizer/model activity until explicitly constructed."""

    def __init__(self, settings: Settings):
        self.settings = settings
        # These defaults must precede even transitive Transformers/tokenizers imports.
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_TORCH", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.lock = json.loads((ROOT / "models.lock.json").read_text(encoding="utf-8"))
        self.version = importlib.metadata.version("laya")
        if self.version != self.lock["laya_version"]:
            raise RuntimeError(f"Expected laya=={self.lock['laya_version']}, found {self.version}; use the provided environment")
        sources, revisions, artifacts = {}, {}, {}
        for name, spec in self.lock["models"].items():
            local = os.getenv(spec["local_env"])
            if local:
                path = Path(local).expanduser().resolve(strict=True)
                # Complete snapshots also carry encoder/ and tokenizer/; Laya checks their contents.
                for required in ("rl_agent_config.json", "model.safetensors", "encoder", "tokenizer"):
                    if not (path / required).exists():
                        raise ValueError(f"Incomplete local checkpoint: missing {path / required}")
                sources[name] = str(path)
                revisions[name] = None
                weights = (path / "model.safetensors").stat()
                artifacts[name] = {
                    "source": str(path), "revision": None,
                    "config_sha256": hashlib.sha256((path / "rl_agent_config.json").read_bytes()).hexdigest(),
                    "weights_size": weights.st_size, "weights_mtime_ns": weights.st_mtime_ns,
                    "identity_note": "Local size/mtime/config identity, not a cryptographic weights digest",
                }
            else:
                sources[name] = spec["repo_id"]
                revisions[name] = spec["revision"]
                artifacts[name] = {"source": spec["repo_id"], "revision": spec["revision"]}
        self.artifacts = artifacts
        # Library messages must not corrupt stdout JSON/JSONL.
        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            from laya import Router
            self.router = Router(models=sources, revisions=revisions, standalone_repos=True,
                                 device=settings.device, max_loaded=settings.max_loaded,
                                 auto_task_detection=False)

    def _kwargs(self, model=None, lang=None):
        selected = model if model is not None else self.settings.model
        if selected not in MODEL_NAMES:
            raise ValueError(f"Unknown model {selected!r}")
        values = {"model": None if selected == "auto" else selected, "lang": lang,
                  "max_len": self.settings.max_len, "head_max_len": self.settings.head_max_len}
        return {key: value for key, value in values.items() if value is not None}

    def predict(self, state, questions, *, model=None, lang=None):
        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            return self.router.predict(state, questions, **self._kwargs(model, lang))

    def predict_batch(self, states, questions, *, model=None, lang=None):
        options = self._kwargs(model, lang)
        requests = [{"state": state, "questions": questions, **options} for state in states]
        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            return self.router.predict_batch(requests, batch_size=self.settings.batch_size, sort_by_length=True)

    def metadata(self):
        return {"laya_version": self.version, "source_commit": self.lock["source_commit"],
                "settings": asdict(self.settings), "artifacts": self.artifacts,
                "loaded_revisions": self.router.loaded_revisions}

    def close(self):
        self.router.unload()
