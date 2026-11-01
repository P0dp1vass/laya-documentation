"""Collect real Laya probabilities, fit binary T, evaluate on disjoint labelled data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from production.calibration import evaluate_artifact, fit_artifact, load_artifact, prediction_identity, read_cases
from production.runtime import Runtime, add_runtime_arguments, emit_json, settings_from_args
from production.schemas import REFUND_QUESTIONS


def collect(args):
    settings = settings_from_args(args)
    if settings.model == "auto":
        raise ValueError("Collection requires --model english, multilingual or typed-decisions; do not mix checkpoints")
    # Validate all labels before opening the output or loading model weights.
    rows, ids, hashes = [], set(), set()
    for number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("id"), str) or not row["id"] or not isinstance(row.get("text"), str) or not row["text"].strip() or type(row.get("label")) is not bool:
            raise ValueError(f"Line {number}: expected non-empty id/text strings and boolean label")
        digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        if row["id"] in ids or digest in hashes:
            raise ValueError(f"Line {number}: duplicate id or text")
        ids.add(row["id"])
        hashes.add(digest)
        rows.append((row, digest))
    if not rows:
        raise ValueError("Empty labelled dataset")
    with args.output.open("x", encoding="utf-8") as output:
        runtime = Runtime(settings)
        try:
            for row, digest in rows:
                result = runtime.predict({"text": row["text"]}, REFUND_QUESTIONS)
                if result["usage"].get("truncated"):
                    raise ValueError(f"Truncated calibration input {row['id']}")
                record = {"id": row["id"], "text_sha256": digest, "label": row["label"],
                          "p": result["answers"]["refund_requested"]["noul"],
                          "identity": prediction_identity(runtime.metadata(), result, REFUND_QUESTIONS)}
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
        finally:
            runtime.close()
    emit_json({"status": "complete", "records": len(rows), "output": str(args.output)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gather = commands.add_parser("collect", help="Runs real model inference on human-labelled text")
    add_runtime_arguments(gather)
    gather.add_argument("--input", type=Path, required=True)
    gather.add_argument("--output", type=Path, required=True)
    fit = commands.add_parser("fit", help="No model loading: fit T on collected public probabilities")
    fit.add_argument("--input", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--cohort", required=True, help="Describe domain/language/question version, e.g. billing-en-2026Q4")
    evaluate = commands.add_parser("evaluate", help="No model loading: evaluate with exact-text/ID leakage checks")
    evaluate.add_argument("--artifact", type=Path, required=True)
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "collect":
            collect(args)
            return
        if args.command == "fit":
            result = fit_artifact(read_cases(args.input), args.cohort)
        else:
            result = evaluate_artifact(load_artifact(args.artifact), read_cases(args.input))
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
        emit_json(result)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}. Existing outputs are not overwritten; collection output may be partial after failure.", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
