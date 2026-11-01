"""Application-level binary temperature scaling of public, already-scaled probabilities.

This format is deliberately NOT Laya's load_calibration format. No logits API is assumed.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from production.runtime import schema_hash

FORMAT = "laya-guide.noul-temperature.v1"
EPS = 1e-6  # Public Laya answers are rounded; clipping prevents log(0), not information loss.


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Expected a finite probability between 0 and 1")
    return float(value)


def scale(p, temperature):
    p = probability(p)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    q = min(1 - EPS, max(EPS, p))
    z = (math.log(q) - math.log1p(-q)) / temperature
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1 + ez)


def nll(pairs):
    terms = []
    for p, label in pairs:
        q = min(1 - EPS, max(EPS, probability(p)))
        if type(label) is not bool:
            raise ValueError("Labels must be JSON true/false, not strings or integers")
        terms.append(-math.log(q if label else 1 - q))
    if not terms:
        raise ValueError("Empty dataset")
    return sum(terms) / len(terms)


def fit_temperature(pairs):
    if {label for _, label in pairs} != {False, True}:
        raise ValueError("Calibration requires examples of both classes")
    # Fixed deterministic log grid includes T=1. This can be audited without SciPy/PyTorch.
    candidates = [1.0] + [math.exp(math.log(.05) + i * (math.log(20) - math.log(.05)) / 400) for i in range(401)]
    temperature = min(candidates, key=lambda t: nll([(scale(p, t), y) for p, y in pairs]))
    return temperature


def metrics(pairs, temperature=1.0):
    values = [(scale(p, temperature), label) for p, label in pairs]
    loss = nll(values)
    brier = sum((p - int(y)) ** 2 for p, y in values) / len(values)
    correct = [((p >= .5) == y) for p, y in values]
    # Top-label ECE: confidence=max(p,1-p), 10 equal-width bins across [0,1].
    ece = 0.0
    for index in range(10):
        bucket = [(max(p, 1-p), (p >= .5) == y) for p, y in values if min(9, int(max(p, 1-p)*10)) == index]
        if bucket:
            mean_conf = sum(c for c, _ in bucket) / len(bucket)
            accuracy = sum(ok for _, ok in bucket) / len(bucket)
            ece += len(bucket) / len(values) * abs(mean_conf - accuracy)
    accepted = [(p, y) for p, y in values if p <= .2 or p >= .8]
    return {"n": len(values), "nll": loss, "brier": brier, "ece_top_label_10_bins": ece,
            "accuracy_at_0_5": sum(correct) / len(values),
            "illustrative_0_2_0_8_coverage": len(accepted) / len(values),
            "illustrative_0_2_0_8_error": (sum((p >= .5) != y for p, y in accepted)/len(accepted)) if accepted else None}


def prediction_identity(metadata, result, questions):
    model = result.get("routing", {}).get("model")
    if model not in metadata["artifacts"]:
        raise ValueError("Missing canonical routed model; cannot bind calibration")
    settings = metadata["settings"]
    return {"model": model, "artifact": metadata["artifacts"][model],
            "laya_version": metadata["laya_version"], "source_commit": metadata["source_commit"],
            "schema_sha256": schema_hash(questions),
            "max_len": settings["max_len"], "head_max_len": settings["head_max_len"]}


def read_cases(path):
    cases = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case.get("id"), str) or not case["id"]:
            raise ValueError(f"Invalid id on line {number}")
        probability(case["p"])
        if type(case.get("label")) is not bool:
            raise ValueError(f"Expected boolean label on line {number}")
        cases.append(case)
    if not cases:
        raise ValueError("No calibration records")
    if len({c["id"] for c in cases}) != len(cases) or len({c["text_sha256"] for c in cases}) != len(cases):
        raise ValueError("Duplicate IDs or texts in dataset")
    identity = cases[0]["identity"]
    if any(c["identity"] != identity for c in cases):
        raise ValueError("Do not mix checkpoints, schemas or token budgets in a calibration dataset")
    return cases


def fit_artifact(cases, cohort):
    pairs = [(c["p"], c["label"]) for c in cases]
    temperature = fit_temperature(pairs)
    return {"format": FORMAT, "identity": cases[0]["identity"], "cohort": cohort,
            "temperature": temperature, "fit_ids": [c["id"] for c in cases],
            "fit_text_hashes": [c["text_sha256"] for c in cases],
            "fit_metrics_before": metrics(pairs), "fit_metrics_after": metrics(pairs, temperature),
            "temperature_near_search_boundary": temperature <= .051 or temperature >= 19.9,
            "validated_for_production": False,
            "warning": "Fit metrics are not held-out metrics. Public probabilities have only 4 decimals. Validate on representative disjoint data; T cannot repair misclassification."}


def load_artifact(path):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("format") != FORMAT:
        raise ValueError("Not a guide binary calibration artifact (not interchangeable with Laya load_calibration)")
    scale(.5, artifact["temperature"])
    return artifact


def evaluate_artifact(artifact, cases):
    if artifact["identity"] != cases[0]["identity"]:
        raise ValueError("Evaluation and calibration identities differ")
    if set(artifact["fit_ids"]) & {c["id"] for c in cases} or set(artifact["fit_text_hashes"]) & {c["text_sha256"] for c in cases}:
        raise ValueError("Evaluation data overlap calibration data")
    pairs = [(c["p"], c["label"]) for c in cases]
    return {"cohort": artifact["cohort"], "before": metrics(pairs), "after": metrics(pairs, artifact["temperature"]),
            "validated_for_production": False,
            "note": "Disjointness is checked by exact text hashes/IDs, not by semantic duplicates or customer/thread groups. Human review of precision/recall, sample size and drift is still required."}


def apply_refund_calibration(result, artifact_path, metadata, questions):
    artifact = load_artifact(artifact_path)
    if artifact["identity"] != prediction_identity(metadata, result, questions):
        raise ValueError("Calibration identity mismatch: model/schema/budgets must match collection")
    if result.get("usage", {}).get("truncated"):
        raise ValueError("Refusing to calibrate an answer to a truncated document")
    calibrated = deepcopy(result)
    answer = calibrated["answers"]["refund_requested"]
    answer["noul"] = scale(answer["noul"], artifact["temperature"])
    answer["confidence"] = max(answer["noul"], 1 - answer["noul"])
    answer["answer_confidence"] = answer["confidence"]
    return calibrated, {"temperature": artifact["temperature"], "cohort": artifact["cohort"],
                        "validated_for_production": False, "format": FORMAT}
