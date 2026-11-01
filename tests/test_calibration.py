"""Numerical and leakage boundaries for the model-independent calibrator."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from production.calibration import (
    apply_refund_calibration, evaluate_artifact, fit_artifact, fit_temperature,
    metrics, nll, prediction_identity, read_cases, scale,
)
from production.schemas import REFUND_QUESTIONS


def metadata():
    return {"artifacts":{"english":{"source":"fixture","revision":"fixture-revision"}},
            "laya_version":"0.4.21", "source_commit":"fixture-commit",
            "settings":{"max_len":None,"head_max_len":None}}


def raw_result():
    return {"routing":{"model":"english"},"usage":{"truncated":False},
            "answers":{"refund_requested":{"type":"noul","noul":.9,"confidence":.9,"answer_confidence":.9}}}


def cases():
    identity = prediction_identity(metadata(), raw_result(), REFUND_QUESTIONS)
    # Mathematical fixtures, deliberately not reported as actual model predictions.
    return [{"id":f"fit-{i}", "text_sha256":f"text-{i}", "p":p,"label":y,"identity":identity}
            for i,(p,y) in enumerate([(.99,True),(.99,False),(.01,False),(.01,True),(.8,True),(.2,False)])]


class CalibrationTests(unittest.TestCase):
    def test_positive_temperature_preserves_order_and_class_side(self):
        inputs = [.01,.1,.4,.5,.6,.9,.99]
        for temperature in (.05,.5,1.,2.,20.):
            outputs = [scale(p,temperature) for p in inputs]
            self.assertEqual(outputs, sorted(outputs))
            self.assertEqual([p>=.5 for p in inputs], [p>=.5 for p in outputs])

    def test_overconfidence_fitting_reduces_fit_nll(self):
        pairs = [(c["p"],c["label"]) for c in cases()]
        temperature = fit_temperature(pairs)
        self.assertGreater(temperature, 1)
        self.assertLess(nll([(scale(p,temperature),y) for p,y in pairs]), nll(pairs))

    def test_single_class_fit_and_invalid_probability_refused(self):
        with self.assertRaises(ValueError):
            fit_temperature([(.8,True),(.7,True)])
        for value in (True,float("nan"),-1.,2.):
            with self.assertRaises(ValueError):
                scale(value,1.)

    def test_empty_acceptance_is_not_zero_error(self):
        report = metrics([(.5,True),(.5,False)])
        self.assertEqual(report["illustrative_0_2_0_8_coverage"],0)
        self.assertIsNone(report["illustrative_0_2_0_8_error"])

    def test_evaluation_refuses_same_text_even_with_new_id(self):
        dataset = cases()
        artifact = fit_artifact(dataset,"fixture-only")
        evaluation = deepcopy(dataset)
        for row in evaluation:
            row["id"] = "eval-" + row["id"]
        with self.assertRaises(ValueError):
            evaluate_artifact(artifact,evaluation)

    def test_disjoint_evaluation_does_not_certify_production(self):
        dataset = cases()
        artifact = fit_artifact(dataset,"fixture-only")
        evaluation = deepcopy(dataset)
        for row in evaluation:
            row["id"] = "eval-" + row["id"]
            row["text_sha256"] = "eval-" + row["text_sha256"]
        self.assertFalse(evaluate_artifact(artifact,evaluation)["validated_for_production"])

    def test_calibration_retains_raw_and_rejects_other_budget(self):
        artifact = fit_artifact(cases(),"fixture-only")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"calibration.json"
            path.write_text(json.dumps(artifact),encoding="utf-8")
            raw = raw_result()
            before = deepcopy(raw)
            calibrated, _ = apply_refund_calibration(raw,path,metadata(),REFUND_QUESTIONS)
            self.assertEqual(raw,before)
            self.assertLess(calibrated["answers"]["refund_requested"]["noul"],.9)
            different = metadata()
            different["settings"]["max_len"] = 8192
            with self.assertRaises(ValueError):
                apply_refund_calibration(raw,path,different,REFUND_QUESTIONS)
            raw["usage"]["truncated"] = True
            with self.assertRaises(ValueError):
                apply_refund_calibration(raw,path,metadata(),REFUND_QUESTIONS)

    def test_reader_rejects_mixed_checkpoint_records(self):
        dataset = cases()
        dataset[0] = deepcopy(dataset[0])
        dataset[0]["identity"]["model"] = "multilingual"
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"collected.jsonl"
            path.write_text(''.join(json.dumps(row)+'\n' for row in dataset),encoding="utf-8")
            with self.assertRaises(ValueError):
                read_cases(path)


if __name__ == "__main__":
    unittest.main()
