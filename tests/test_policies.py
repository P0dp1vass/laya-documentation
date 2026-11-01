"""Consumer-visible policy boundaries; no model loading or mocked inference."""
import importlib
import io
import unittest

from production.policies import refund_policy, route_policy, moderation_policy, priority_policy
from production.schemas import MODERATION_QUESTIONS, PRIORITY_QUESTIONS


def result(answers, truncated=False):
    return {"answers": answers, "usage": {"truncated": truncated}}


def binary(p):
    return {"type": "noul", "noul": p, "confidence": max(p, 1-p), "answer_confidence": max(p, 1-p)}


class RefundPolicyTests(unittest.TestCase):
    def test_unvalidated_answer_never_authorizes_action(self):
        decision = refund_policy(result({"refund_requested": binary(.99)}))
        self.assertEqual(decision["action"], "review")
        self.assertIsNone(decision["refund_requested"])

    def test_positive_negative_and_review_band(self):
        for p, expected, boolean in [(.8, "refund_review_candidate", True), (.2, "no_refund_request", False), (.5, "review", None)]:
            with self.subTest(p=p):
                decision = refund_policy(result({"refund_requested": binary(p)}), validated=True)
                self.assertEqual(decision["action"], expected)
                self.assertIs(decision["refund_requested"], boolean)
                self.assertFalse(decision["eligibility_assessed"])

    def test_truncated_or_collapsed_options_force_review(self):
        raw = result({"refund_requested": binary(.99)}, truncated=True)
        self.assertEqual(refund_policy(raw, validated=True)["action"], "review")
        raw["usage"] = {"truncated": False, "options": {"refund_requested": {"options_distinct": 1}}}
        self.assertEqual(refund_policy(raw, validated=True)["action"], "review")

    def test_invalid_probability_never_becomes_false(self):
        for value in (True, None, "0.9", float("nan"), float("inf"), -.1, 1.1):
            raw = result({"refund_requested": {**binary(.9), "noul": value}})
            self.assertEqual(refund_policy(raw, validated=True)["action"], "review")

    def test_bad_threshold_configuration_raises(self):
        for low, high in [(.8,.2), (.5,.5), (-.1,.8), (.2,float("nan"))]:
            with self.assertRaises(ValueError):
                refund_policy(result({}), low, high)


class OtherPolicyTests(unittest.TestCase):
    def test_router_does_not_dispatch_other_or_ambiguous_top(self):
        answer = {"type":"choice", "choice":"other", "probabilities":{"billing":.01,"technical":.01,"sales":.01,"other":.97},"confidence":.9,"answer_confidence":.97}
        self.assertEqual(route_policy(result({"department":answer}), validated=True)["action"], "review")
        answer.update(choice="billing", probabilities={"billing":.5,"technical":.5,"sales":0.,"other":0.}, answer_confidence=1.)
        self.assertEqual(route_policy(result({"department":answer}), min_answer_confidence=0, validated=True)["action"], "review")

    def test_moderation_checks_all_risks(self):
        ids = list(MODERATION_QUESTIONS)
        raw = result({ids[0]:binary(.02),ids[1]:binary(.98)})
        self.assertEqual(moderation_policy(raw, validated=True)["action"], "block_candidate")
        raw["answers"][ids[1]] = binary(.5)
        self.assertEqual(moderation_policy(raw, validated=True)["action"], "review")
        raw["answers"][ids[1]] = binary(.02)
        self.assertEqual(moderation_policy(raw, validated=True)["action"], "allow_candidate")

    def test_priority_reports_modal_not_rounded_mean(self):
        levels = PRIORITY_QUESTIONS["priority"]["criteria"]
        answer = {"type":"score", "score":1.2, "legend":{str(i):v for i,v in enumerate(levels)},
                  "probabilities":{"0":.6,"1":0.,"2":0.,"3":.4},"confidence":.5,"answer_confidence":.6}
        decision = priority_policy(result({"priority":answer}))
        self.assertEqual(decision["expected_index"], 1.2)
        self.assertEqual(decision["modal_index"], 0)
        self.assertNotEqual(round(decision["expected_index"]), decision["modal_index"])
        self.assertFalse(decision["sla_assigned"])


class BatchInputTests(unittest.TestCase):
    def test_order_and_chunk_boundaries(self):
        module = importlib.import_module("examples.05_batch")
        rows = io.StringIO('{"id":"a","text":"first"}\n{"id":"b","text":"second"}\n{"id":"c","text":"third"}\n')
        chunks = list(module._chunks(rows, 2))
        self.assertEqual([[r[1] for r in c] for c in chunks], [["a","b"],["c"]])

    def test_duplicate_or_nonstandard_input_rejected(self):
        module = importlib.import_module("examples.05_batch")
        for text in ['{"id":"a","text":"x"}\n{"id":"a","text":"y"}\n', '{"id":"a","id":"b","text":"x"}\n', '{"id":"a","text":NaN}\n', '\n']:
            with self.assertRaises(ValueError):
                list(module._chunks(io.StringIO(text), 2))


if __name__ == "__main__":
    unittest.main()
