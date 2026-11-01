"""Чистые политики без денежных операций, удаления контента или сетевых вызовов.

validated=True — заявление вызывающей стороны о внешней проверке политики,
а не результат проверки этим модулем. Пороги учебные; вероятности и confidence
не являются гарантией правильности. Повреждённый ответ приводит к review;
неверная конфигурация порогов — к ValueError, которую CLI/API обязаны показать.
"""

import math

from production.schemas import MODERATION_QUESTIONS, PRIORITY_QUESTIONS, ROUTE_QUESTIONS


def _probability(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}: expected a number")
    if not 0 <= value <= 1 or not math.isfinite(value):
        raise ValueError(f"{field}: expected a finite value in [0, 1]")
    return float(value)


def _decision(action, reason, validated, **evidence):
    return {"action": action, "reason": reason, "policy_validated": validated, **evidence}


def _answers(result, validated):
    if not isinstance(validated, bool):
        raise ValueError("validated must be a bool")
    if not isinstance(result, dict):
        return None, _decision("review", "invalid_result: expected an object", validated)
    usage = result.get("usage")
    if not isinstance(usage, dict) or not isinstance(usage.get("truncated"), bool):
        return None, _decision("review", "invalid_result: missing boolean usage.truncated", validated)
    if usage["truncated"]:
        return None, _decision("review", "input_truncated", validated)
    if usage.get("options"):
        return None, _decision("review", "option_budget_degraded", validated)
    answers = result.get("answers")
    if not isinstance(answers, dict):
        return None, _decision("review", "invalid_result: missing answers object", validated)
    return answers, None


def _answer(answers, qid, kind):
    answer = answers.get(qid)
    if not isinstance(answer, dict) or answer.get("type") != kind:
        raise ValueError(f"{qid}: missing answer or wrong type")
    _probability(answer.get("confidence"), f"{qid}.confidence")
    _probability(answer.get("answer_confidence"), f"{qid}.answer_confidence")
    return answer


def _distribution(answer, labels):
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
        raise ValueError("probabilities: labels do not match the question schema")
    values = {key: _probability(probabilities[key], f"probabilities.{key}") for key in labels}
    # Official results round each probability to four decimals independently.
    if not math.isclose(sum(values.values()), 1.0, rel_tol=0, abs_tol=0.001):
        raise ValueError("probabilities: sum is not one within rounding tolerance")
    return values


def refund_policy(result, negative_threshold=0.2, positive_threshold=0.8, validated=False):
    """Распознать просьбу, но НЕ право на возврат и НЕ исполнить возврат."""
    low = _probability(negative_threshold, "negative_threshold")
    high = _probability(positive_threshold, "positive_threshold")
    if low >= high:
        raise ValueError("negative_threshold must be less than positive_threshold")
    answers, failure = _answers(result, validated)
    if failure is not None:
        return failure
    try:
        answer = _answer(answers, "refund_requested", "noul")
        probability = _probability(answer.get("noul"), "refund_requested.noul")
    except ValueError as exc:
        return _decision("review", f"invalid_result: {exc}", validated)
    evidence = {"request_probability": probability, "answer_confidence": answer["answer_confidence"],
                "negative_threshold": low, "positive_threshold": high,
                "eligibility_assessed": False, "refund_requested": None}
    if not validated:
        return _decision("review", "policy_not_validated", validated, **evidence)
    if answer["answer_confidence"] < 0.8:
        return _decision("review", "low_answer_confidence", validated, **evidence)
    if probability >= high:
        return _decision("refund_review_candidate", "refund_request_detected_not_eligibility", validated,
                         **{**evidence, "refund_requested": True})
    if probability <= low:
        return _decision("no_refund_request", "no_explicit_refund_request_detected", validated,
                         **{**evidence, "refund_requested": False})
    return _decision("review", "probability_in_review_band", validated, **evidence)


def route_policy(result, min_answer_confidence=0.9, validated=False):
    """Рекомендовать очередь; не вызывать LLM и не передавать сообщение наружу."""
    threshold = _probability(min_answer_confidence, "min_answer_confidence")
    answers, failure = _answers(result, validated)
    if failure is not None:
        return failure
    try:
        answer = _answer(answers, "department", "choice")
        probabilities = _distribution(answer, ROUTE_QUESTIONS["department"]["criteria"])
        department = answer.get("choice")
        if not isinstance(department, str) or department not in probabilities:
            raise ValueError("department.choice: unknown department")
        peak = max(probabilities.values())
        if probabilities[department] != peak:
            raise ValueError("department.choice: inconsistent with probabilities")
    except ValueError as exc:
        return _decision("review", f"invalid_result: {exc}", validated)
    evidence = {"department": department, "answer_confidence": answer["answer_confidence"],
                "min_answer_confidence": threshold}
    if not validated:
        return _decision("review", "policy_not_validated", validated, **evidence)
    if answer["answer_confidence"] < threshold:
        return _decision("review", "low_answer_confidence", validated, **evidence)
    if list(probabilities.values()).count(peak) > 1:
        return _decision("review", "ambiguous_modal_department", validated, **evidence)
    if department == "other":
        return _decision("review", "no_specialist_department", validated, **evidence)
    return _decision("route_candidate", "department_confidence_gate_passed", validated, **evidence)


def moderation_policy(result, validated=False):
    """Только advisory: это не защитная граница и не команда удалить сообщение."""
    answers, failure = _answers(result, validated)
    if failure is not None:
        return failure
    probabilities = {}
    confidences = {}
    try:
        for qid in MODERATION_QUESTIONS:
            answer = _answer(answers, qid, "noul")
            probabilities[qid] = _probability(answer.get("noul"), f"{qid}.noul")
            confidences[qid] = answer["answer_confidence"]
    except ValueError as exc:
        return _decision("review", f"invalid_result: {exc}", validated)
    evidence = {"risk_probabilities": probabilities, "answer_confidences": confidences,
                "negative_threshold": 0.1, "positive_threshold": 0.9, "advisory_only": True}
    if not validated:
        return _decision("review", "policy_not_validated", validated, **evidence)
    if any(value < 0.9 for value in confidences.values()):
        return _decision("review", "low_answer_confidence", validated, **evidence)
    if any(value >= 0.9 for value in probabilities.values()):
        return _decision("block_candidate", "at_least_one_risk_detected", validated, **evidence)
    if all(value <= 0.1 for value in probabilities.values()):
        return _decision("allow_candidate", "no_listed_risk_detected_not_a_safety_guarantee", validated, **evidence)
    return _decision("review", "probability_in_review_band", validated, **evidence)


def priority_policy(result, validated=False):
    """Ожидание индекса и модальный класс различны; SLA здесь не назначается."""
    answers, failure = _answers(result, validated)
    if failure is not None:
        return failure
    levels = PRIORITY_QUESTIONS["priority"]["criteria"]
    labels = [str(index) for index in range(len(levels))]
    try:
        answer = _answer(answers, "priority", "score")
        probabilities = _distribution(answer, labels)
        score = answer.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("priority.score: expected a number")
        if not 0 <= score <= len(levels) - 1 or not math.isfinite(score):
            raise ValueError("priority.score: invalid expected zero-based index")
        expected = sum(int(key) * value for key, value in probabilities.items())
        if not math.isclose(score, expected, rel_tol=0, abs_tol=0.001):
            raise ValueError("priority.score: inconsistent with probabilities")
        legend = answer.get("legend")
        if not isinstance(legend, dict) or legend != dict(zip(labels, levels)):
            raise ValueError("priority.legend: does not match the question schema")
    except ValueError as exc:
        return _decision("review", f"invalid_result: {exc}", validated)
    modal_key = max(labels, key=probabilities.get)
    modal_index = int(modal_key)
    evidence = {"expected_index": score, "modal_index": modal_index,
                "modal_label": levels[modal_index], "modal_probability": probabilities[modal_key],
                "answer_confidence": answer["answer_confidence"],
                "min_answer_confidence": 0.9, "sla_assigned": False}
    if not validated:
        return _decision("review", "policy_not_validated", validated, **evidence)
    if answer["answer_confidence"] < 0.9:
        return _decision("review", "low_answer_confidence", validated, **evidence)
    if list(probabilities.values()).count(probabilities[modal_key]) > 1:
        return _decision("review", "ambiguous_modal_priority", validated, **evidence)
    return _decision("priority_candidate", "modal_priority_confidence_gate_passed", validated, **evidence)
