"""Рекомендация отдела поддержки без вызова внешней LLM и без отправки сообщения.

Запуск из корня: python -m examples.02_route --help
Вероятность и answer_confidence не равны измеренной точности. По умолчанию
все решения направлены на review; флаг --validated-policy не выполняет валидацию.
Качество русскоязычного входа на английских checkpoints здесь не установлено.
"""

import argparse
import math
import sys

from production.policies import route_policy
from production.runtime import Runtime, add_runtime_arguments, emit_json, settings_from_args
from production.schemas import ROUTE_QUESTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="The integration returns errors and our team cannot use the product.",
                        help="Текст обращения; по умолчанию синтетическая техническая проблема.")
    parser.add_argument("--min-answer-confidence", type=float, default=0.9,
                        help="Порог answer_confidence, не entropy-confidence (default: 0.9).")
    parser.add_argument("--validated-policy", action="store_true",
                        help="Заявить внешнюю проверку политики на ваших отложенных данных; иначе review.")
    add_runtime_arguments(parser)
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("--text не должен быть пустым")
    if not math.isfinite(args.min_answer_confidence) or not 0 <= args.min_answer_confidence <= 1:
        parser.error("--min-answer-confidence должен быть конечным числом в [0, 1]")
    try:
        runtime = Runtime(settings_from_args(args))
        try:
            raw = runtime.predict({"text": args.text}, ROUTE_QUESTIONS)
            decision = route_policy(raw, args.min_answer_confidence, validated=args.validated_policy)
            routing = {"target_queue": decision["department"] if decision["action"] == "route_candidate" else "human_review",
                       "dispatch_performed": False, "external_llm_called": False}
            emit_json({"runtime": runtime.metadata(), "result": raw,
                       "decision": decision, "routing": routing})
        finally:
            runtime.close()
    except Exception as exc:
        print(f"route: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
