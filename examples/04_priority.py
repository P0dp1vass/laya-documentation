"""Рекомендация приоритета по упорядоченной рубрике, а не назначение SLA.

Запуск из корня: python -m examples.04_priority --help
score — математическое ожидание индекса 0..3; модальный класс вычисляется
через argmax вероятностей, НЕ round(score). Учебный порог answer_confidence=0.9.
Даже высокая уверенность не гарантирует правильный приоритет. Для русского
текста качество каждого выбранного checkpoint нужно проверять отдельно.
"""

import argparse
import sys

from production.policies import priority_policy
from production.runtime import Runtime, add_runtime_arguments, emit_json, settings_from_args
from production.schemas import PRIORITY_QUESTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="Production is unavailable for all users. No workaround is available.",
                        help="Описание операционного влияния; по умолчанию синтетическая общая недоступность.")
    parser.add_argument("--validated-policy", action="store_true",
                        help="Заявить внешнюю проверку политики на ваших данных; иначе только review.")
    add_runtime_arguments(parser)
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("--text не должен быть пустым")
    try:
        runtime = Runtime(settings_from_args(args))
        try:
            raw = runtime.predict({"text": args.text}, PRIORITY_QUESTIONS)
            decision = priority_policy(raw, validated=args.validated_policy)
            emit_json({"runtime": runtime.metadata(), "result": raw, "decision": decision})
        finally:
            runtime.close()
    except Exception as exc:
        print(f"priority: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
