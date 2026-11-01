"""Advisory-модерация двумя noul-вопросами: не защитная граница и не удаление.

Запуск из корня: python -m examples.03_moderation --help
Учебные пороги: P(true) <= 0.1 / >= 0.9, answer_confidence >= 0.9 для обоих
ответов. Неопределённость, усечение и непроверенная политика дают review.
Список рисков не исчерпывает вредный контент. Русские тексты требуют отдельной
проверки качества; английские checkpoints не становятся многоязычными от флага CLI.
"""

import argparse
import sys

from production.policies import moderation_policy
from production.runtime import Runtime, add_runtime_arguments, emit_json, settings_from_args
from production.schemas import MODERATION_QUESTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="I disagree with the product decision. Please explain the change respectfully.",
                        help="Текст для advisory-модерации; по умолчанию синтетическая вежливая критика.")
    parser.add_argument("--validated-policy", action="store_true",
                        help="Заявить внешнюю проверку политики; флаг не измеряет качество. Иначе review.")
    add_runtime_arguments(parser)
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("--text не должен быть пустым")
    try:
        runtime = Runtime(settings_from_args(args))
        try:
            raw = runtime.predict({"text": args.text}, MODERATION_QUESTIONS)
            decision = moderation_policy(raw, validated=args.validated_policy)
            emit_json({"runtime": runtime.metadata(), "result": raw, "decision": decision})
        finally:
            runtime.close()
    except Exception as exc:
        print(f"moderation: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
