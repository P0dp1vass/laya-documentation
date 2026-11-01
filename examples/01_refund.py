"""Выявление просьбы о возврате: не проверка права и не возврат денег.

Запуск из корня: python -m examples.01_refund --help
Пороги учебные. --validated-policy означает вашу внешнюю проверку политики,
а не измеряет качество модели. Для русского текста используйте подходящий
checkpoint; английские checkpoints не имеют здесь гарантии качества на русском.
"""

import argparse
import math
import sys

from production.calibration import apply_refund_calibration
from production.policies import refund_policy
from production.runtime import Runtime, add_runtime_arguments, emit_json, settings_from_args
from production.schemas import REFUND_QUESTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="I was charged twice. Please refund the duplicate payment.",
                        help="Текст обращения; по умолчанию синтетическая просьба вернуть двойное списание.")
    parser.add_argument("--negative-threshold", type=float, default=0.2,
                        help="Верхняя граница P(просьба=true) для отрицательного решения (default: 0.2).")
    parser.add_argument("--positive-threshold", type=float, default=0.8,
                        help="Нижняя граница P(просьба=true) для положительного решения (default: 0.8).")
    parser.add_argument("--calibration", help="Артефакт tools.calibrate fit; привязан к checkpoint, схеме и token budget.")
    parser.add_argument("--validated-policy", action="store_true",
                        help="Заявить, что политика проверена на ваших отложенных данных; иначе всегда review.")
    add_runtime_arguments(parser)
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("--text не должен быть пустым")
    if (not math.isfinite(args.negative_threshold) or not math.isfinite(args.positive_threshold)
            or not 0 <= args.negative_threshold < args.positive_threshold <= 1):
        parser.error("требуется 0 <= --negative-threshold < --positive-threshold <= 1")
    try:
        runtime = Runtime(settings_from_args(args))
        try:
            raw = runtime.predict({"text": args.text}, REFUND_QUESTIONS)
            metadata = runtime.metadata()
            policy_result, calibration_info = raw, None
            if args.calibration:
                policy_result, calibration_info = apply_refund_calibration(
                    raw, args.calibration, metadata, REFUND_QUESTIONS)
            decision = refund_policy(policy_result, args.negative_threshold, args.positive_threshold,
                                     validated=args.validated_policy)
            emit_json({"runtime": metadata, "result": raw, "decision": decision,
                       "calibration": calibration_info,
                       "calibrated_result": policy_result if args.calibration else None})
        finally:
            runtime.close()
    except Exception as exc:
        print(f"refund: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
