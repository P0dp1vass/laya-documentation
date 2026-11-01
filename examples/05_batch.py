"""Потоковый JSONL batch с явной полнотой, ID, схемой и provenance.

Запуск из корня: python -m examples.05_batch --output /tmp/tickets-output.jsonl
Вход: непустой JSON-объект {"id": "уникальная строка", "text": "текст"} на
каждой строке; пустые строки, повторные JSON-ключи и ID считаются ошибками.
Память: O(размер chunk + максимальная строка + N уникальных ID), не O(1).
Runtime может сортировать внутри batch, но его контракт возвращает исходный
порядок; этот пример сохраняет порядок входа. Выход создаётся только через x.

Выход — JSONL-протокол: run_start(status=partial), prediction*, run_end.
Успехом является ТОЛЬКО последний run_end(status=complete) и код выхода 0.
При ошибке код ненулевой, существующий частичный файл сохраняется; по возможности
добавляется run_end(status=partial). При аварийном завершении/заполнении диска
конечного маркера может не быть — такой файл тоже неполный. Для повтора задайте
новое имя файла. Автовозобновления и перезаписи нет. Модели не гарантируют
правильность решений, в том числе на русском; --validated-policy не измеряет её.
"""

import argparse
import json
import sys
from pathlib import Path

from production.policies import moderation_policy, priority_policy, refund_policy, route_policy
from production.runtime import Runtime, add_runtime_arguments, emit_json, settings_from_args
from production.schemas import MODERATION_QUESTIONS, PRIORITY_QUESTIONS, REFUND_QUESTIONS, ROUTE_QUESTIONS


SCHEMAS = {
    "refund": (REFUND_QUESTIONS, refund_policy),
    "route": (ROUTE_QUESTIONS, route_policy),
    "moderation": (MODERATION_QUESTIONS, moderation_policy),
    "priority": (PRIORITY_QUESTIONS, priority_policy),
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"повторный JSON-ключ: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"недопустимая JSON-константа: {value}")


def _chunks(source, chunk_size):
    seen = set()
    chunk = []
    for line_number, line in enumerate(source, 1):
        try:
            row = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
            if not isinstance(row, dict):
                raise ValueError("ожидается JSON-объект")
            identifier, text = row.get("id"), row.get("text")
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError("id должен быть непустой строкой")
            if identifier in seen:
                raise ValueError(f"повторный id: {identifier}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text должен быть непустой строкой")
        except ValueError as exc:
            raise ValueError(f"строка {line_number}: {exc}") from exc
        seen.add(identifier)
        chunk.append((line_number, identifier, text))
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
    if not seen:
        raise ValueError("вход не содержит записей")


def _write_record(destination, value):
    destination.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/tickets.jsonl"),
                        help="Входной UTF-8 JSONL (default: data/tickets.jsonl).")
    parser.add_argument("--output", type=Path, required=True,
                        help="НОВЫЙ выходной JSONL; существующий файл никогда не перезаписывается.")
    parser.add_argument("--chunk-size", type=int, default=32,
                        help="Максимальное число текстов в памяти chunk (default: 32).")
    parser.add_argument("--schema", choices=tuple(SCHEMAS), default="route",
                        help="Общая схема и политика для всех записей (default: route).")
    parser.add_argument("--validated-policy", action="store_true",
                        help="Заявить внешнюю проверку политики; без флага все решения review.")
    add_runtime_arguments(parser)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size должен быть >= 1")
    questions, policy = SCHEMAS[args.schema]
    schema = {"name": args.schema, "version": 1}
    written = 0
    try:
        # Проверяем доступность входа и создаём выход до загрузки модели.
        with args.input.open("r", encoding="utf-8") as source, args.output.open("x", encoding="utf-8") as destination:
            try:
                _write_record(destination, {"record_type": "run_start", "status": "partial",
                                            "schema": schema, "questions": questions,
                                            "input": str(args.input), "policy_validated": args.validated_policy})
                destination.flush()
                runtime = Runtime(settings_from_args(args))
                try:
                    for chunk in _chunks(source, args.chunk_size):
                        states = [{"text": text} for _, _, text in chunk]
                        results = runtime.predict_batch(states, questions)
                        if not isinstance(results, list) or len(results) != len(chunk):
                            raise ValueError("predict_batch вернул неверное число результатов или не список")
                        provenance = runtime.metadata()
                        for (line_number, identifier, _), raw in zip(chunk, results):
                            decision = policy(raw, validated=args.validated_policy)
                            if decision["reason"].startswith("invalid_result:"):
                                raise ValueError(f"строка {line_number}, id={identifier}: {decision['reason']}")
                            _write_record(destination, {"record_type": "prediction", "id": identifier,
                                                        "input_line": line_number, "schema": schema,
                                                        "runtime": provenance, "result": raw, "decision": decision})
                            written += 1
                        destination.flush()
                finally:
                    runtime.close()
                _write_record(destination, {"record_type": "run_end", "status": "complete", "count": written})
                destination.flush()
            except (Exception, KeyboardInterrupt) as exc:
                error = f"{type(exc).__name__}: {exc}"
                try:
                    _write_record(destination, {"record_type": "run_end", "status": "partial",
                                                "count": written, "error": error})
                    destination.flush()
                except (OSError, ValueError) as marker_error:
                    print(f"batch: не удалось записать partial-маркер: {marker_error}", file=sys.stderr)
                print(f"batch: {error}; неполный файл сохранён: {args.output}; записей: {written}", file=sys.stderr)
                return 130 if isinstance(exc, KeyboardInterrupt) else 1
        emit_json({"status": "complete", "output": str(args.output), "count": written, "schema": schema})
    except (Exception, KeyboardInterrupt) as exc:
        print(f"batch: {type(exc).__name__}: {exc}; завершение не подтверждено", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
