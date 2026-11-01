# Laya: документация и production-примеры

Русскоязычное руководство по **Laya 0.4.21** и шесть прикладных примеров: запрос на возврат, маршрутизация, модерация, приоритет, пакетная обработка и HTTP API. Для примеров предусмотрен запуск обычным Python и через Docker на CPU или NVIDIA GPU.

**Это не официальный репозиторий Laya.** Исходный проект: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya). Здесь — разбор его механизмов и отдельная прикладная обвязка. Laya возвращает типизированные решения, а не генерирует свободный текст.

> Веса в репозиторий не включены. При первом реальном предсказании нужный checkpoint скачивается с Hugging Face. Скачивание и инференс выполняйте на машине, где это разрешено. При подготовке комплекта модели не запускались; аппаратное качество и производительность нужно проверить самостоятельно.

## С чего начать

| Задача | Что открыть |
|---|---|
| Понять архитектуру, математику, API и три чекпоинта | [01_LAYA_GUIDE.md](01_LAYA_GUIDE.md) |
| Найти разбор всех 41 официального примера | [Карта примеров в общем руководстве](01_LAYA_GUIDE.md#15-карта-официальных-примеров-0141) |
| Запустить код, разобраться в порогах, API и Docker | [02_PRODUCTION.md](02_PRODUCTION.md) |
| Изменить формулировки вопросов | [production/schemas.py](production/schemas.py) |
| Изменить правила решений и ручной проверки | [production/policies.py](production/policies.py) |
| Посмотреть версии весов и границы выполненных проверок | [models.lock.json](models.lock.json), [delivery_manifest.json](delivery_manifest.json) |

Рекомендуемый порядок: установка ниже → `--help` → один пример на вашей машине → production-руководство → проверка на собственных размеченных данных. Архитектура и ограничения подробно раскрыты в общем руководстве.

## 1. Скачать репозиторий

```bash
git clone https://github.com/P0dp1vass/laya-documentation.git
cd laya-documentation
```

Все дальнейшие команды выполнять **из корня клонированного репозитория**. Если ваша локальная папка называется `laya-production-guide`, переименовывать её не требуется.

## 2. Установить окружение

Reference-окружение — **Python 3.12**. Используется отдельный `.venv`; активация не обязательна. Команды с `.venv/bin/python` подходят для Linux/macOS и не зависят от того, используете вы Bash, Zsh или Fish.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

Выберите **один** вариант установки PyTorch.

**CPU, без NVIDIA:**

```bash
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU, CUDA 12.8 wheel:**

```bash
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

GPU-вариант требует совместимый драйвер. Для MPS/XPU/другой CUDA используйте [инструкцию PyTorch](https://pytorch.org/get-started/locally/) и при необходимости адаптируйте reference constraint на torch. Одна команда не гарантирует совместимость с любым ускорителем.

После установки PyTorch:

```bash
.venv/bin/python -m pip install -r requirements.txt -c constraints-reference.txt
.venv/bin/python -c "import importlib.metadata as m; print(m.version('laya'))"
```

`constraints-reference.txt` фиксирует основные ML-зависимости, но не является полным транзитивным lock. Подробности воспроизводимости — в [production-руководстве](02_PRODUCTION.md#2-окружение-и-воспроизводимость). Не считайте системный Python 3.14 проверенной заменой reference Python 3.12.

**Windows:** создайте окружение через `py -3.12 -m venv .venv`, затем в командах замените `.venv/bin/python` на `.\.venv\Scripts\python.exe`. Docker-команды и Python module names остаются теми же.

## 3. Первый пример

Просмотр параметров **не загружает модель**:

```bash
.venv/bin/python -m examples.01_refund --help
```

Следующая команда уже выполняет реальное предсказание и при необходимости скачивает веса:

```bash
.venv/bin/python -m examples.01_refund --model multilingual \
  --text 'За март списали оплату дважды. Прошу вернуть повторное списание.'
```

Без `--device` устройство выбирает Laya. Для явного выбора добавьте `--device cpu` или `--device cuda`; CUDA указывайте только при доступном соответствующем backend.

Вывод — JSON: `runtime` с происхождением и настройками, полный `result` модели и отдельный `decision` прикладной политики. Наиболее важные поля:

- `result.routing.model` — фактически выбранный checkpoint;
- `result.answers` — ответы по IDs вопросов;
- `result.usage.truncated` — было ли потеряно содержимое из-за token budget;
- `decision.action` и `decision.reason` — что рекомендует политика и почему.

**По умолчанию `decision.action` будет `review`.** Это намеренное поведение: пороги не признаны проверенными на ваших данных. Флаг `--validated-policy` означает ваше заявление о проведённой внешней проверке — он сам не обучает и не оценивает модель. Не включайте его только ради получения другого action. Распознавание просьбы о возврате никогда не означает право на возврат или исполненную денежную операцию.

## 4. Шесть сценариев

| Модуль | Назначение | Основной механизм |
|---|---|---|
| [examples.01_refund](examples/01_refund.py) | Явная просьба вернуть деньги | `noul`, два порога, boolean/отказ от решения, optional calibration |
| [examples.02_route](examples/02_route.py) | Выбор отдела для обращения | `choice`, распределение по очередям |
| [examples.03_moderation](examples/03_moderation.py) | Несколько независимых рисков | Несколько `noul`, advisory policy |
| [examples.04_priority](examples/04_priority.py) | Срочность по operational impact | `score`, математическое ожидание и модальный класс |
| [examples.05_batch](examples/05_batch.py) | Обработка JSONL с сохранением IDs | Ограниченные chunks, батчинг, partial/complete protocol |
| [examples.06_api](examples/06_api.py) | HTTP-доступ к четырём политикам | FastAPI, startup preload, API key, один inference slot |

Запускать через `python -m`, а не копировать отдельный файл без пакета `production/`:

```bash
.venv/bin/python -m examples.02_route --model multilingual --text 'Не получается войти в приложение.'
.venv/bin/python -m examples.03_moderation --model multilingual --text 'Не согласен с ценой, но спасибо за помощь.'
.venv/bin/python -m examples.04_priority --model english --text 'Production checkout is unavailable for all customers.'
```

Параметры конкретного сценария доступны через `--help`. Общие настройки: `--model`, `--device`, `--max-len`, `--head-max-len`, `--batch-size`, `--max-loaded`.

### Выбор checkpoint

| `--model` | Когда использовать |
|---|---|
| `english` | Англоязычный `convaiinnovations/laya` |
| `multilingual` | `convaiinnovations/laya-multilingual`, в том числе русский |
| `typed-decisions` | Англоязычные специализированные workflows `convaiinnovations/laya-typed-decisions` |
| `auto` | Router выбирает по языку; это не ансамбль трёх моделей |

Точные Hub revisions записаны в [models.lock.json](models.lock.json). Один API не означает одинакового качества всех checkpoint на русском. В нашей обвязке автоматическое распознавание typed-workflow выключено; специализированную модель выбирайте явно. Длинный контекст `--max-len 8192` предназначен прежде всего для multilingual и требует отдельной проверки качества/памяти.

## 5. Пакетная обработка

```bash
mkdir -p outputs
.venv/bin/python -m examples.05_batch \
  --model multilingual --schema route \
  --input data/tickets.jsonl --output outputs/tickets-route-001.jsonl \
  --chunk-size 32 --batch-size 8
```

Вход: UTF-8 JSONL, одна запись `{"id":"ticket-001","text":"Текст обращения"}` на строку. Готовый файл: [data/tickets.jsonl](data/tickets.jsonl). Доступные схемы: `refund`, `route`, `moderation`, `priority`.

Выходной файл должен быть **новым**: существующий не перезаписывается. Успех подтверждают exit code 0 и завершающая запись `run_end` со `status=complete`; при ошибке файл может остаться partial. Для нового запуска укажите новое имя. Подробный формат и память — в [разделе batch](02_PRODUCTION.md#9-пример-05-пакетный-jsonl).

## 6. HTTP API

Сначала задайте собственный секрет через окружение.

**Bash/Zsh:**

```bash
export LAYA_API_KEY="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

**Fish:**

```fish
set -gx LAYA_API_KEY (.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')
```

Запуск на выбранной машине:

```bash
.venv/bin/python -m examples.06_api --model multilingual --host 127.0.0.1 --port 8000
```

Для явного выбора GPU добавьте `--device cuda`. Startup загружает веса; при `--model auto` preload включает все три checkpoint и требует больше памяти. Готовность endpoint не означает проверенное качество модели.

В другом терминале задайте **то же значение** `LAYA_API_KEY` (не генерируйте другое), затем:

```bash
curl --fail-with-body http://127.0.0.1:8000/readyz
curl --fail-with-body http://127.0.0.1:8000/v1/decide \
  -H "X-API-Key: $LAYA_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @data/api_request.json
```

Тело запроса: `text`, `scenario` (`refund`, `route`, `moderation`, `priority`), optional `request_id`. Живой процесс — `/healthz`, readiness — `/readyz`. Неверный ключ даёт 401, неверное тело — 422, занятый inference slot — 429, недоступность инференса — 503. Без отдельно проверенных политик решения остаются review. API key — минимальная защита примера; перед внешней публикацией нужны TLS, ограничения размера/частоты запросов и подходящая авторизация. Подробности — в [разделе API](02_PRODUCTION.md#10-пример-06-http-api).

## 7. Docker

Задайте `LAYA_API_KEY` способом для вашей оболочки из предыдущего раздела.

**CPU:**

```bash
docker compose up --build
```

**NVIDIA GPU:**

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

GPU-вариант требует совместимый драйвер, NVIDIA Container Toolkit и Compose с поддержкой `gpus: all`. Первый запуск загружает checkpoint в cache volume. Порт опубликован только на `127.0.0.1:8000`; сервис запускается non-root. Эти команды не являются отчётом о выполненной здесь Docker/GPU-проверке. Офлайн-контейнеры и обычные CLI внутри образа — в [разделе Docker](02_PRODUCTION.md#12-docker-cpu-и-nvidia-gpu).

## 8. Офлайн, калибровка и адаптация

- **Офлайн:** на разрешённой машине заранее выполните `.venv/bin/python -m tools.download_models --model all --directory models`. Перенесите snapshots, задайте локальные пути и offline-переменные по [инструкции](02_PRODUCTION.md#3-все-чекпоинты-hub-и-офлайн). Пустой cache сам по себе офлайн-запуск не обеспечивает.
- **Своя задача:** измените критерии в [production/schemas.py](production/schemas.py), затем проверьте их на размеченной выборке. Изменение текста/порядка вариантов может менять результат и делает старую калибровку непереносимой.
- **Калибровка:** полный процесс `tools.calibrate collect → fit → evaluate` и применение в примере 01 — в [разделе калибровки](02_PRODUCTION.md#11-калибровка-без-выдуманного-logits-api). `collect` запускает модель; `fit/evaluate` работают с уже собранными probabilities. Учебные datasets не заменяют production validation, а формат нашего artifact не совместим с `agent.load_calibration`.
- **Надёжность:** confidence не гарантирует правильность, temperature scaling не исправляет неверный класс, а усечение текста нельзя игнорировать. Нужны собственные evaluation, пороги и допустимая доля ручной проверки.

## 9. Проверки без запуска моделей

Тесты чистых политик и численной калибровки используют стандартную библиотеку; для этого не нужно скачивать веса или устанавливать полный ML-стек:

```bash
python3 -B -m unittest discover -s tests -v
```

При подготовке прошли 18 model-free tests, CLI help, проверки HTTP-контракта до инференса, численный calibration smoke и route-only настоящего SDK. Подробная область проверки и список **не выполненных** model/GPU/Docker-сценариев сохранены в [delivery_manifest.json](delivery_manifest.json). Для аппаратной приёмки используйте [чек-лист](02_PRODUCTION.md#13-приёмка-на-своей-машине).

## Что хранить в Git

Документы, исходный код, синтетические `data/`, зависимости и Docker-файлы — в репозитории. `.gitignore` исключает `.venv/`, `.env`, `models/`, `outputs/` и Python-кеш. **Не коммитьте API/Hugging Face токены, реальные обращения клиентов и веса.** `.env.example` — только шаблон, не действующий секрет.

Политики примеров не выполняют денежные операции, удаление контента и отправку обращений внешним сервисам. Подключение таких действий требует отдельных полномочий, бизнес-проверок, аудита и идемпотентности. Это учебный production-oriented комплект, а не сертификат готовности вашей бизнес-задачи.
