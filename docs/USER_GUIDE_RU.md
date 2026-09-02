# Руководство пользователя: AADD Attack Pipeline

> **Снимок состояния.** Этот документ отражает проект по состоянию на 2 сентября
> 2026 и с тех пор разошёлся с кодом. Актуальны:
> `specs/attacks/*.yaml` — что за атаки и с какими параметрами;
> `CAMPAIGN.yaml` — порядок задач;
> `docs/OPERATOR_GPT_RU.md` — как запускать и следить.
>
> Что здесь заведомо неверно:
> набор атак изменён (`mifgsm`, `ti-di-mi-fgsm`, `frequency-eot`, `dd-fcma`
> больше не планируются; вместо них `mi-di-fgsm`, `ssa`, `ensemble-mi-eot`);
> задач в кампании 27, а не 21, потому что каждая атака идёт в двух направлениях;
> планировщик на этой машине требует флаг `--allow-shared-gpu`, иначе он молча
> не возьмёт ни одну карту.
>
> Проверить фактическое состояние:
> `python3 -m attacklab.cli validate-specs --config configs/pipeline/server.yaml`
> и `python3 scripts/run_campaign.py development --dry-run`.


> Подробная инструкция по использованию системы оркестрации AI-агентов для совместного запуска adversarial атак.

---

## Оглавление

1. [Что это такое](#1-что-это-такое)
2. [Архитектура системы](#2-архитектура-системы)
3. [Быстрый старт](#3-быстрый-старт)
4. [Пошаговая инструкция](#4-пошаговая-инструкция)
5. [Команды и взаимодействие](#5-команды-и-взаимодействие)
6. [Мониторинг и управление](#6-мониторинг-и-управление)
7. [Восстановление после сбоев](#7-восстановление-после-сбоев)
8. [Безопасность и ограничения](#8-безопасность-и-ограничения)

---

## 1. Что это такое

Эта система — **оркестратор adversarial-атак** на AI-детекторы deepfake-изображений. Она автоматизирует полный цикл исследования:

- реализацию атаки (через LLM-агента);
- тестирование на CPU;
- постановку в очередь GPU-экспериментов;
- детерминированную верификацию результата;
- научный review с применением замороженных критериев.

**Ключевая идея:** LLM-агент подготавливает эксперимент, но не является источником истины. Источники истины — committed config, замороженный job spec, checkpoint hashes, machine-generated артефакты, post-save norm audit, детерминированный верификатор и независимый read-only review.

### Научная задача

Система реализует план исследования из `research_experiment_plan.md`: сравнение 10 targeted adversarial атак (FGSM, I-FGSM, PGD, MI-FGSM, DI-MI-FGSM, TI-DI-MI-FGSM, Frequency-EOT, MIG-COW, Prototype, DD-FCMA) против двух AI-детекторов (ViT-B/16 spatial и DenseNet-121 DCT) на датасете CelebA.

---

## 2. Архитектура системы

Система состоит из **четырёх независимых уровней**:

```
┌─────────────────────────────────────────────────────────────┐
│                   OpenCode Control Plane                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ coordinator  │  │attack-worker │  │campaign-reviewer  │  │
│  │              │  │              │  │   (read-only)     │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────────────┘  │
│         │                  │                                  │
│         │    /campaign     │  /attack                         │
│         │    /campaign     │  /attack                         │
│         │    /campaign-    │                                  │
│         │    review        │                                  │
└─────────┼──────────────────┼──────────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────┐  ┌──────────────────────────────────────┐
│ Campaign        │  │         GPUQ (GPU Queue)              │
│ Controller      │  │  SQLite ──► scheduler ──► runner     │
│ (run_campaign)  │  │                │               │      │
│                 │  │                ▼               ▼      │
│ reconciles      │  │         attacklab.cli    verifier     │
│ queue state     │  │              run                      │
└─────────────────┘  └──────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────┐
│                   Tracking Ledger (Git)                    │
│  configs/  manifests/  tracking/  schemas/  tests/       │
└──────────────────────────────────────────────────────────┘
```

### Роли агентов

| Агент | Роль | Может | Не может |
|---|---|---|---|
| **coordinator** | Управление кампанией | Запускать `run_campaign.py`, читать состояние, опрашивать GPUQ | Редактировать код, запускать GPU, scheduler |
| **attack-worker** | Реализация одной атаки | Писать код атаки, тесты, конфиги, submit в GPUQ | Выбирать GPU, запускать runner/verifier, commit/push |
| **campaign-reviewer** | Научный review | Читать evidence, применять frozen gates, вернуть JSON | Редактировать что-либо, запускать процессы |

### Жизненный цикл задачи

```
planned → agent_running → queued → running → validating → passed
                                    \→ failed | blocked | cancelled
passed → needs_review → retain | reject | baseline
```

---

## 3. Быстрый старт

### Предварительные требования

```text
Python:         3.12.3
PyTorch:        2.3.0+cu121
NVIDIA driver:  >= 525.60
OpenCode:       1.18.26
Модель:         naapi/gpt-5.6-luna
```

### Серверные пути

```text
/home/aiattacks/oleg/aadd-attack-pipeline/     Git checkout (проект)
/home/aiattacks/dataset/celebA/                 Read-only датасет
/home/aiattacks/oleg/aadd-attack-assets/weights/ Веса детекторов
/home/aiattacks/oleg/aadd-attack-runs/          Генерируемые изображения
/home/aiattacks/.opencode/bin/opencode          Бинарник OpenCode
```

### Установка (выполняет человек, не агент)

```bash
# 1. Клонировать ветку на сервер
git clone --branch codex/aadd-agent-pipeline --single-branch \
  https://github.com/sangwyn/deepfake-attacks.git \
  /home/aiattacks/oleg/aadd-attack-pipeline

cd /home/aiattacks/oleg/aadd-attack-pipeline

# 2. Подготовить веса детекторов
bash scripts/prepare_server_assets.sh

# 3. Создать Python-окружение
bash scripts/bootstrap_environment.sh

# 4. Строгая проверка (preflight)
.venv/bin/python scripts/preflight.py --strict --output tracking/preflight.json

# 5. Создать manifests датасета
.venv/bin/python -m attacklab.cli build-manifests \
  --config configs/pipeline/server.yaml \
  --output-dir manifests/celebA

# 6. Проверить очередь GPU
.venv/bin/python -m ops.gpuq doctor --json

# 7. Проверить OpenCode
/home/aiattacks/.opencode/bin/opencode --version   # должно быть 1.18.26
/home/aiattacks/.opencode/bin/opencode models       # должен содержать naapi/gpt-5.6-luna
```

---

## 4. Пошаговая инструкция

### 4.1. Первый smoke-эксперимент (ручной)

Это минимальный безопасный сценарий для проверки, что всё работает end-to-end:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline

# Предварительный просмотр scheduler (безопасный, ничего не запускает)
.venv/bin/python -m ops.gpuq run-scheduler --once

# Поставить job в очередь
.venv/bin/python -m ops.gpuq submit \
  tracking/jobs/ifgsm-smoke/job-spec.json

# Запустить scheduler в tmux
tmux new -s aadd-gpuq
.venv/bin/python -m ops.gpuq run-scheduler \
  --execute \
  --max-running 1 \
  --poll-seconds 10 \
  --idle-samples 3 \
  --headroom-mb 4096

# Отключиться от tmux: Ctrl-b d

# Из другого терминала — мониторинг
.venv/bin/python -m ops.gpuq list --json
.venv/bin/python -m ops.gpuq status <JOB_ID> --json
```

После `succeeded`:

```bash
# Верификация результата
.venv/bin/python -m attacklab.cli verify \
  --run-dir tracking/runs/<campaign>/<task>/attempt-0001

# Review и commit артефактов (делает человек)
git status --short
git add configs/experiments manifests tracking
git commit -m "experiment: record verified IFGSM smoke run"
```

### 4.2. Запуск автоматической кампании

После успешного smoke и закрытия readiness checklist:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline

# Запустить development campaign (последовательно, одна атака за раз)
.venv/bin/python scripts/run_campaign.py development
```

**Что происходит:**

1. Контроллер читает `CAMPAIGN.yaml` и строит граф зависимостей.
2. Для каждой задачи запускается отдельный процесс OpenCode с агентом `attack-worker`.
3. Worker реализует атаку, пишет тесты, создаёт конфиг и job spec, submit-ит в GPUQ.
4. Worker выходит. Контроллер опрашивает GPUQ до терминального состояния.
5. Читает `verification.json`, обновляет status, переходит к следующей задаче.

### 4.3. Интерактивный режим через OpenCode

Для ручного управления через координатора:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline
/home/aiattacks/.opencode/bin/opencode \
  --agent coordinator \
  --model naapi/gpt-5.6-luna \
  .
```

Внутри OpenCode доступны команды:

```text
/campaign status          — показать текущее состояние кампании
/campaign development     — запустить development campaign
/campaign resume          — продолжить прерванную кампанию
/campaign resume --retry <task-id>  — повторить упавшую задачу
/campaign full            — запустить full run (требует явной авторизации)
```

Для подготовки одной атаки вручную (в отдельном процессе):

```bash
/home/aiattacks/.opencode/bin/opencode \
  --agent attack-worker \
  --model naapi/gpt-5.6-luna \
  .
```

Затем внутри:

```text
/attack fgsm smoke
/attack pgd development
/attack mifgsm smoke
```

Формат: `/attack <имя-атаки> <smoke|development|full> [status-file] [task-id]`

### 4.4. Non-interactive запуск (используется контроллером)

```bash
/home/aiattacks/.opencode/bin/opencode run \
  --command attack \
  --dir /home/aiattacks/oleg/aadd-attack-pipeline \
  --title '<campaign-id>:<task-id>' \
  '<attack>' '<scope>' '<status-file>' '<task-id>'
```

Dry-run для проверки:

```bash
.venv/bin/python scripts/run_campaign.py development --dry-run
```

---

## 5. Команды и взаимодействие

### 5.1. Карта всех команд

#### Окружение и preflight

```bash
bash scripts/prepare_server_assets.sh          # скопировать и проверить веса
bash scripts/bootstrap_environment.sh          # создать .venv
.venv/bin/python scripts/preflight.py --strict  # строгая проверка
```

#### Manifests

```bash
.venv/bin/python -m attacklab.cli build-manifests \
  --config configs/pipeline/server.yaml \
  --output-dir manifests/celebA
```

#### GPU очередь (GPUQ)

```bash
.venv/bin/python -m ops.gpuq doctor --json                    # диагностика
.venv/bin/python -m ops.gpuq submit JOB_SPEC.json             # поставить job
.venv/bin/python -m ops.gpuq list --json                      # список jobs
.venv/bin/python -m ops.gpuq status JOB_ID --json             # статус job
.venv/bin/python -m ops.gpuq cancel JOB_ID                    # отменить job
.venv/bin/python -m ops.gpuq run-scheduler --once             # preview
.venv/bin/python -m ops.gpuq run-scheduler --execute          # реальный запуск
```

#### Scientific runner

```bash
.venv/bin/python -m attacklab.cli run --config CONFIG --run-dir RUN_DIR
.venv/bin/python -m attacklab.cli verify --run-dir RUN_DIR
.venv/bin/python -m attacklab.cli validate-status --kind attack --status-file STATUS
.venv/bin/python -m attacklab.cli validate-status --kind review --status-file STATUS
```

#### OpenCode

```bash
/home/aiattacks/.opencode/bin/opencode --version
/home/aiattacks/.opencode/bin/opencode models
/home/aiattacks/.opencode/bin/opencode debug config
/home/aiattacks/.opencode/bin/opencode debug agent coordinator
/home/aiattacks/.opencode/bin/opencode debug agent attack-worker
/home/aiattacks/.opencode/bin/opencode debug agent campaign-reviewer
/home/aiattacks/.opencode/bin/opencode debug skill
```

### 5.2. Граф зависимостей атак (из CAMPAIGN.yaml)

Атаки запускаются строго в порядке зависимостей:

```
ifgsm-smoke
  ├─► ifgsm-development
  │     ├─► pgd-smoke ─► pgd-development
  │     └─► mifgsm-smoke ─► mifgsm-development
  │                               └─► di-mi-fgsm-smoke ─► di-mi-fgsm-development
  │                                                             └─► ti-di-mi-fgsm-smoke ─► ti-di-mi-fgsm-development
  │                                                                       └─► frequency-eot-smoke ─► frequency-eot-development
  │                                                                                                       │
  ├─► fgsm-smoke ─► fgsm-development                                                                       │
                                                                                                           │
frequency-eot-development ─► dd-fcma-smoke ─► dd-fcma-development ─► select-finalists
mig-cow-smoke ─► mig-cow-development ──────────────────────────────┘
prototype-smoke ─► prototype-development ──────────────────────────┘
```

### 5.3. Job spec формат

GPUQ принимает только schema-v1 JSON:

```json
{
  "schema_version": 1,
  "task_kind": "attack-experiment",
  "config_path": "configs/experiments/ifgsm-smoke.yaml",
  "run_dir": "tracking/runs/<campaign>/<task>",
  "requested_memory_mb": 24000,
  "timeout_seconds": 7200,
  "priority": 0,
  "max_attempts": 1
}
```

**Запрещено:** произвольные команды, environment variables, GPU index/UUID, абсолютные пути, `..`, секреты.

---

## 6. Мониторинг и управление

### 6.1. Проверка статуса кампании

```bash
# Через контроллер
.venv/bin/python scripts/run_campaign.py status

# Через OpenCode
/campaign status

# Через GPUQ
.venv/bin/python -m ops.gpuq list --json
.venv/bin/python -m ops.gpuq status <JOB_ID> --json
```

### 6.2. Состояния GPUQ

```text
queued ──► reserving ──► running ──► validating ──► succeeded
  │             │            │             │
  └─► cancelled ├─► failed   ├─► failed    ├─► failed
                ├─► retry_wait             ├─► retry_wait
                └─► orphaned               └─► orphaned
```

### 6.3. Критерии научного статуса

| Статус задачи | Значение |
|---|---|
| `queued` | Worker поставил job в очередь и вышел |
| `running` | Контроллер видит, что scheduler выполняет job |
| `passed` | Верификатор одобрил, контроллер принял (требует `verification.json`) |
| `failed` | Runner/verifier упал или верификация не пройдена |
| `blocked` | Отсутствует prerequisite (манифест, чекпоинт, API) |
| `cancelled` | Job отменён оператором |

| Решение (decision) | Значение |
|---|---|
| `pending` | Ещё не прошёл review |
| `baseline` | Базовый метод, не кандидат на удержание |
| `retain` | Прошёл frozen gate, удержан для следующих фаз |
| `reject` | Технически корректен, но не показал нужного улучшения |
| `not_applicable` | Не применим (опциональная фаза пропущена) |

### 6.4. Критерии удержания (из OPENCODE_LUNA.md)

| Гипотеза | Критерий retain |
|---|---|
| MI-FGSM | >=5pp cross-model ASR или >=3% quality-score без ухудшения SSIM/LPIPS >0.01 |
| DI | >=5pp mean transfer, paired 95% CI исключает 0 |
| TI | Дополнительно >=3pp или >=3% score |
| Frequency EOT | >=5pp на худшем transfer-направлении или >=5% official score |
| COW | >=5pp median held-out, ни один target не теряет >2pp |
| Prototype | >=3pp или >=3% score без quality regression |

### 6.5. Файлы статуса

- **`PHASE_STATUS.md`** — человеческий ledger, обновляется только из verifier-approved артефактов.
- **`.campaign/runs/<id>/state.json`** — машинное состояние кампании (контроллер).
- **`tracking/runs/<campaign>/<task>/status.json`** — статус конкретной задачи.
- **`tracking/runs/<campaign>/<task>/status.worker.json`** — оригинальный провизорный статус от worker.

---

## 7. Восстановление после сбоев

### 7.1. Остановился OpenCode worker

GPU job **не** отменён. Проверить:

```bash
.venv/bin/python -m ops.gpuq status <JOB_ID> --json
```

### 7.2. Остановился контроллер (`run_campaign.py`)

Job продолжает выполняться scheduler-ом. При следующем запуске контроллер выполнит реконсиляцию:

```bash
.venv/bin/python scripts/run_campaign.py resume
```

Реконсиляция идемпотентна: она перечитает GPUQ и verifier report.

### 7.3. Остановился scheduler

1. Не запускать второй scheduler сразу.
2. Проверить процессы: `.venv/bin/python -m ops.gpuq doctor --json`
3. Проверить состояния `running`, `validating`, `orphaned`.
4. Только после reconciliation перезапустить scheduler.

### 7.4. Job failed

```bash
.venv/bin/python -m ops.gpuq status <JOB_ID> --json
```

Проверить attempt directory:
- `failure.json` — причина сбоя
- `verification.json` — вердикт верификатора
- `summary.json` — агрегированные результаты

**Не перезаписывать attempt.** Retry создаёт `attempt-0002`.

Через контроллер:

```bash
.venv/bin/python scripts/run_campaign.py resume --retry <task-id>
```

### 7.5. Ctrl+C во время опроса

**Не отменяет GPU job.** Задачей владеет scheduler. Задача остаётся `running`, и следующий `resume` её подберёт.

### 7.6. Отмена job

```bash
.venv/bin/python -m ops.gpuq cancel <JOB_ID>
```

---

## 8. Безопасность и ограничения

### 8.1. Что агенты НЕ могут делать

| Ограничение | Почему |
|---|---|
| Выбирать GPU / ставить `CUDA_VISIBLE_DEVICES` | GPUQ — единственный арбитр ресурсов |
| Запускать `attacklab.cli run` напрямую | Обходит resource arbitration |
| Запускать scheduler | Только один scheduler на проект |
| Коммитить / пушить | Commit делает человек после review |
| Устанавливать пакеты / скачивать | Окружение заморожено |
| Редактировать evaluator / manifests / checkpoints | Защищённые файлы |
| Вызывать другого агента | Concurrency контролируется снаружи |
| Использовать `--auto` | Permission prompt = config error |

### 8.2. Защищённые файлы (нельзя менять из attack-worker)

- `evaluate.py`, `AADD-2026/AADD_2026_evaluation.py`
- Детекторные preprocessing/adapters
- Замороженные manifests и checkpoints
- Metric definitions
- Completed run artifacts
- Campaign manifest (`CAMPAIGN.yaml`)
- Queue implementation и state
- Другие attacks

### 8.3. OpenCode permissions — не sandbox

OpenCode permissions снижают риск случайных действий, но **не являются OS-level sandbox**. Для production:

- Отдельный Git worktree на каждый worker
- Read-only mount датасета и checkpoints
- Нет SSH/push/package-manager credentials в worker
- Только scheduler имеет доступ к GPU devices

### 8.4. Секреты

- API-ключи существуют **только** во внешней OpenCode config пользователя.
- Никогда не копируются в repository, job spec, log или status.
- Worker не получает SSH/push credentials.

---

## Приложение A: Структура каталогов проекта

```
aadd-attack-pipeline/
├── .opencode/
│   ├── agents/              # Определения агентов
│   │   ├── coordinator.md
│   │   ├── attack-worker.md
│   │   └── campaign-reviewer.md
│   ├── commands/            # Команды OpenCode
│   │   ├── attack.md
│   │   ├── campaign.md
│   │   └── campaign-review.md
│   └── skills/
│       └── attack-execution/  # Skill для worker
│           ├── SKILL.md
│           └── references/
├── .gpuq/                   # Runtime: SQLite, locks, logs (не в Git)
├── attacklab/               # Scientific layer
│   ├── preprocessing.py     # Общий дифференцируемый preprocessing
│   ├── runner.py            # Экспериментальный runner
│   ├── cli.py               # CLI: run, verify, preflight, ...
│   ├── config.py            # Валидация конфигов
│   ├── manifest.py          # Manifest builder
│   └── ...
├── attacks/                 # Attack plug-ins
│   ├── ifgsm.py             # Реализована
│   ├── fgsm.py              # Реализована
│   ├── template.py          # Шаблон для новых
│   └── ...                  # Остальные создаёт attack-worker
├── configs/
│   ├── pipeline/server.yaml # Server config
│   └── experiments/         # Experiment configs
├── manifests/celebA/        # Dataset manifests
├── ops/gpuq/                # GPU очередь
├── schemas/                 # JSON schemas
├── scripts/
│   ├── run_campaign.py      # Campaign controller
│   ├── preflight.py         # Preflight checker
│   ├── bootstrap_environment.sh
│   └── prepare_server_assets.sh
├── tests/                   # Unit/integration tests
├── tracking/                # Experiment ledger (в Git)
│   ├── jobs/                # Job specs
│   └── runs/                # Run artifacts
├── docs/                    # Документация
├── AGENTS.md                # Agent rules (читается каждым агентом)
├── OPENCODE_LUNA.md         # Attack specifications
├── CAMPAIGN.yaml            # Campaign graph
├── PHASE_STATUS.md          # Human-readable status ledger
├── research_experiment_plan.md  # Полный научный план
└── opencode.jsonc           # OpenCode project config
```

## Приложение B: Полезные ссылки внутри проекта

| Файл | Назначение |
|---|---|
| `AGENTS.md` | Правила для всех агентов |
| `OPENCODE_LUNA.md` | Спецификация атак и критерии удержания |
| `CAMPAIGN.yaml` | Граф задач и зависимостей |
| `PHASE_STATUS.md` | Текущий статус всех атак |
| `research_experiment_plan.md` | Полный научный план (15 фаз) |
| `docs/RUNBOOK_RU.md` | Детальный runbook (1800+ строк) |
| `docs/AGENT_PIPELINE.md` | Контракт agent/GPU execution pipeline |
| `docs/SCIENTIFIC_PIPELINE.md` | Scientific pipeline contract |
| `AGENT_SETUP.md` | Инструкция по настройке control plane |
| `ops/gpuq/README.md` | Документация GPUQ |
