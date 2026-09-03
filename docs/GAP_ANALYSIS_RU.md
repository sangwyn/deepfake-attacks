# Аудит проекта: что реализовано и что нужно добавить

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


> Подробный анализ текущего состояния AADD Attack Pipeline с классификацией компонентов по степени готовности и описанием недостающих элементов.

---

## Оглавление

1. [Сводная таблица готовности](#1-сводная-таблица-готовности)
2. [Что полностью реализовано](#2-что-полностью-реализовано)
3. [Что частично реализовано](#3-что-частично-реализовано)
4. [Что отсутствует или заблокировано](#4-что-отсутствует-или-заблокировано)
5. [Рекомендации по развитию](#5-рекомендации-по-развитию)

---

## 1. Сводная таблица готовности

| Компонент | Статус | Готовность |
|---|---|---|
| OpenCode control plane (агенты, команды, skill) | Полностью | ~95% |
| Campaign controller (`run_campaign.py`) | Полностью | ~90% |
| GPUQ (очередь, scheduler, CLI) | Полностью | ~90% |
| Scientific layer (`attacklab/`) | Полностью | ~85% |
| Attack plug-ins | Минимально | ~20% |
| Experiment configs | Минимально | ~15% |
| Test coverage | Базовая | ~60% |
| Documentation | Хорошая | ~85% |
| Manifests датасета | Созданы | ~80% |
| Environment lock | Зафиксирован | ~90% |
| Real GPU smoke | Не запущен | 0% |
| Development campaign | Не запущена | 0% |
| Full campaign | Не авторизована | 0% |
| Parallel workers | Не реализованы | 0% |
| Artifact store | Нет политики | 0% |

---

## 2. Что полностью реализовано

### 2.1. OpenCode Control Plane (~95%)

**Агенты (`.opencode/agents/`):**
- `coordinator.md` — координатор кампании: read-only, запускает `run_campaign.py`, опрашивает GPUQ.
- `attack-worker.md` — реализатор атак: edit, CPU-тесты, submit в GPUQ. Имеет строгий bash allowlist.
- `campaign-reviewer.md` — read-only reviewer: без bash, без edit, только чтение evidence.

Все три агента настроены с:
- Моделью `naapi/gpt-5.6-terra`
- `task: false` (нет subagents)
- `webfetch: deny`, `websearch: deny`
- `external_directory: deny`
- Fail-closed bash policy

**Команды (`.opencode/commands/`):**
- `attack.md` — `/attack <name> <scope>` — подготовка одной атаки
- `campaign.md` — `/campaign <action>` — управление кампанией
- `campaign-review.md` — `/campaign-review` — научный review

**Skill (`.opencode/skills/attack-execution/`):**
- `SKILL.md` — полный workflow из 5 шагов (preflight, implementation, CPU checks, freeze & enqueue, status)
- `references/status-contract.md` — JSON schema статуса задачи

**Конфигурация:**
- `opencode.jsonc` — project config с fail-closed permissions
- Built-in agents (build, plan, general, explore) отключены
- `task` tool отключён глобально

### 2.2. Campaign Controller (~90%)

`scripts/run_campaign.py` (1238 строк) — полноценный контроллер:

- Парсит `CAMPAIGN.yaml` и строит DAG задач
- Запускает внешние OpenCode процессы для каждой задачи
- Поддерживает состояния: `planned`, `agent_running`, `queued`, `running`, `validating`, `passed`, `failed`, `blocked`, `cancelled`
- Реконсиляция с GPUQ (идемпотентная, запускается при `resume` и `status`)
- Валидация status JSON (worker не может написать `passed`)
- Атомарная запись статусов
- Поддержка `--dry-run`
- Поддержка `--poll-timeout-seconds`
- Обработка review-задач (`select-finalists`)
- Построение full-задач из finalists

`CAMPAIGN.yaml`:
- 21 задача в development profile (10 smoke + 10 development + 1 review)
- Full profile с `max_finalists: 2`
- Корректный DAG с `needs` и `after` зависимостями

### 2.3. GPUQ — GPU Queue (~90%)

`ops/gpuq/` — полноценная GPU-очередь:

- **SQLite** хранение jobs и transitions
- **Singleton lock** — не более одного scheduler на проект
- **GPU selection по UUID** (не по индексу)
- **Idle sampling** — GPU должна быть idle N наблюдений подряд
- **Memory headroom** — requested + 4 GiB
- **Cooperative flock** перед dispatch
- **Idempotency** — повторная submit идентичного spec возвращает существующий job_id
- **Fixed argv** — нет произвольных команд
- **Schema validation** через `job-spec.schema.json`
- **Scheduler state machine**: queued → reserving → running → validating → succeeded
- **Process supervision**: shell=False, new session, timeout, captured logs
- **Orphan detection**: после crash scheduler не убивает существующие PID

CLI:
- `submit`, `list`, `status`, `cancel`, `doctor`, `run-scheduler`
- Preview mode по умолчанию (безопасный)
- `--execute` для реального запуска

### 2.4. Scientific Layer — `attacklab/` (~85%)

Модули:
- `preprocessing.py` — общий дифференцируемый preprocessing (spatial ViT + DCT DenseNet) и Linf projector
- `config.py` — валидация experiment и server configs
- `manifest.py` — построение детерминированных JSONL manifests
- `attack_api.py` — динамическая загрузка attack plug-ins
- `runner.py` — экспериментальный runner (загрузка моделей, clean predictions, attack, save, metrics)
- `cli.py` — CLI: `run`, `verify`, `preflight`, `build-manifests`, `validate-status`
- `artifacts.py` — верификация результатов (проверка файлов, hashes, norms, schema)
- `provenance.py` — сбор Git/Python/CUDA/model provenance
- `io.py` — I/O утилиты
- `status.py` — валидация status JSON
- `preflight.py` — deep preflight checker

### 2.5. Schemas (~90%)

JSON schemas:
- `schemas/attack-status.schema.json`
- `schemas/review-status.schema.json`
- `schemas/experiment.schema.json`
- `schemas/run-summary.schema.json`
- `schemas/server.schema.json`
- `ops/gpuq/job-spec.schema.json`

### 2.6. Documentation (~85%)

- `AGENTS.md` — правила для агентов (47 строк)
- `OPENCODE_LUNA.md` — спецификация атак и критерии (56 строк)
- `AGENT_SETUP.md` — инструкция по setup (134 строки)
- `research_experiment_plan.md` — полный научный план (309 строк)
- `docs/RUNBOOK_RU.md` — подробный runbook на русском (1833 строки)
- `docs/AGENT_PIPELINE.md` — контракт agent pipeline (300 строк)
- `docs/SCIENTIFIC_PIPELINE.md` — scientific pipeline contract (279 строк)
- `ops/gpuq/README.md` — документация GPUQ (140 строк)
- `tracking/README.md` — tracking ledger docs
- `manifests/README.md` — manifest docs

### 2.7. Environment Lock (~90%)

- `requirements.lock` — зафиксированные pip packages
- `environment.lock.json` — зафиксированные версии Python/Torch/CUDA
- `pyproject.toml` — project metadata
- `.python-version` — Python version pin
- `environment/reference-server-freeze.txt` — reference freeze
- `scripts/bootstrap_environment.sh` — автоматический bootstrap
- `scripts/prepare_server_assets.sh` — подготовка весов

### 2.8. Tests (~60%)

Существующие тесты:
- `tests/test_gpu_queue.py` — GPUQ unit tests (submit, idempotency, cancel, state machine)
- `tests/test_preprocessing.py` — preprocessing tests (DCT round-trip, project_linf, etc.)
- `tests/test_scientific_contract.py` — experiment contract, manifest, verification
- `tests/test_run_campaign.py` — campaign controller tests (530 строк, extensive mocking)
- `tests/conftest.py` — shared fixtures

---

## 3. Что частично реализовано

### 3.1. Attack Plug-ins (20%)

**Реализовано:**
- `attacks/ifgsm.py` — I-FGSM: полная реализация с shared preprocessing
- `attacks/fgsm.py` — FGSM: one-step targeted
- `attacks/template.py` — шаблон для новых атак

**Не реализовано (8 из 10 атак):**

| Атака | Зависимость | Сложность | Описание |
|---|---|---|---|
| `pgd` | I-FGSM | Низкая | PGD = I-FGSM + random start в epsilon-шаре |
| `mifgsm` | I-FGSM | Средняя | Momentum с нормализацией gradient |
| `di-mi-fgsm` | MI-FGSM | Средняя | Differentiable resize + random pad |
| `ti-di-mi-fgsm` | DI-MI-FGSM | Средняя | Gaussian convolution gradientов |
| `frequency-eot` | Transfer baseline | Высокая | DCT/IDCT, frequency masking, 3 варианта |
| `mig-cow` | Multi-source | Высокая | Integrated Gradients + COW consensus |
| `dd-fcma` | Все retained компоненты | Очень высокая | Комбинация всего + ablation |
| `prototype` | Disjoint manifest | Средняя | Cosine distance to target prototype |

**Примечание:** По замыслу системы, эти атаки реализует `attack-worker` агент по одной за сессию. Код не должен быть написан заранее человеком — агент делает это в рамках `/attack <name> <scope>`. Поэтому отсутствие кода — это **нормальное** состояние, а не баг.

### 3.2. Experiment Configs (15%)

**Существуют:**
- `configs/experiments/ifgsm-smoke.yaml` — smoke для I-FGSM
- `configs/experiments/fgsm-smoke.yaml` — smoke для FGSM
- `configs/pipeline/server.yaml` — server config

**Отсутствуют:**
- Development configs для всех атак (должны создаваться attack-worker'ом)
- Full configs (создаются только после finalist selection)
- Configs для DCT source model
- Configs для обоих направлений (real→fake, fake→real)
- Budget sensitivity configs (4/255, 8/255, 16/255)
- Replication configs (seeds 0, 1, 2)

### 3.3. Test Coverage (60%)

**Есть:**
- GPUQ unit tests (хорошее покрытие)
- Preprocessing tests
- Config validation tests
- Campaign controller tests (extensive)
- Manifest generation tests

**Не хватает:**
- Focused tests для конкретных атак (кроме ifgsm)
- Integration tests для полного run→verify цикла (с реальным GPU)
- Test fixtures с обоими классами и clean-correct примерами
- Regression tests для attack API contract
- End-to-end test: submit → schedule → run → verify → reconcile

---

## 4. Что отсутствует или заблокировано

### 4.1. Критически важное для первого запуска

| Компонент | Описание | Приоритет |
|---|---|---|
| **Real GPU smoke** | Ни один GPU-эксперимент ещё не был запущен | Критический |
| **Preflight pass на сервере** | Нужно подтвердить на целевом сервере | Критический |
| **Manifest review + commit** | Manifests созданы, но не reviewed/committed | Критический |
| **Scheduler recovery test** | Singleton scheduler test не покрыт автоматически | Высокий |
| **Byte-for-byte reproducibility** | Подтвердить двумя одинаковыми smoke runs | Высокий |

### 4.2. Инфраструктурные gaps

| Компонент | Описание | Сложность |
|---|---|---|
| **Per-worker Git worktrees** | Для параллельных workers. Сейчас контроллер последовательный. | Высокая |
| **Parallel workers** | `max_parallel_agents: 1` в CAMPAIGN.yaml — только цель, не реальность. Зависит от worktrees. | Высокая |
| **Artifact store** | Нет политики для управления adversarial image trees. Они хранятся in-place, без DVC/LFS. | Средняя |
| **Scheduler systemd service** | Scheduler запускается в tmux. Нет systemd unit / watchdog. | Низкая |
| **Notification/alerting** | Нет механизма уведомления о завершении/сбое задачи. | Средняя |
| **Dashboard / UI** | Только CLI и JSON. Нет веб-интерфейса для мониторинга. | Средняя |

### 4.3. Научные gaps

| Компонент | Описание | Блокирует |
|---|---|---|
| **Development manifest** | Нужен frozen, hash-identified manifest с 200 примерами на класс. TEST содержит только 100. | Все development runs |
| **DCT source model support** | `ifgsm` и `fgsm` поддерживают только `vit_b_16`. DCT source требует отдельный gradient path. | DCT→DCT и DCT→ViT evaluation |
| **Disjoint prototype manifest** | Для Phase F (prototype hypothesis) нужен отдельный reference manifest. | Phase F |
| **Official AADD test set** | `official_aadd_root: null`. Нет canonical location. | Official runs |
| **Multi-source evaluation** | Для MIG-COW нужно >=3 detector sources. Есть только 2. | Phase E (MIG-COW) |

### 4.4. Readiness checklist (из RUNBOOK_RU.md §29)

Незакрытые пункты:

- [ ] Ветка опубликована и установлена по canonical path
- [ ] Commit author — пользователь, нет agent contributor trailers
- [ ] Оба detector checkpoint прошли SHA-256
- [ ] LPIPS backbone и calibration прошли SHA-256
- [ ] Отдельный `.venv` создан
- [ ] `pip check` успешен
- [ ] Strict preflight полностью pass
- [ ] Manifests reviewed и committed
- [ ] OpenCode 1.18.26 загрузил project config
- [ ] Все 3 агента видны
- [ ] Skill виден
- [ ] Reviewer фактически read-only
- [ ] Singleton scheduler test прошёл
- [ ] Один synthetic real-GPU smoke прошёл
- [ ] Scheduler не выбрал занятую GPU
- [ ] Staged secret scan и human diff review

**Закрытые пункты:**
- [x] GPUQ unit tests прошли
- [x] Mocked nvidia-smi tests прошли
- [x] Duplicate/idempotency test прошёл
- [x] Cancel test прошёл
- [x] `run_campaign.py` поддерживает `queued/job_id`
- [x] Controller сопоставляет GPUQ `succeeded` с verifier `passed`
- [x] Controller не доверяет agent-written `passed`
- [x] Structured JSON reviewer parsing протестирован
- [x] Maximum external agents ограничен (1)
- [x] CAMPAIGN.yaml без legacy paths
- [x] Policy поэтапной реализации attacks объявлена
- [x] git diff --check успешен
- [x] Все Python tests прошли локально

---

## 5. Рекомендации по развитию

### 5.1. Немедленные шаги (перед первым GPU smoke)

1. **На сервере:**
   - Запустить `bash scripts/prepare_server_assets.sh`
   - Запустить `bash scripts/bootstrap_environment.sh`
   - Запустить `.venv/bin/python scripts/preflight.py --strict`
   - Запустить `.venv/bin/python -m ops.gpuq doctor --json`
   - Проверить OpenCode version и models

2. **Подготовить manifests:**
   - `.venv/bin/python -m attacklab.cli build-manifests ...`
   - Review и commit

3. **Первый smoke:**
   - Submit `ifgsm-smoke` job
   - Запустить scheduler (`--execute --max-running 1`)
   - Подтвердить `succeeded` + `verification.json: passed`
   - Два одинаковых smoke для подтверждения byte-for-byte reproducibility

### 5.2. Краткосрочные улучшения

| Улучшение | Описание | Эффект |
|---|---|---|
| **Development manifest** | Создать frozen manifest с 200+ примерами на класс (из TRAIN или отдельного split) | Разблокировать все development runs |
| **DCT source gradient path** | Реализовать DCT preprocessing в `attacklab/preprocessing.py` для использования как source | Разблокировать полную source×target matrix |
| **Focused attack tests** | Добавить test fixtures и shared test utilities для attack plug-ins | Ускорить agent работу и снизить failure rate |
| **Singleton scheduler test** | Автоматизировать тест singleton lock | Закрыть readiness checklist |

### 5.3. Среднесрочные улучшения

| Улучшение | Описание | Эффект |
|---|---|---|
| **Per-worker Git worktrees** | Каждый attack-worker получает отдельный worktree + task branch | Параллельные workers (2+ одновременно) |
| **Artifact store / DVC** | Политика управления adversarial image trees (контент-addressable, DVC или аналог) | Контроль за дисковым пространством |
| **Scheduler systemd unit** | Systemd service с watchdog и auto-restart | Надёжность для unattended campaigns |
| **Notification webhook** | Отправка уведомлений о статусе задач (Telegram, Slack, email) | Оперативное реагирование |
| **Budget sensitivity automation** | Автоматическое создание configs для epsilon grid после finalist selection | Ускорение Phase H |

### 5.4. Долгосрочные улучшения

| Улучшение | Описание |
|---|---|
| **Multi-GPU parallelism** | `--max-running > 1` с race/recovery tests |
| **Slurm integration** | Замена user-space GPUQ на Slurm/GRES/cgroups для production |
| **Web dashboard** | Flask/Streamlit для мониторинга кампании, графиков ASR, status |
| **Container isolation** | Docker/Podman для каждого worker-а (filesystem sandbox) |
| **CI/CD pipeline** | Автоматические tests при push, lint, secret scan |
| **Additional detectors** | Третий и более детекторы (для полноценного held-out COW) |
| **Official AADD integration** | Замороженный official test set и evaluator pipeline |
| **Experiment database** | Structured DB для cross-experiment comparison и meta-analysis |

### 5.5. Что теоретически стоит добавить в контроллер

1. **Retry policy** — автоматический retry для transient scheduler failures (не для научных failures).
2. **Timeout escalation** — увеличивать timeout для development/full runs на основе smoke timing.
3. **Dependency visualization** — экспорт графа CAMPAIGN.yaml в DOT/Mermaid для визуализации.
4. **Rollback** — возможность откатить статус задачи (с аудитом).
5. **Cost accounting** — учёт GPU-часов на каждую задачу и кампанию.
6. **Campaign comparison** — сравнение результатов двух кампаний (при изменении кода/конфига).
7. **Pre-computation hooks** — pre-compute clean predictions и cache для ускорения development runs.

### 5.6. Что стоит добавить в GPUQ

1. **Priority queue** — сейчас `priority` хранится, но не используется для ordering.
2. **Resource profiling** — запись фактического GPU memory usage и utilization во время run.
3. **Multi-GPU scheduling** — поддержка jobs, требующих >1 GPU (для будущих large-scale experiments).
4. **Graceful shutdown** — сигнал scheduler-у остановиться после завершения текущих jobs.
5. **Queue backup/restore** — экспорт и импорт SQLite state.
6. **Historical analytics** — хранение завершённых jobs для анализа patterns.

---

## Приложение: Критические пути

### Путь к первому development run

```
preflight pass → manifest review+commit → GPU smoke (ifgsm) →
  → byte-for-byte reproducibility → readiness checklist closed →
  → /campaign development
```

### Путь к full run

```
все development runs passed → select-finalists review →
  → explicit user authorization → /campaign full →
  → budget sensitivity + seed replication + full manifest →
  → official evaluation
```

### Путь к parallel workers

```
per-worker Git worktrees → concurrency tests →
  → max_parallel_agents > 1 → scheduler race tests →
  → 2 workers одновременно
```
