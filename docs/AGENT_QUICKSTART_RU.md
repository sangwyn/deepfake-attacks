# Как запускать агентов и давать им задачи

Короткая рабочая инструкция. Полное описание — в `docs/RUNBOOK_RU.md`.

Везде подразумевается:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline
```

---

## 0. Один раз за сессию: планировщик

Агенты **никогда** не запускают GPU сами. Они только кладут задание в очередь.
Считает планировщик — отдельный процесс, живущий в `tmux`.

```bash
tmux new-session -d -s gpuq -c /home/aiattacks/oleg/aadd-attack-pipeline \
  ".venv/bin/python -m ops.gpuq run-scheduler --execute --allow-shared-gpu \
   --max-running 1 --poll-seconds 10 --idle-samples 3 --headroom-mb 4096 \
   2>&1 | tee -a .gpuq/scheduler.log"
```

`--allow-shared-gpu` обязателен на этой машине: все 8 карт постоянно заняты
другими пользователями, без флага планировщик не возьмёт ни одну.
`--max-running 1` не повышать — очередь делится с коллегами.

Проверить, что живёт и видит карты:

```bash
tmux ls | grep gpuq
tail -1 .gpuq/scheduler.log | python3 -m json.tool | head -5
```

В поле `eligible_gpu_uuids` должны быть карты. Пустой список — планировщик
ничего не запустит, см. раздел «Если что-то не так».

---

## 1. Дать агенту одну задачу

```bash
opencode run --command attack --dir "$PWD" <атака> <scope>
```

Пример:

```bash
opencode run --command attack --dir "$PWD" fgsm smoke
```

Что произойдёт: агент `attack-worker` прочитает протокол, напишет
`attacks/fgsm.py`, добавит тесты, прогонит CPU-проверки, заморозит конфиг и
job spec, отправит задание в очередь и **выйдет**. Он не ждёт GPU.

Атаки кампании, в порядке возрастания стоимости:

| Атака | Градиентов на изображение | Обязательна |
|---|---|---|
| `ifgsm` | 10 | регрессия пайплайна, уже реализована |
| `fgsm` | 1 | да |
| `pgd` | 10 | да |
| `mi-di-fgsm` | ~11 | да |
| `ensemble-mi-eot` | ~100 | нет |
| `ssa` | ~200 | да |
| `mig-cow` | ~500 | нет |

Кроме `ifgsm` не реализована ни одна — их пишет агент, это его работа.

Unified Latent Optimization и DAELTA в кампанию не входят: на сервере нет
диффузионной модели, а `requirements.lock` проверяется по хешу, поэтому её
добавление — отдельное ревью окружения, а не задача агента.

Scope: `smoke` — 8 изображений, проверка работоспособности. `development` —
замороженный набор, seed 0, научный результат. `full` — только для
финалистов и только с явной авторизацией.

Другая модель (по умолчанию `naapi/gpt-5.6-luna`):

```bash
opencode run --command attack --dir "$PWD" --model bailian-payg/qwen3.7-plus fgsm smoke
```

Учтите: `AGENTS.md` требует ровно `naapi/gpt-5.6-luna`, и дисциплинированный
воркер на другой модели вправе отказаться работать. Это корректное поведение,
а не поломка.

---

## 2. Запустить всю кампанию

21 задача: 10 атак × (smoke + development) + финальный отбор.

```bash
tmux new-session -d -s campaign -c /home/aiattacks/oleg/aadd-attack-pipeline \
  ".venv/bin/python scripts/run_campaign.py development 2>&1 | tee -a .campaign/driver.log"
```

Контроллер сам поднимает по одному процессу OpenCode на задачу, ждёт её job в
очереди, сверяет отчёт верификатора и только потом идёт дальше. Задачи связаны
зависимостями: `pgd` требует пройденного `ifgsm-development`, `di-mi-fgsm`
требует `mifgsm`, и так далее.

Посмотреть план, ничего не запуская:

```bash
.venv/bin/python scripts/run_campaign.py development --dry-run
```

**`--auto` не передавать.** Неожиданный запрос прав — это ошибка конфигурации,
которую надо разобрать, а не проштамповать.

---

## 3. Проверять прогресс

Состояние кампании — главная команда:

```bash
.venv/bin/python scripts/run_campaign.py status
```

Она заодно идемпотентно сверяется с очередью, поэтому её безопасно звать
сколько угодно. Состояния задач: `pending`, `running`, `passed`, `failed`,
`blocked`, `cancelled`, `skipped`.

Очередь:

```bash
.venv/bin/python -m ops.gpuq list --json
.venv/bin/python -m ops.gpuq status <job-id> --json
```

Состояния job: `queued` → `reserving` → `running` → `validating` →
`succeeded` / `failed` / `cancelled` / `orphaned`.

Живые логи:

```bash
tmux attach -t gpuq        # планировщик, выйти Ctrl+B затем D
tmux attach -t campaign    # контроллер
tail -f .gpuq/scheduler.log
```

Лог конкретного агента лежит рядом с его статусом:

```bash
ls .campaign/runs/*/          # каталоги попыток
tail -50 .campaign/runs/*/01-*/agent.log
```

---

## 4. Проверять результаты

Артефакты прогона:

```
tracking/runs/<кампания>/<задача>/attempt-0001/
├── summary.json              главные числа
├── verification.json         вердикт верификатора
├── per_sample_metrics.jsonl  по каждому изображению
├── norm_audit.json           проверка бюджета L-inf после сохранения
├── provenance.json           хеши, коммит, версии, режим детерминизма
└── resolved_config.yaml      замороженный конфиг
```

Быстрый просмотр:

```bash
python3 -m json.tool tracking/runs/<кампания>/<задача>/attempt-0001/summary.json
python3 -m json.tool tracking/runs/<кампания>/<задача>/attempt-0001/verification.json
```

Что смотреть в `summary.json`:

| Поле | Смысл |
|---|---|
| `targeted_asr_on_source_eligible` | доля успеха по каждому детектору |
| `denominator` | сколько изображений реально участвовало |
| `clean_accuracy_on_selected` | точность детектора до атаки |
| `constraint_violations` | **обязан быть 0** |
| `mean_ssim`, `mean_lpips` | качество изображения |

`verification.json` с `"outcome": "passed"` — единственное основание считать
прогон состоявшимся. Успех smoke это инженерное доказательство, а не научное.

Перепроверить готовый прогон вручную:

```bash
.venv/bin/python -m attacklab.cli verify --run-dir tracking/runs/<кампания>/<задача>/attempt-0001
```

---

## 5. Остановка и возобновление

Ctrl+C на контроллере **не отменяет** GPU-задание — им владеет планировщик.
Задача останется `running` с сохранённым `job_id`, и следующий `resume` её
подберёт:

```bash
.venv/bin/python scripts/run_campaign.py status
.venv/bin/python scripts/run_campaign.py resume
```

Перезапустить упавшую задачу и всё после неё:

```bash
.venv/bin/python scripts/run_campaign.py resume --retry <task-id>
```

Отменить задание в очереди — только явно:

```bash
.venv/bin/python -m ops.gpuq cancel <job-id>
```

Вручную не редактировать: `.gpuq/`, `.campaign/`, файлы статусов, готовые
прогоны. Это операционные базы, их правит только код.

---

## 6. Если что-то не так

| Симптом | Причина и что делать |
|---|---|
| `eligible_gpu_uuids: []` | планировщик запущен без `--allow-shared-gpu`; перезапустить с флагом |
| `opencode: command not found` | `echo 'export PATH="$HOME/.opencode/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc` |
| `No module named pip` | повторно запустить `bash scripts/bootstrap_environment.sh`, он чинит venv сам |
| `Expected a project-relative path` | путь должен быть относительным от корня репозитория |
| `Scientific input is not committed to Git` | конфиг, манифест или lock не закоммичены — проверить и закоммитить |
| Задача сразу `failed` со `Invalid status JSON` | агент нарушил контракт статуса; смотреть `agent.log` рядом со статусом |
| Job долго `queued` | нормально: нужно 3 подряд наблюдения с достаточной свободной памятью |

Полная таблица — `docs/RUNBOOK_RU.md`, раздел 27.

---

## 7. Что ещё не проверено

- Первый реальный GPU-прогон и режим детерминизма `warn_only=True`
  (`attacklab/runner.py`). До двух одинаковых smoke-прогонов с совпадающими
  хешами выходов результатам репликации доверять нельзя.
- Параллельные воркеры и отдельные git worktree не реализованы. Контроллер
  последовательный, один воркер за раз.
- Форматный конфаунд датасета: все real — JPG, все fake — PNG. Метка
  предсказывается по расширению файла. Направление `real→fake` из-за этого
  под вопросом; `fake→real` (официальное) от этого свободно.
