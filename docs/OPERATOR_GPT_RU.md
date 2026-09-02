# Инструкция оператора: запуск и контроль агентов

Для ассистента, который управляет системой с компьютера пользователя через SSH.
Читать целиком до первой команды.

---

## 1. Что это за система

Атаки на детекторы дипфейков. Работа разделена на два слоя, и смешивать их
нельзя.

**Научный слой** — `attacklab/`, `attacks/`, `evaluate.py`. Проводит эксперимент
и проверяет результат.

**Управляющий слой** — `.opencode/`, `scripts/run_campaign.py`, `ops/gpuq/`.
Раздаёт задачи агентам и владеет очередью GPU.

Цепочка одной задачи:

```
кампания → агент пишет атаку → кладёт задание в очередь → выходит
                                        ↓
        планировщик берёт GPU → запускает эксперимент → верификатор
                                        ↓
                     контроллер читает вердикт → passed / failed
```

**Четыре правила, которые нельзя нарушать:**

1. Агент никогда не выбирает GPU и не запускает эксперимент. Он только кладёт
   задание в очередь и завершается.
2. `passed` появляется **только** из отчёта верификатора. Написанному агентом
   `passed` контроллер не верит и отвергает задачу.
3. Каждый научный вход обязан быть в Git до старта эксперимента.
4. Прерывание контроллера не отменяет задание на GPU — им владеет планировщик.

---

## 2. Где что лежит

| | |
|---|---|
| Сервер | `aiattacks@10.24.1.21` |
| Рабочий каталог | `/home/aiattacks/oleg/aadd-attack-pipeline` |
| Ветка | `codex/aadd-agent-pipeline` |
| Датасет | `/home/aiattacks/dataset/celebA` — **только чтение** |
| Веса детекторов | `/home/aiattacks/oleg/aadd-attack-assets/weights` |
| Тяжёлые результаты | `/home/aiattacks/oleg/aadd-attack-runs` |

Внутри репозитория:

| Путь | Что это |
|---|---|
| `specs/attacks/*.yaml` | **источник правды** об атаках: параметры, бюджеты, тесты, пороги |
| `CAMPAIGN.yaml` | порядок задач и зависимости |
| `configs/experiments/` | замороженные конфиги прогонов, их пишет агент |
| `manifests/celebA/` | замороженные списки изображений |
| `tracking/runs/` | компактные результаты |
| `.gpuq/`, `.campaign/` | рабочие базы, **руками не трогать** |

---

## 3. Первый шаг: всё обновить

Всегда начинать с этого.

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline && git pull --rebase && git log --oneline -1
```

Сеть до GitHub нестабильна. Если оборвалось:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline && for i in $(seq 1 15); do git fetch --quiet origin codex/aadd-agent-pipeline 2>/dev/null && break; sleep 5; done && git rebase FETCH_HEAD
```

Проверить, что окружение цело:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline && .venv/bin/python -m attacklab.cli preflight --config configs/pipeline/server.yaml --deep 2>&1 | tail -3
```

Должно закончиться `{"status": "pass", "deep": true}`. Если нет — **дальше не
идти**, разбираться с тем, на что пожаловался preflight.

Проверить, что спецификации согласованы с реальностью:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline && .venv/bin/python -m attacklab.cli validate-specs --config configs/pipeline/server.yaml
```

`[BLOCK]` — задача не запустится, чинить до запуска. `[WARN ]` — принять к
сведению. Нужен `status: pass`.

---

## 4. Поднять планировщик

Без него очередь стоит и ничего не считается.

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline && tmux new-session -d -s gpuq -c "$PWD" ".venv/bin/python -m ops.gpuq run-scheduler --execute --allow-shared-gpu --max-running 1 --poll-seconds 10 --idle-samples 3 --headroom-mb 4096 2>&1 | tee -a .gpuq/scheduler.log"
```

`--allow-shared-gpu` **обязателен**: машина общая, все восемь карт постоянно
заняты другими пользователями, и без флага планировщик не возьмёт ни одну.

Проверить, что он видит карты:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline && tail -1 .gpuq/scheduler.log | python3 -c "import json,sys; d=json.load(sys.stdin); print('карт доступно:', len(d['eligible_gpu_uuids']))"
```

Ноль — планировщик запущен без флага, перезапустить.

---

## 5. Запустить работу

**Одна атака:**

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline && opencode run --command attack --dir "$PWD" <имя> <smoke|development>
```

**Вся кампания:**

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline && mkdir -p .campaign && tmux new-session -d -s campaign -c "$PWD" ".venv/bin/python scripts/run_campaign.py development 2>&1 | tee -a .campaign/driver.log"
```

Посмотреть план, ничего не запуская:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline && .venv/bin/python scripts/run_campaign.py development --dry-run
```

**`--auto` не передавать никогда.** Неожиданный запрос прав — это ошибка
конфигурации, её надо разобрать, а не проштамповать.

---

## 6. Следить за прогрессом

Главная команда, безопасно вызывать сколько угодно:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline && .venv/bin/python scripts/run_campaign.py status
```

Состояния задачи: `pending` → `running` → `passed` / `failed` / `blocked` /
`cancelled` / `skipped`.

Очередь:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline && .venv/bin/python -m ops.gpuq list --json
```

Состояния задания: `queued` → `reserving` → `running` → `validating` →
`succeeded` / `failed` / `cancelled` / `orphaned`.

Живые логи: `tmux attach -t gpuq` и `tmux attach -t campaign`, выход из tmux —
Ctrl+B затем D. Лог конкретного агента лежит рядом с его статусом в
`.campaign/runs/*/NN-задача-attempt-N/agent.log`.

---

## 7. Как сохраняются результаты

Два места, разного назначения.

**Компактные результаты** — в репозитории,
`tracking/runs/<кампания>/<задача>/attempt-0001/`:

| Файл | Содержимое |
|---|---|
| `summary.json` | итоговые числа: ASR, точность, SSIM, LPIPS, время |
| `verification.json` | вердикт верификатора, `passed` или `failed` |
| `per_sample_metrics.jsonl` | по строке на изображение |
| `norm_audit.json` | проверка бюджета по перечитанному с диска PNG |
| `provenance.json` | коммит, хеши весов и манифеста, версии, режим детерминизма |
| `resolved_config.yaml` | замороженный конфиг прогона |
| `gpuq_status.json` | состояние задания и длительность |

**Тяжёлые артефакты** — вне репозитория, в `/home/aiattacks/oleg/aadd-attack-runs`.
Там сами состязательные изображения. Каталог адресуется хешем конфига и
**неизменяем**: повторный прогон с тем же конфигом упадёт, а не перезапишет.

Прочитать результат:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline && python3 -m json.tool tracking/runs/<кампания>/<задача>/attempt-0001/summary.json
```

Что смотреть:

| Поле | Смысл |
|---|---|
| `per_model.*.targeted_asr_on_source_eligible` | доля успеха по каждому детектору |
| `per_model.*.denominator` | сколько изображений реально участвовало |
| `per_model.*.clean_accuracy_on_selected` | точность детектора до атаки |
| `constraint_violations` | **обязан быть 0** |
| `mean_ssim`, `mean_lpips` | качество изображения |
| `timing.elapsed_seconds` | сколько шёл прогон |

Диагональ матрицы (источник = цель) — белый ящик. Недиагональ — перенос.
У обычного I-FGSM перенос нулевой, это нормальная точка отсчёта.

**`verification.json` с `"outcome": "passed"` — единственное основание считать
прогон состоявшимся.** Успех smoke это инженерное доказательство, а не научное.

---

## 8. Если сломалось

| Симптом | Причина и что делать |
|---|---|
| `eligible_gpu_uuids: []` | планировщик без `--allow-shared-gpu`, перезапустить с флагом |
| `opencode: command not found` | `echo 'export PATH="$HOME/.opencode/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc` |
| `No module named pip` | повторно `bash scripts/bootstrap_environment.sh`, он чинит venv сам |
| `Scientific input is not committed to Git` | агент не сделал `git add`; проверить `git status`, застейджить и перезапустить задачу |
| `Expected a project-relative path` | путь должен быть относительным от корня репозитория |
| Задача сразу `failed`, `Invalid status JSON` | агент нарушил контракт статуса, смотреть `agent.log` рядом со статусом |
| Задание долго `queued` | нормально: нужно три подряд наблюдения с достаточной свободной памятью |
| tmux-сессия умирает мгновенно, лог не создан | `tee` не может писать в несуществующий каталог; `mkdir -p .campaign` перед запуском |
| Агент пишет `blocked`, `ModuleNotFoundError: numpy` | воркер запустил тест голым `python3`; зависимости в `.venv`. Проверить, что в правах агента разрешён `.venv/bin/python` |
| `blocked`: development manifest absent | манифесты существуют, но протокол требует **назначенного**. Он назван в `OPENCODE_LUNA.md`: `manifests/celebA/test_fake.jsonl` |
| `Heavy artifact directory already exists` | повтор с тем же конфигом; нужен новый `run_dir` |
| SSH рвётся на `kex_exchange_identification` | сработал лимит частых подключений, подождать 15 минут, не долбить |

Остановка и возобновление. **Ctrl+C на контроллере не отменяет задание на GPU.**
Задача остаётся `running` с сохранённым `job_id`:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline && .venv/bin/python scripts/run_campaign.py resume
```

Перезапустить упавшую задачу и всё после неё:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline && .venv/bin/python scripts/run_campaign.py resume --retry <task-id>
```

Отменить задание — только явно:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline && .venv/bin/python -m ops.gpuq cancel <job-id>
```

---

## 9. Как добавить новую атаку

Два файла, больше ничего. Код пишет агент.

**Первое** — спецификация `specs/attacks/<имя>.yaml`. Взять за образец
`specs/attacks/fgsm.yaml`. `idea_id` обязан совпадать с именем файла.

**Второе** — задачи в `CAMPAIGN.yaml`:

```yaml
      - id: <имя>-smoke
        attack: <имя>
        scope: smoke
        required: true
        needs: [<предыдущая задача>]

      - id: <имя>-development
        attack: <имя>
        scope: development
        required: true
        needs: [<имя>-smoke]
```

Ограничения: имена подходят под `^[a-z0-9][a-z0-9-]*$`, а `needs` и `after`
ссылаются только на задачи **выше** по файлу.

Проверить:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline && .venv/bin/python -m attacklab.cli validate-specs --config configs/pipeline/server.yaml && .venv/bin/python -m pytest tests -q 2>/dev/null || for f in tests/test_*.py; do PYTHONPATH=. .venv/bin/python "$f" 2>&1 | tail -1; done
```

Тест соответствия падает, если задача есть в расписании без спецификации или
наоборот.

---

## 10. Границы

- **Не коммитить от имени агента.** Автор только `Danchik757 <danzero.oreo@gmail.com>`,
  без упоминаний ассистентов и без trailer'ов.
- **Не менять** датасет, веса, `evaluate.py`, замороженные манифесты, готовые прогоны.
- **Не ослаблять проверки ради зелёных тестов.** Если `_require_tracked_inputs`,
  контракт статусов или верификатор мешают — это они работают, а не ломаются.
- **`--max-running` не выше 4.** Машина общая, у NVIDIA нет резервирования памяти:
  запас снижает вероятность столкновения с соседом, но не устраняет её.
- **Руками не редактировать** `.gpuq/`, `.campaign/`, файлы статусов и завершённые прогоны.
- Расхождение кода с документацией — сообщать, а не подгонять документацию под код.

---

## 11. Что сейчас не сделано

Знать, чтобы не выдать за готовое:

- **Воспроизводимость не подтверждена.** Runner использует
  `use_deterministic_algorithms(warn_only=True)`, потому что строгий режим падает
  на дифференцируемом resize. Двух одинаковых прогонов со сравнением хешей никто
  не делал.
- **Матрица переноса неполная.** Все атаки берут градиент только от ViT; строки
  DCT-источника нет.
- **NPR и AIDE не подключены.** Веса есть, адаптеров нет. У NPR один логит с
  сигмоидой, у AIDE **противоположное** соглашение о метках — перепутать значит
  получить зеркальный ASR, который выглядит правдоподобно.
- **Форматный конфаунд датасета.** Все real — JPG, все fake — PNG; метка
  предсказывается по расширению файла.
- **Контроллер последовательный.** Один агент за раз, отдельных worktree нет.
- **Пороги удержания нулевые** во всех спецификациях: удерживают при любом
  улучшении и не отбирают кандидатов.
