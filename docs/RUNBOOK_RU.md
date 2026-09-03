# AADD Attack Pipeline: полная инструкция по установке и запуску

## 1. Назначение документа

Этот runbook описывает полный жизненный цикл проекта adversarial attacks on AI
detectors:

1. установка Git-ветки на сервер;
2. подготовка внешних весов и отдельного Python/CUDA-окружения;
3. проверка датасета и создание воспроизводимых manifests;
4. проверка OpenCode control plane;
5. запуск безопасной очереди GPU-задач;
6. подготовка атаки OpenCode-агентом;
7. выполнение и deterministic verification эксперимента;
8. отслеживание, остановка и возобновление задач;
9. сохранение компактных результатов и provenance в Git;
10. диагностика типичных ошибок.

Документ рассчитан на сервер:

```text
aiattacks@10.24.1.21
```

Все команды на сервере выполняются от пользователя `aiattacks`, если явно не
сказано обратное.

> Важно: текущая ветка ещё проходит интеграцию и review. До завершения раздела
> «Критерии готовности» нельзя запускать полную автоматическую campaign. При
> этом отдельные read-only проверки, manifests, `gpuq doctor`, preview
> scheduler и прошедший review одиночный smoke-job запускать можно.

---

## 2. Что входит в систему

Система состоит из четырёх независимых уровней.

### 2.1. Scientific layer

Каталог `attacklab/` отвечает за:

- строгую проверку server config и experiment config;
- воспроизводимые dataset manifests;
- загрузку attack plug-in;
- запуск одного frozen experiment;
- сохранение per-sample metrics;
- post-save проверку ограничения L-infinity;
- сбор Git/Python/CUDA/model provenance;
- deterministic verification результата.

Основная точка входа:

```bash
.venv/bin/python -m attacklab.cli --help
```

Доступные команды:

```text
preflight         проверить сервер, датасет, веса и окружение
build-manifests   построить manifests датасета
run               выполнить один experiment
verify            проверить готовый experiment
validate-status   проверить JSON статуса агента или reviewer-а
```

### 2.2. OpenCode control plane

OpenCode используется для подготовки кода атаки, frozen config и постановки
эксперимента в очередь. В проекте определены три агента:

| Агент | Роль |
|---|---|
| `coordinator` | управляет campaign и читает её состояние |
| `attack-worker` | реализует одну атаку, запускает CPU-тесты и ставит job в очередь |
| `campaign-reviewer` | read-only проверяет scientific evidence и выбирает finalists |

Команды OpenCode:

| Команда | Назначение |
|---|---|
| `/attack` | подготовить одну атаку и поставить её в GPUQ |
| `/campaign` | создать, продолжить или показать campaign |
| `/campaign-review` | выполнить read-only review завершённых результатов |

Skill:

```text
.opencode/skills/attack-execution/SKILL.md
```

Он задаёт единый protocol реализации и проверки attack plug-in.

### 2.3. GPUQ

Каталог `ops/gpuq/` содержит пользовательскую GPU-очередь:

- SQLite хранит jobs и transitions;
- singleton lock не даёт запустить два scheduler-а одного проекта;
- `nvidia-smi` используется для проверки GPU;
- GPU выбирается по UUID, а не только по индексу;
- задача стартует только после нескольких последовательных idle samples;
- используется memory headroom;
- агент не может передать произвольную shell-команду;
- scheduler запускает только фиксированный `attacklab.cli run`;
- после run автоматически вызывается `attacklab.cli verify`;
- queue никогда не завершает чужие процессы.

Основная точка входа:

```bash
.venv/bin/python -m ops.gpuq --help
```

### 2.4. Git experiment ledger

Каталог `tracking/` хранит небольшие файлы, необходимые для понимания и
воспроизведения эксперимента:

- experiment config;
- job spec;
- status JSON;
- manifest snapshot или его hash;
- summary;
- norm audit;
- verifier report;
- provenance;
- hashes тяжёлых artifacts.

В Git не сохраняются:

- API-ключи;
- `.gpuq/*.sqlite`;
- live locks;
- полные agent logs;
- model weights;
- adversarial image trees;
- Python virtual environment;
- `node_modules`.

---

## 3. Каноническая организация каталогов

На сервере используются следующие пути:

```text
/home/aiattacks/oleg/aadd-attack-pipeline/
    Git checkout, scientific code, OpenCode control plane, tracking metadata

/home/aiattacks/dataset/celebA/
    внешний read-only dataset

/home/aiattacks/oleg/aadd-attack-assets/weights/
    внешние detector weights

/home/aiattacks/.cache/torch/hub/checkpoints/
    общий read-only cache AlexNet для LPIPS

/home/aiattacks/oleg/aadd-attack-runs/
    тяжёлые generated images и runtime artifacts
```

Dataset, weights и generated images специально отделены от Git checkout.
Агент не должен копировать датасет внутрь проекта или изменять исходные
изображения.

Канонический CelebA layout:

```text
/home/aiattacks/dataset/celebA/
├── TRAIN/
│   ├── TRAIN_REAL/    # label 0, ожидается 1500 изображений
│   └── TRAIN_FAKE/    # label 1, ожидается 1500 изображений
└── TEST/
    ├── TEST_REAL/     # label 0, ожидается 100 изображений
    └── TEST_FAKE/     # label 1, ожидается 100 изображений
```

Пути `/data2/aiattacks/...` устарели и не должны использоваться.

---

## 4. Требуемые веса

### 4.1. Detector checkpoints

Нужны два detector checkpoint:

```text
vit_b_16.pth
densenet121_dct.pth
```

Ожидаемые SHA-256:

```text
vit_b_16.pth
5e9677d88a7af10791001796eb43d0d060fada3758369814d6d7832934758d81

densenet121_dct.pth
5bbaf5c5c0e296d5e819a0b401198c73ad69c6bbc8f372579de5ee5c11d5e643
```

Это веса детекторов, а не отдельной attack model.

### 4.2. LPIPS weights

Для `lpips==0.1.4` используются два компонента.

AlexNet backbone:

```text
/home/aiattacks/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
SHA-256:
7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02
```

LPIPS calibration внутри Python package:

```text
weights/v0.1/alex.pth
SHA-256:
df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0
```

Scheduler и OpenCode-агенты не должны автоматически скачивать или заменять
эти файлы. Любое несовпадение hash является blocking error.

---

## 5. Зафиксированное окружение

Целевое окружение:

```text
Python:               3.12.3
PyTorch:              2.3.0+cu121
torchvision:          0.18.0+cu121
PyTorch CUDA runtime: 12.1
cuDNN runtime:        8.9.2 / 8902
NVIDIA driver:        не ниже 525.60
server driver:        550.120
GPU:                  NVIDIA L40, 8 устройств
OpenCode:             1.18.26
OpenCode model:       naapi/gpt-5.6-terra
```

Файлы окружения:

```text
.python-version
requirements.lock
environment.lock.json
environment/reference-server-freeze.txt
pyproject.toml
```

Нельзя использовать venv другого пользователя как рабочее окружение. Ветка
содержит reference freeze ранее проверенного venv только как evidence.

---

## 6. Установка ветки на сервер

### 6.1. Проверить, что ветка опубликована

На локальном компьютере:

```bash
cd /Users/admin/Documents/ChatGPT/Detecting/aadd-attack-pipeline
git branch --show-current
git status --short
git log -1 --format='author=%an <%ae>%ncommitter=%cn <%ce>'
```

Ожидаемая ветка:

```text
codex/aadd-agent-pipeline
```

Коммит должен быть сделан от пользовательской Git identity, без
`Co-authored-by` от агентов. Перед push:

```bash
git config user.name
git config user.email
git diff --check
git diff --cached
```

После review:

```bash
git push -u origin codex/aadd-agent-pipeline
```

Если authentication не работает, сначала восстановить пользовательскую GitHub
аутентификацию. Contributor определяется прежде всего author email коммита, а
не именем процесса, выполнившего `git push`.

### 6.2. Клонировать ветку

На сервере:

```bash
ssh aiattacks@10.24.1.21
```

Проверить, что целевой каталог ещё не занят другим checkout:

```bash
test ! -e /home/aiattacks/oleg/aadd-attack-pipeline
```

Клонировать только нужную ветку:

```bash
git clone \
  --branch codex/aadd-agent-pipeline \
  --single-branch \
  https://github.com/sangwyn/deepfake-attacks.git \
  /home/aiattacks/oleg/aadd-attack-pipeline
```

Перейти в корень:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline
```

Проверить binding:

```bash
pwd
git branch --show-current
git rev-parse HEAD
git status --short
```

OpenCode, preflight, GPUQ и campaign всегда запускаются из этого каталога.

---

## 7. Подготовка внешних detector weights

В проекте предусмотрен проверяемый helper:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline
bash scripts/prepare_server_assets.sh
```

Он:

1. читает существующие веса из
   `/home/aiattacks/oleg/deepfake-attacks/weights`;
2. проверяет их SHA-256;
3. создаёт отдельную директорию
   `/home/aiattacks/oleg/aadd-attack-assets/weights`;
4. копирует веса без изменения source files;
5. повторно проверяет SHA-256;
6. отказывается перезаписывать уже существующий target.

Ручная проверка:

```bash
sha256sum \
  /home/aiattacks/oleg/aadd-attack-assets/weights/vit_b_16.pth \
  /home/aiattacks/oleg/aadd-attack-assets/weights/densenet121_dct.pth
```

Если target уже существует, helper специально завершится с ошибкой. Нельзя
удалять или перезаписывать файл вслепую: сначала сравнить его hash и выяснить
происхождение.

---

## 8. Создание Python environment

Запускать bootstrap должен человек вне automated OpenCode session:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline
bash scripts/bootstrap_environment.sh
```

Bootstrap:

- требует именно `python3.12` версии 3.12.3;
- создаёт `.venv` внутри проекта;
- фиксирует pip/setuptools/wheel;
- устанавливает `requirements.lock`;
- устанавливает сам проект editable без повторной установки dependencies;
- выполняет deep preflight.

### Про `ensurepip` на Debian и Ubuntu

Системный `python3.12` в Debian и Ubuntu поставляется **без** `ensurepip` —
он вынесен в отдельный пакет `python3.12-venv`, а на общем сервере его
установка требует root. Поэтому bootstrap не полагается на `ensurepip`:

- если `ensurepip` доступен, `.venv` создаётся обычным способом;
- если нет, окружение создаётся через `venv --without-pip`, после чего pip
  ставится скриптом `get-pip.py` сразу нужных закреплённых версий.

Проверка наличия pip выполняется отдельно от создания `.venv`, поэтому
повторный запуск чинит и окружение, оставшееся без pip после неудачной
попытки. Удалять `.venv` вручную не нужно.

Источник `get-pip.py` при необходимости переопределяется (например для
внутреннего зеркала или офлайн-установки):

```bash
GET_PIP_URL=https://внутреннее-зеркало/get-pip.py \
  bash scripts/bootstrap_environment.sh
```

Если бинарник называется иначе:

```bash
PYTHON_BIN=/полный/путь/к/python3.12 \
  bash scripts/bootstrap_environment.sh
```

Проверить версии:

```bash
.venv/bin/python -c 'import platform; print(platform.python_version())'
.venv/bin/python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.backends.cudnn.version()); print(torch.cuda.is_available()); print(torch.cuda.device_count())'
.venv/bin/python -m pip check
```

Ожидается:

```text
3.12.3
2.3.0+cu121
12.1
8902
True
8
```

Не использовать системный `python3` для pipeline-команд после создания venv.
Во всех последующих разделах канонический интерпретатор:

```text
/home/aiattacks/oleg/aadd-attack-pipeline/.venv/bin/python
```

---

## 9. Preflight

### 9.1. Строгая проверка

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline
.venv/bin/python scripts/preflight.py \
  --strict \
  --output tracking/preflight.json
```

Эквивалент через module CLI:

```bash
.venv/bin/python -m attacklab.cli preflight \
  --config configs/pipeline/server.yaml \
  --deep \
  --output tracking/preflight.json
```

Preflight проверяет:

- checkout находится ровно по configured project path;
- четыре class directory датасета существуют;
- counts равны 1500/1500/100/100;
- detector checkpoints существуют и совпадают по SHA-256;
- AlexNet backbone совпадает по SHA-256;
- LPIPS calibration внутри package совпадает по SHA-256;
- Python и packages соответствуют lock;
- Torch видит CUDA;
- Torch/CUDA/cuDNN имеют нужные версии;
- `nvidia-smi` доступен;
- OpenCode существует по абсолютному пути;
- OpenCode version равна 1.18.26;
- runtime parent и tracking root существуют;
- environment lock и reference freeze не изменены.

Любой `[FAIL]` блокирует запуск. Нельзя вручную заменить failed status на
warning.

### 9.2. Проверка OpenCode отдельно

```bash
/home/aiattacks/.opencode/bin/opencode --version
/home/aiattacks/.opencode/bin/opencode models
```

Ожидается:

```text
1.18.26
naapi/gpt-5.6-terra
```

---

## 10. Dataset manifests

Эксперимент не должен читать population через случайный glob. Population
задаётся frozen JSONL manifest.

Создать manifests:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline
.venv/bin/python -m attacklab.cli build-manifests \
  --config configs/pipeline/server.yaml \
  --output-dir manifests/celebA
```

Будут созданы:

```text
manifests/celebA/train_real.jsonl
manifests/celebA/train_fake.jsonl
manifests/celebA/test_real.jsonl
manifests/celebA/test_fake.jsonl
manifests/celebA/catalog.json
```

Каждая строка manifest содержит:

- schema version;
- stable sample ID;
- split;
- class name;
- explicit label;
- relative dataset path;
- SHA-256 исходного файла;
- размер в байтах.

Команда также запрещает duplicate image content между entries и по умолчанию
не перезаписывает существующий manifest.

Проверить:

```bash
wc -l manifests/celebA/*.jsonl
sha256sum manifests/celebA/*.jsonl manifests/celebA/catalog.json
git status --short manifests/celebA
```

Ожидаемые counts:

```text
train_real.jsonl  1500
train_fake.jsonl  1500
test_real.jsonl    100
test_fake.jsonl    100
```

До настоящего experiment manifests должны быть reviewed и закоммичены
пользователем:

```bash
git add manifests/celebA tracking/preflight.json
git diff --cached
git commit -m "data: freeze audited CelebA manifests"
```

Scientific runner специально отказывается использовать untracked manifest.

---

## 11. Experiment config

Smoke example:

```text
configs/experiments/ifgsm-smoke.yaml
```

Основные поля:

```yaml
schema_version: 1
experiment_id: ifgsm-vit-test-fake-smoke-seed0
scope: smoke
seed: 0
server_config: configs/pipeline/server.yaml

dataset:
  manifest: manifests/celebA/test_fake.jsonl
  source_label: 1
  selection: manifest-order
  sample_limit: 8
  require_clean_correct: true

attack:
  name: ifgsm
  module: attacks.ifgsm
  source_model: vit_b_16
  target_class: 0
  parameters:
    epsilon: 0.03137254901960784
    step_size: 0.00784313725490196
    iterations: 10
```

`0.03137254901960784` соответствует `8/255`, а
`0.00784313725490196` — `2/255`.

Правила:

- config находится внутри Git project;
- `experiment_id` уникален;
- manifest уже закоммичен;
- `source_label=1` означает fake;
- `target_class=0` означает targeted fake-to-real attack;
- source model входит в evaluation model list;
- `attack.parameters.epsilon` точно совпадает с
  `constraint.epsilon`;
- output format — lossless PNG;
- SSIM, LPIPS и targeted ASR обязательны;
- полный run не запускается с `sample_limit` smoke-конфига;
- изменение seed, budget или population создаёт новый config/experiment ID.

Перед queue submission config должен быть закоммичен:

```bash
git status --short configs/experiments
git add configs/experiments/ifgsm-smoke.yaml
git diff --cached
git commit -m "experiment: freeze IFGSM smoke configuration"
```

---

## 12. Attack plug-in

Каждая атака находится в отдельном module под `attacks/`.

Минимальный interface:

```python
ATTACK_CONTRACT = {
    "version": 1,
    "supported_source_models": ["vit_b_16"],
}

def attack(
    image,
    classifiers,
    device,
    source_model,
    target_class,
    **parameters,
):
    ...
```

Input:

```text
numpy.ndarray
dtype uint8
shape H x W x 3
RGB range 0..255
```

Output обязан иметь тот же shape и dtype.

Текущий `ifgsm` поддерживает только source model `vit_b_16`. Нельзя просто
поменять YAML на `densenet121_dct`: для DCT source нужна отдельная атака с
дифференцируемым и проверенным DCT preprocessing path.

Чтобы заменить атаку:

1. скопировать interface из `attacks/template.py`;
2. создать новый module;
3. объявить source models;
4. добавить focused CPU/unit tests;
5. создать отдельный experiment config;
6. не изменять evaluator, manifests, labels и weights без отдельного review;
7. поставить experiment в GPUQ.

---

## 13. Проверка GPUQ

Инициализировать/проверить очередь:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline
.venv/bin/python -m ops.gpuq doctor --json
```

Doctor показывает:

- project root;
- путь `.gpuq`;
- наличие SQLite database;
- наличие fixed runner `attacklab.cli`;
- путь `nvidia-smi`;
- GPU inventory;
- UUID;
- free memory;
- utilization;
- compute PIDs;
- доступность singleton scheduler lock.

`scheduler_lock.available=false` допустимо только если уже работает ровно один
ожидаемый scheduler этого проекта.

Посмотреть queue:

```bash
.venv/bin/python -m ops.gpuq list --json
```

Посмотреть только определённое состояние:

```bash
.venv/bin/python -m ops.gpuq list --state queued --json
.venv/bin/python -m ops.gpuq list --state running --json
.venv/bin/python -m ops.gpuq list --state failed --json
```

---

## 14. Job spec

GPUQ не принимает shell-команду. Она принимает только schema-v1 JSON для
фиксированного task kind `attack-experiment`.

Пример:

```json
{
  "schema_version": 1,
  "task_kind": "attack-experiment",
  "config_path": "configs/experiments/ifgsm-smoke.yaml",
  "run_dir": "tracking/runs/manual-ifgsm/ifgsm-smoke",
  "requested_memory_mb": 24000,
  "timeout_seconds": 7200,
  "priority": 0,
  "max_attempts": 1
}
```

Сохранить input, например, как:

```text
tracking/runs/manual-ifgsm/ifgsm-smoke/job-spec.input.json
```

Ограничения:

- `task_kind` только `attack-experiment`;
- `config_path` project-relative и существует;
- hash config вычисляется при submission;
- `run_dir` project-relative;
- `run_dir` начинается с `tracking/runs/`;
- после `tracking/runs/` должно быть не менее двух компонентов;
- абсолютные пути запрещены;
- `..` запрещён;
- arbitrary `command` запрещён;
- arbitrary environment variables запрещены;
- GPU index/UUID в job spec запрещён;
- повтор идентичного spec использует idempotency key и не создаёт duplicate.

Поставить job:

```bash
.venv/bin/python -m ops.gpuq submit \
  tracking/runs/manual-ifgsm/ifgsm-smoke/job-spec.input.json
```

Ответ содержит:

```json
{
  "created": true,
  "idempotency_key": "...",
  "job_id": "...",
  "job_spec_snapshot": ".../job_spec.json",
  "schema_version": 1,
  "state": "queued"
}
```

Сохранить `job_id`. Snapshot создаётся автоматически в `run_dir`.

---

## 15. Запуск scheduler-а

### 15.1. Безопасный preview

По умолчанию scheduler ничего не запускает:

```bash
.venv/bin/python -m ops.gpuq run-scheduler --once
```

Он покажет, какие jobs и GPU видит, но не запустит runner.

### 15.2. Реальный запуск

Реальное выполнение требует явного `--execute`:

```bash
.venv/bin/python -m ops.gpuq run-scheduler \
  --execute \
  --max-running 1 \
  --poll-seconds 10 \
  --idle-samples 3 \
  --headroom-mb 4096 \
  --max-idle-utilization 5 \
  --retry-delay-seconds 30 \
  --validation-timeout-seconds 600
```

Для длительной работы сначала допустим `tmux`:

```bash
tmux new -s aadd-gpuq
cd /home/aiattacks/oleg/aadd-attack-pipeline
.venv/bin/python -m ops.gpuq run-scheduler \
  --execute \
  --max-running 1 \
  --poll-seconds 10 \
  --idle-samples 3 \
  --headroom-mb 4096
```

Отключиться от tmux:

```text
Ctrl-b d
```

Вернуться:

```bash
tmux attach -t aadd-gpuq
```

На первом этапе оставлять `--max-running 1`. Увеличивать количество
одновременных GPU jobs можно только после real-GPU race/recovery tests и
согласования с другими пользователями сервера.

### 15.3. Как scheduler выбирает GPU

GPU подходит только если:

- на неё нет lease нашей queue;
- получен cooperative file lock;
- `nvidia-smi` успешно вернул telemetry;
- нет compute process;
- достаточно free memory с учётом headroom;
- utilization ниже threshold;
- idle condition наблюдалось несколько раз подряд;
- config hash не изменился после submission.

Перед worker устанавливается `CUDA_VISIBLE_DEVICES` на выбранный GPU UUID.
Агент сам этого не делает.

> Ограничение: user-space scheduler не может запретить другому пользователю
> обойти очередь и запустить процесс между проверкой и нашим стартом. Строгая
> гарантия требует общей очереди для всех пользователей или Slurm/GRES/cgroups.

---

## 16. Monitoring, status и cancel

Состояния GPUQ:

```text
queued
reserving
running
validating
succeeded
failed
retry_wait
cancelled
orphaned
```

Проверить job:

```bash
.venv/bin/python -m ops.gpuq status <JOB_ID> --json
```

Проверить все jobs:

```bash
.venv/bin/python -m ops.gpuq list --json
```

Отменить queued job или запросить остановку принадлежащего queue process:

```bash
.venv/bin/python -m ops.gpuq cancel <JOB_ID>
```

GPUQ может завершать только собственный runner process group. Она не должна
выполнять `kill` для чужого PID.

Не редактировать `.gpuq/queue.sqlite` вручную. Не удалять lock files во время
работы scheduler-а.

---

## 17. Что делает scientific runner

Scheduler строит фиксированные argv:

```text
python -m attacklab.cli run
  --config <validated project-relative config>
  --run-dir <tracking run dir>/attempt-0001

python -m attacklab.cli verify
  --run-dir <tracking run dir>/attempt-0001
```

Runner:

1. выполняет strict config validation;
2. выполняет deep preflight;
3. требует, чтобы config, server config, manifest и environment lock были в Git;
4. проверяет manifest hashes исходных изображений;
5. фиксирует seeds и deterministic Torch settings;
6. загружает оба detector checkpoint;
7. вычисляет clean predictions;
8. формирует clean-correct denominator source detector-а;
9. вызывает attack plug-in только для eligible samples;
10. сохраняет adversarial examples в lossless PNG;
11. повторно читает сохранённый PNG;
12. измеряет реальный post-save L-infinity;
13. вычисляет SSIM и LPIPS;
14. вычисляет targeted predictions для всех detectors;
15. сохраняет metadata и provenance;
16. вызывает deterministic verifier.

Тяжёлые PNG сохраняются снаружи Git в:

```text
/home/aiattacks/oleg/aadd-attack-runs/<experiment-id>/<content-key>/images/
```

---

## 18. Run artifacts

Каждый attempt содержит:

```text
resolved_config.yaml
resolved_server_config.yaml
manifest.snapshot.jsonl
preflight.json
selection.jsonl
per_sample_metrics.jsonl
norm_audit.json
summary.json
provenance.json
artifacts.json
verification.json
```

Назначение:

| Файл | Содержимое |
|---|---|
| `resolved_config.yaml` | точный experiment config |
| `resolved_server_config.yaml` | точные server paths и hashes |
| `manifest.snapshot.jsonl` | frozen dataset inventory |
| `preflight.json` | состояние системы непосредственно перед run |
| `selection.jsonl` | clean predictions и eligibility |
| `per_sample_metrics.jsonl` | predictions, L-inf, SSIM, LPIPS, output hashes |
| `norm_audit.json` | post-save constraint verification |
| `summary.json` | denominators и aggregate metrics |
| `provenance.json` | Git SHA, Python, packages, CUDA и weight hashes |
| `artifacts.json` | index metadata/heavy artifacts |
| `verification.json` | финальный deterministic verdict |

Успех эксперимента определяется только так:

```text
verification.json.outcome == "passed"
```

Exit code 0 от OpenCode или runner сам по себе не является scientific proof.

---

## 19. Ручная verification

После завершения job:

```bash
.venv/bin/python -m attacklab.cli verify \
  --run-dir tracking/runs/manual-ifgsm/ifgsm-smoke/attempt-0001
```

Verifier проверяет:

- обязательные файлы;
- JSON/JSONL structure;
- selected/eligible/evaluated counts;
- уникальные sample IDs;
- отсутствие NaN/Infinity;
- наличие clean/adversarial predictions;
- наличие каждого output PNG;
- SHA-256 каждого output PNG;
- post-save L-infinity;
- согласованность violation counts;
- config и manifest hashes в provenance.

При failed verification нельзя вручную поставить task status `passed`.

---

## 20. Проверка status JSON

Attack status schema:

```text
schemas/attack-status.schema.json
```

Проверить:

```bash
.venv/bin/python -m attacklab.cli validate-status \
  --kind attack \
  --status-file tracking/runs/<campaign>/<task>/status.json
```

Минимальный queued status:

```json
{
  "schema_version": 1,
  "task_id": "ifgsm-smoke",
  "attack": "ifgsm",
  "scope": "smoke",
  "outcome": "queued",
  "decision": "pending",
  "summary": "CPU checks passed; immutable experiment was queued.",
  "job_id": "<GPUQ_JOB_ID>",
  "job_spec": "tracking/runs/.../job_spec.json"
}
```

Для `passed` дополнительно обязательны:

- `configs`;
- `results`;
- `evidence`;
- `verifier_report`.

Reviewer status schema:

```text
schemas/review-status.schema.json
```

Проверить:

```bash
.venv/bin/python -m attacklab.cli validate-status \
  --kind review \
  --status-file tracking/runs/<campaign>/select-finalists/status.json
```

---

## 21. Запуск OpenCode

### 21.1. Проверить project config

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline
/home/aiattacks/.opencode/bin/opencode debug config
/home/aiattacks/.opencode/bin/opencode debug agent coordinator
/home/aiattacks/.opencode/bin/opencode debug agent attack-worker
/home/aiattacks/.opencode/bin/opencode debug agent campaign-reviewer
/home/aiattacks/.opencode/bin/opencode debug skill
```

Нужно убедиться, что:

- используется project-local `opencode.jsonc`;
- model везде `naapi/gpt-5.6-terra`;
- built-in Task/subagents выключены;
- reviewer не имеет edit/bash/network;
- attack-worker не может запускать GPU runner напрямую;
- skill `attack-execution` виден;
- commands `/attack`, `/campaign`, `/campaign-review` видны.

### 21.2. Интерактивный coordinator

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline
/home/aiattacks/.opencode/bin/opencode \
  --agent coordinator \
  --model naapi/gpt-5.6-terra \
  .
```

Внутри OpenCode:

```text
/campaign status
/campaign development
/campaign resume
/campaign resume --retry <task-id>
/campaign full
```

`/campaign full` является отдельной явной авторизацией. Development run не
даёт автоматического разрешения на full.

### 21.3. Подготовить одну атаку

В отдельной OpenCode session:

```text
/attack ifgsm smoke
```

Формальный interface:

```text
/attack <name> <smoke|development|full> [status-file] [task-id]
```

Worker должен:

1. прочитать `AGENTS.md`;
2. загрузить skill `attack-execution`;
3. проверить prerequisites;
4. изменить только нужную атаку/config/tests;
5. выполнить разрешённые CPU tests;
6. создать immutable job spec;
7. вызвать только `ops.gpuq submit`;
8. записать provisional `queued`, `failed` или `blocked` status;
9. остановиться.

Worker не должен:

- выбирать GPU;
- вызывать `nvidia-smi` для выбора карты;
- устанавливать `CUDA_VISIBLE_DEVICES`;
- запускать `attacklab.cli run`;
- запускать verifier;
- запускать scheduler;
- ждать освобождения GPU;
- устанавливать packages;
- скачивать weights;
- коммитить или пушить;
- вызывать другого агента.

### 21.4. Non-interactive OpenCode

Controller использует именно такую форму (агент выбирается frontmatter'ом
команды, поэтому `--agent` не передаётся; `--model` и `--variant` добавляются
только если переданы явно):

```bash
opencode run \
  --command attack \
  --dir /home/aiattacks/oleg/aadd-attack-pipeline \
  --title '<campaign-id>:<task-id>' \
  '<attack>' '<scope>' '<status-file>' '<task-id>'
```

Проверить фактический argv, ничего не запуская:

```bash
.venv/bin/python scripts/run_campaign.py development --dry-run
```

Не использовать `--auto`. Неожиданный permission prompt означает ошибку
policy/config, которую надо исследовать.

---

## 22. Несколько агентов

Предполагаемая начальная политика:

```text
максимум внешних OpenCode workers: 2
максимум одновременно GPU jobs:    1
```

Это разные лимиты. Два агента могут параллельно готовить независимые атаки, но
GPUQ запускает только один GPU experiment.

Нельзя запускать два attack-worker в одном mutable Git checkout, если они могут
редактировать пересекающиеся файлы. Для каждого worker нужен отдельный Git
worktree и отдельная task branch. Интегрировать изменения должен controller
или человек после review.

`scripts/run_campaign.py` поддерживает `queued/job_id` и queue reconciliation,
но **не** поддерживает параллельные workers и external worktrees. Контроллер
последовательный: он запускает один OpenCode worker, дожидается его job в
очереди и только затем переходит к следующей задаче. Поэтому фактический лимит
внешних workers сейчас равен 1, а `control.max_parallel_agents` в
`CAMPAIGN.yaml` — заявленная цель, а не действующее поведение.

Параллельный запуск двух workers без отдельных worktrees остаётся запрещённым.

---

## 23. Campaign lifecycle

Целевой lifecycle:

```text
pending
   |
   v
OpenCode attack-worker
   |
   v
queued -> reserving -> running -> validating
                                  |          |
                                  |          v
                                  |       failed
                                  v
                              succeeded
                                  |
                                  v
deterministic controller converts task to passed
                                  |
                                  v
read-only campaign-reviewer
```

Attack status outcomes:

```text
queued | running | passed | failed | blocked | cancelled
```

Scientific decisions:

```text
pending | baseline | retain | reject | not_applicable
```

`passed` означает technical/scientific validity run artifact. Это не всегда
означает, что атака лучше baseline. Слабая, но корректно выполненная атака может
иметь `outcome=passed` и позднее `decision=reject`.

### Как это реализовано в контроллере

Worker завершает работу сразу после submit и пишет только провизорный
`queued` (либо `failed`/`blocked`). Дальше действует контроллер:

1. сохраняет `job_id` и `job_spec` в durable campaign state и переводит задачу
   в `running` — это состояние принадлежит контроллеру, не агенту;
2. опрашивает `gpuq` (только чтение) до терминального состояния job;
3. читает `verification.json` из attempt-директории, которую создал scheduler;
4. отображает результат в состояние задачи:

   | Состояние `gpuq` | Отчёт верификатора | Задача |
   |---|---|---|
   | `succeeded` | `outcome: passed` | `passed` |
   | `succeeded` | `outcome: failed` | `failed` |
   | `succeeded` | отсутствует | `failed` |
   | `failed`, `orphaned` | — | `failed` |
   | `cancelled` | — | `cancelled` |

5. атомарно перезаписывает status JSON собственным документом, сохраняя
   оригинал воркера рядом как `status.worker.json` для аудита.

`passed` возникает **только** из отчёта детерминированного верификатора.
Написанный агентом `passed` контроллер отклоняет с ошибкой.

Реконсиляция идемпотентна и выполняется также в начале `resume` и `status`,
поэтому убитый во время опроса контроллер восстанавливается:

```bash
.venv/bin/python scripts/run_campaign.py status
.venv/bin/python scripts/run_campaign.py resume
```

Ctrl+C во время опроса **не** отменяет GPU job: задачей владеет scheduler.
Задача остаётся `running` с сохранённым `job_id`, и следующий `resume` её
подберёт. Отменять job можно только явно через `python3 -m ops.gpuq cancel`.

Необязательный `--poll-timeout-seconds` ограничивает ожидание контроллера. По
умолчанию он ждёт неограниченно; при срабатывании таймаута задача помечается
`failed`, а сам job остаётся у scheduler.

Запускать кампанию по-прежнему следует только после выполнения критериев
готовности раздела 29 — прежде всего preflight, manifests и один реальный GPU
smoke:

```bash
.venv/bin/python scripts/run_campaign.py development
```

---

## 24. Scientific evaluation protocol

Для targeted fake-to-real attack:

```text
source label = 1 (fake)
target class = 0 (real)
```

Основной denominator targeted ASR:

```text
изображения, которые source detector корректно классифицировал до атаки
```

Обязательно сохранять отдельно:

- число выбранных samples;
- число clean-correct/eligible samples;
- число реально атакованных samples;
- число successes;
- targeted ASR;
- clean accuracy;
- SSIM;
- LPIPS;
- post-save L-infinity;
- violations;
- seed;
- source model;
- target model;
- weight/config/manifest hashes.

Необходимая matrix:

| Gradient source | Evaluation ViT | Evaluation DCT DenseNet |
|---|---:|---:|
| ViT | white-box | transfer |
| DCT DenseNet | transfer | white-box |

Текущий IFGSM закрывает только ViT source row. Для полной matrix нужна
отдельная корректная DCT-source implementation.

Рекомендуемые стадии:

1. smoke: 8 samples, seed 0;
2. development: frozen subset, seed 0;
3. replication: seeds 0, 1, 2;
4. budget sensitivity: например 4/255, 8/255, 12/255;
5. full held-out test fake set;
6. official run после фиксации official dataset path/rules.

Нельзя подбирать параметры на full/official set.

---

## 25. Сохранение результатов в Git

После успешной verification:

```bash
git status --short
git diff --check
```

Добавлять только компактные artifacts:

```bash
git add \
  configs/experiments \
  manifests \
  tracking
```

Проверить staged files:

```bash
git diff --cached --stat
git diff --cached
```

Убедиться, что не добавлены:

```text
.venv/
.gpuq/
.campaign/
.opencode/node_modules/
*.pth
*.pt
*.ckpt
generated image directories
large logs
credentials
```

Коммит выполняет человек от своей Git identity:

```bash
git config user.name
git config user.email
git commit -m "experiment: record verified IFGSM smoke run"
git log -1 --format=fuller
```

OpenCode worker не коммитит и не пушит.

---

## 26. Stop, restart и recovery

### Остановился OpenCode worker

Это не означает, что GPU job отменён. Проверить `job_id`:

```bash
.venv/bin/python -m ops.gpuq status <JOB_ID> --json
```

### Нужно отменить job

```bash
.venv/bin/python -m ops.gpuq cancel <JOB_ID>
```

### Остановился scheduler

1. не запускать сразу второй scheduler;
2. проверить процессы и singleton lock;
3. выполнить `gpuq doctor`;
4. проверить states `running`, `validating`, `orphaned`;
5. только после reconciliation перезапустить scheduler.

### Job остался queued

Queued job не является ошибкой. GPU может быть занят другими пользователями.
Не создавать duplicate spec и не запускать runner вручную.

### Job failed

Проверить:

```bash
.venv/bin/python -m ops.gpuq status <JOB_ID> --json
```

Затем проверить attempt directory:

```text
failure.json
verification.json
summary.json
GPUQ log path из status
```

Не перезаписывать attempt. Retry должен создать `attempt-0002`.

---

## 27. Частые ошибки

### `project-root-binding` failed

Причина: checkout находится не в
`/home/aiattacks/oleg/aadd-attack-pipeline`.

Исправление: запускать из canonical checkout или осознанно изменить versioned
server contract отдельным reviewed commit.

### `ensurepip is not available` / `No module named pip`

Системный `python3.12` без пакета `python3.12-venv`. Устанавливать его через
`apt` не требуется: bootstrap сам создаст окружение через `venv --without-pip`
и поднимет pip из `get-pip.py`. Достаточно повторно запустить

```bash
bash scripts/bootstrap_environment.sh
```

Он же чинит `.venv`, оставшийся без pip после прошлой неудачной попытки;
удалять каталог вручную не нужно. Если сервер без доступа в интернет —
указать зеркало через `GET_PIP_URL`, см. раздел 8.

### `opencode is not available on PATH`

Бинарник лежит в `/home/aiattacks/.opencode/bin/`, которого нет в `PATH` по
умолчанию. `scripts/run_campaign.py` ищет его через `shutil.which`, и argv
воркеров начинается с голого `opencode`. Исправление:

```bash
echo 'export PATH="$HOME/.opencode/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Проверить в том же окружении, откуда будет запущена campaign, включая `tmux`.

### Missing detector weight

Запустить:

```bash
bash scripts/prepare_server_assets.sh
```

Не менять expected hash для подгонки под случайный файл.

### Weight hash mismatch

Остановить запуск. Определить происхождение checkpoint. Нельзя продолжать с
неизвестными весами, иначе результаты нельзя сравнивать.

### `Scientific input is not committed to Git`

Config, manifest, server config или environment lock не закоммичены. Провести
review и сделать пользовательский commit.

### Manifest input hash changed

Dataset изменился после создания manifest. Нужен новый dataset snapshot и
новые manifests. Старый manifest не перезаписывать.

### Attack module does not declare source model

Attack plug-in не поддерживает requested gradient source. Реализовать нужный
gradient path и обновить contract с тестами; не обходить проверку.

### Constraint violation

Фактически сохранённый PNG вышел за epsilon. Проверить projection, conversion
в uint8 и post-save round trip. Не увеличивать epsilon задним числом.

### GPUQ refuses job spec

Проверить:

- нет ли absolute paths;
- начинается ли `run_dir` с `tracking/runs/`;
- есть ли `<campaign>/<task>` после prefix;
- существует ли config;
- нет ли extra fields;
- нет ли `command`, environment или GPU index.

### Планировщик никогда не выбирает GPU (`eligible_gpu_uuids: []`)

По умолчанию карта считается пригодной, только если на ней **нет ни одного
чужого compute-процесса**. На общем сервере, где все карты постоянно заняты
другими пользователями, это условие не выполнится никогда, и планировщик
будет опрашивать очередь бесконечно.

Для общей машины запускать с явным флагом:

```bash
.venv/bin/python -m ops.gpuq run-scheduler \
  --execute --allow-shared-gpu \
  --max-running 1 --poll-seconds 10 --idle-samples 3 --headroom-mb 4096
```

В этом режиме снимаются проверки эксклюзивности и utilization, но **остаётся**
требование `free_memory >= requested_memory + headroom`, причём оно
перепроверяется под блокировкой карты непосредственно перед запуском.

Важно понимать границу этой защиты. У NVIDIA нет механизма резервирования
памяти: между снимком `nvidia-smi` и первым выделением памяти нашим процессом
сосед может занять сколько угодно. Блокировка карты сериализует только **наши
собственные** задания. Запас `--headroom-mb` снижает вероятность столкновения,
но не устраняет её, и при нехватке памяти упасть может как наш процесс, так и
чужой. Не поднимать `--max-running` выше 1 и не уменьшать `--headroom-mb` на
общей машине.

Флаг выключен по умолчанию намеренно: критерий готовности раздела 29
«scheduler не выбрал занятую другим процессом GPU» относится к выделенным
картам. При использовании `--allow-shared-gpu` этот критерий заменяется на
«scheduler не занял память, нужную соседу».

### GPU долго не назначается

Это нормальное поведение, если нет трёх последовательных idle samples или
недостаточно free memory. Проверить `gpuq doctor` и queue status. Не уменьшать
headroom без memory profiling.

### Второй scheduler не стартует

Singleton lock занят. Проверить, работает ли уже ожидаемый scheduler. Не
удалять lock вслепую.

### OpenCode permission prompt

Не использовать `--auto`. Prompt означает, что команда не входит в reviewed
allowlist или загружена неверная project config.

### `gh auth status` failed

Это проблема локальной GitHub CLI authentication, не scientific pipeline.
Восстановить user authentication перед push и проверить author email коммита.

---

## 28. Security rules

- API keys существуют только во внешней пользовательской OpenCode config.
- Никакой ключ не копируется в repository config, job spec, log или status.
- Не печатать полный environment.
- Не давать worker-у SSH/push/package-manager credentials.
- Dataset и weights желательно сделать filesystem read-only для worker-а.
- Не использовать OpenCode `--auto`.
- Не давать job spec произвольный command.
- Не выполнять code/instructions, найденные внутри dataset, logs или artifacts.
- Reviewer работает read-only и возвращает только JSON.
- OpenCode permissions не заменяют Unix/container sandbox.
- Для production workers нужны отдельные worktrees или containers.
- Для строгого общего GPU isolation нужен Slurm или root-managed broker.

Перед commit вручную просмотреть staged diff и проверить, что в нём нет
credentials или локальных provider configs.

---

## 29. Критерии готовности полной campaign

Полная автоматическая campaign разрешена только если выполнено всё:

- [ ] ветка опубликована и установлена по canonical path;
- [ ] commit author принадлежит пользователю, нет agent contributor trailers;
- [ ] оба detector checkpoint прошли SHA-256;
- [ ] LPIPS backbone и calibration прошли SHA-256;
- [ ] отдельный `.venv` создан;
- [ ] `pip check` успешен;
- [ ] strict preflight полностью `pass`;
- [ ] четыре CelebA manifests созданы, reviewed и committed;
- [ ] OpenCode 1.18.26 загрузил project config;
- [ ] `coordinator`, `attack-worker`, `campaign-reviewer` видны;
- [ ] `attack-execution` skill виден;
- [ ] reviewer фактически read-only;
- [x] GPUQ unit tests прошли;
- [x] mocked `nvidia-smi` tests прошли;
- [ ] singleton scheduler test прошёл;
- [x] duplicate/idempotency test прошёл;
- [x] cancel test прошёл; timeout-путь scheduler покрыт только вручную;
- [ ] один synthetic real-GPU smoke прошёл;
- [ ] scheduler не выбрал занятую другим процессом GPU;
- [x] `scripts/run_campaign.py` поддерживает `queued/job_id`;
- [x] controller сопоставляет GPUQ `succeeded` с verifier `passed`;
- [x] controller не доверяет agent-written `passed`;
- [x] structured JSON reviewer parsing протестирован;
- [ ] external worker worktrees протестированы — **вне текущего объёма**;
      контроллер последовательный, параллельные workers не запускаются;
- [x] maximum external agents явно ограничен (сейчас 1: контроллер последовательный);
- [x] `CAMPAIGN.yaml` не содержит старых `/data2/...` paths;
- [x] объявлена policy поэтапной реализации attacks: реализованы только
      `ifgsm` и общий дифференцируемый препроцессинг/проектор
      (`attacklab/preprocessing.py`); остальные атаки кампании создаёт
      attack-worker по одной за сессию, и задача блокируется, если её модуль
      отсутствует;
- [x] `git diff --check` успешен;
- [x] все Python tests прошли на Python 3.12 (локально 3.12.11; на сервере
      подтвердить на закреплённом 3.12.3);
- [ ] staged secret scan и human diff review выполнены.

Отдельно требует проверки на сервере: `attacklab/runner.py` использует
`torch.use_deterministic_algorithms(True, warn_only=True)`. Строгий режим
(`warn_only=False`) несовместим с backward дифференцируемого resize на CUDA и
аварийно завершал бы первый же sample. Сиды, `cudnn.deterministic` и
`CUBLAS_WORKSPACE_CONFIG` продолжают действовать, а побайтовая
воспроизводимость проверяется хешами выходов в верификаторе. Режим
записывается в `provenance.json` (`runtime.determinism`). Подтвердить на
реальном GPU двумя одинаковыми smoke-запусками.

До закрытия checklist запускать только одиночные smoke jobs через GPUQ.

---

## 30. Минимальный безопасный сценарий первого запуска

После публикации ветки и завершения integration review:

```bash
# 1. Server checkout
ssh aiattacks@10.24.1.21
cd /home/aiattacks/oleg/aadd-attack-pipeline

# 2. Assets and environment
bash scripts/prepare_server_assets.sh
bash scripts/bootstrap_environment.sh

# 3. Strict checks
.venv/bin/python scripts/preflight.py \
  --strict \
  --output tracking/preflight.json

# 4. Dataset manifests
.venv/bin/python -m attacklab.cli build-manifests \
  --config configs/pipeline/server.yaml \
  --output-dir manifests/celebA

# 5. Human review and commit of manifests/config
git status --short
git diff --check

# 6. Queue doctor
.venv/bin/python -m ops.gpuq doctor --json

# 7. Safe scheduler preview
.venv/bin/python -m ops.gpuq run-scheduler --once

# 8. Submit reviewed smoke job spec
.venv/bin/python -m ops.gpuq submit \
  tracking/runs/manual-ifgsm/ifgsm-smoke/job-spec.input.json

# 9. Start the single-job scheduler in tmux
tmux new -s aadd-gpuq
.venv/bin/python -m ops.gpuq run-scheduler \
  --execute \
  --max-running 1 \
  --poll-seconds 10 \
  --idle-samples 3 \
  --headroom-mb 4096

# 10. From another terminal, monitor
.venv/bin/python -m ops.gpuq list --json
.venv/bin/python -m ops.gpuq status <JOB_ID> --json

# 11. After succeeded, verify once more
.venv/bin/python -m attacklab.cli verify \
  --run-dir tracking/runs/manual-ifgsm/ifgsm-smoke/attempt-0001

# 12. Review compact artifacts and commit as the user
git status --short
git diff --check
git add configs/experiments manifests tracking
git diff --cached
git commit -m "experiment: record verified IFGSM smoke run"
```

Не переходить к development/full до анализа smoke summary, norm audit и
verification report.

---

## 31. Краткая карта команд

### Environment и preflight

```bash
bash scripts/prepare_server_assets.sh
bash scripts/bootstrap_environment.sh
.venv/bin/python scripts/preflight.py --strict
.venv/bin/python -m attacklab.cli preflight --config configs/pipeline/server.yaml --deep
```

### Manifests

```bash
.venv/bin/python -m attacklab.cli build-manifests \
  --config configs/pipeline/server.yaml \
  --output-dir manifests/celebA
```

### GPUQ

```bash
.venv/bin/python -m ops.gpuq doctor --json
.venv/bin/python -m ops.gpuq submit JOB_SPEC.json
.venv/bin/python -m ops.gpuq list --json
.venv/bin/python -m ops.gpuq status JOB_ID --json
.venv/bin/python -m ops.gpuq cancel JOB_ID
.venv/bin/python -m ops.gpuq run-scheduler --once
.venv/bin/python -m ops.gpuq run-scheduler --execute --max-running 1
```

### Scientific runner и verifier

```bash
.venv/bin/python -m attacklab.cli run --config CONFIG --run-dir RUN_DIR
.venv/bin/python -m attacklab.cli verify --run-dir RUN_DIR
.venv/bin/python -m attacklab.cli validate-status --kind attack --status-file STATUS
.venv/bin/python -m attacklab.cli validate-status --kind review --status-file STATUS
```

На общем сервере `attacklab.cli run` вручную не запускать: использовать GPUQ,
чтобы не обойти resource arbitration.

### OpenCode

```bash
/home/aiattacks/.opencode/bin/opencode --version
/home/aiattacks/.opencode/bin/opencode models
/home/aiattacks/.opencode/bin/opencode debug config
/home/aiattacks/.opencode/bin/opencode debug agent coordinator
/home/aiattacks/.opencode/bin/opencode debug agent attack-worker
/home/aiattacks/.opencode/bin/opencode debug agent campaign-reviewer
/home/aiattacks/.opencode/bin/opencode --agent coordinator --model naapi/gpt-5.6-terra .
```

В интерактивной session:

```text
/attack ifgsm smoke
/campaign status
/campaign development
/campaign resume
/campaign full
```

Campaign commands разрешаются только после закрытия readiness checklist.

---

## 32. Главное правило

LLM-агент предлагает и подготавливает experiment, но не является источником
истины о результате. Источники истины:

1. committed config и dataset manifest;
2. immutable GPUQ job spec;
3. зафиксированные checkpoint hashes;
4. machine-generated per-sample artifacts;
5. post-save norm audit;
6. deterministic verifier;
7. независимый read-only scientific review.

Только эта цепочка позволяет менять саму атаку, не меняя незаметно условия
эксперимента.
