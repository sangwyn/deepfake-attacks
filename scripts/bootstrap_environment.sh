#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_PATH="${PROJECT_ROOT}/.venv"
readonly PYTHON_BIN="${PYTHON_BIN:-python3.12}"

actual_python="$(${PYTHON_BIN} -c 'import platform; print(platform.python_version())')"
if [[ "${actual_python}" != "3.12.3" ]]; then
  echo "Expected Python 3.12.3, found ${actual_python} via ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi

"${VENV_PATH}/bin/python" -m pip install --upgrade \
  "pip==26.2.1" "setuptools==75.8.2" "wheel==0.45.1"
"${VENV_PATH}/bin/python" -m pip install --requirement \
  "${PROJECT_ROOT}/requirements.lock"
"${VENV_PATH}/bin/python" -m pip install --no-deps --editable "${PROJECT_ROOT}"
"${VENV_PATH}/bin/python" -m attacklab.cli preflight \
  --config "${PROJECT_ROOT}/configs/pipeline/server.yaml" --deep
