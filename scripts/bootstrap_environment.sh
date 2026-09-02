#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_PATH="${PROJECT_ROOT}/.venv"
readonly PYTHON_BIN="${PYTHON_BIN:-python3.12}"
readonly GET_PIP_URL="${GET_PIP_URL:-https://bootstrap.pypa.io/get-pip.py}"
readonly PIP_PIN="pip==26.2.1"
readonly SETUPTOOLS_PIN="setuptools==75.8.2"
readonly WHEEL_PIN="wheel==0.45.1"

actual_python="$(${PYTHON_BIN} -c 'import platform; print(platform.python_version())')"
if [[ "${actual_python}" != "3.12.3" ]]; then
  echo "Expected Python 3.12.3, found ${actual_python} via ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  if "${PYTHON_BIN}" -c 'import ensurepip' >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m venv "${VENV_PATH}"
  else
    # Debian and Ubuntu ship the interpreter without ensurepip; it lives in a
    # separate python3.12-venv package that needs root to install. Build the
    # environment without pip and bootstrap pip below instead, so a first
    # install never requires root.
    echo "ensurepip is unavailable; creating the environment without pip." >&2
    "${PYTHON_BIN}" -m venv --without-pip "${VENV_PATH}"
  fi
fi

# Also repairs an environment a previous run left without pip.
if ! "${VENV_PATH}/bin/python" -m pip --version >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    download() { curl --proto '=https' --tlsv1.2 --fail --silent --show-error \
      --location "$1" --output "$2"; }
  elif command -v wget >/dev/null 2>&1; then
    download() { wget --https-only --quiet --output-document "$2" "$1"; }
  else
    echo "Neither curl nor wget is available to bootstrap pip" >&2
    exit 2
  fi
  echo "Bootstrapping pip from ${GET_PIP_URL}" >&2
  get_pip="$(mktemp)"
  trap 'rm -f "${get_pip}"' EXIT
  download "${GET_PIP_URL}" "${get_pip}"
  "${VENV_PATH}/bin/python" "${get_pip}" \
    "${PIP_PIN}" "${SETUPTOOLS_PIN}" "${WHEEL_PIN}"
fi

"${VENV_PATH}/bin/python" -m pip install --upgrade \
  "${PIP_PIN}" "${SETUPTOOLS_PIN}" "${WHEEL_PIN}"
"${VENV_PATH}/bin/python" -m pip install --requirement \
  "${PROJECT_ROOT}/requirements.lock"
"${VENV_PATH}/bin/python" -m pip install --no-deps --editable "${PROJECT_ROOT}"
"${VENV_PATH}/bin/python" -m attacklab.cli preflight \
  --config "${PROJECT_ROOT}/configs/pipeline/server.yaml" --deep
