#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_ROOT="/home/aiattacks/oleg/deepfake-attacks/weights"
readonly TARGET_ROOT="/home/aiattacks/oleg/aadd-attack-assets/weights"
readonly VIT_SHA="5e9677d88a7af10791001796eb43d0d060fada3758369814d6d7832934758d81"
readonly DCT_SHA="5bbaf5c5c0e296d5e819a0b401198c73ad69c6bbc8f372579de5ee5c11d5e643"

install -d -m 0750 "${TARGET_ROOT}"

copy_verified() {
  local filename="$1"
  local expected_sha="$2"
  local source_path="${SOURCE_ROOT}/${filename}"
  local target_path="${TARGET_ROOT}/${filename}"

  if [[ ! -f "${source_path}" ]]; then
    echo "Missing audited source weight: ${source_path}" >&2
    return 2
  fi
  if [[ -e "${target_path}" ]]; then
    echo "Refusing to overwrite existing asset: ${target_path}" >&2
    return 2
  fi
  if [[ "$(sha256sum "${source_path}" | cut -d ' ' -f1)" != "${expected_sha}" ]]; then
    echo "Source hash mismatch: ${source_path}" >&2
    return 2
  fi
  cp --reflink=auto --preserve=mode,timestamps "${source_path}" "${target_path}"
  if [[ "$(sha256sum "${target_path}" | cut -d ' ' -f1)" != "${expected_sha}" ]]; then
    echo "Copied asset hash mismatch: ${target_path}" >&2
    return 2
  fi
}

copy_verified "vit_b_16.pth" "${VIT_SHA}"
copy_verified "densenet121_dct.pth" "${DCT_SHA}"
echo "Verified detector assets prepared at ${TARGET_ROOT}"
