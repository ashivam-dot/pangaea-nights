#!/usr/bin/env bash
# Use hosted-runner tools first. Refresh only Ubuntu's package index, with bounded retries.
set -euo pipefail
if [ "$#" -eq 0 ]; then exit 2; fi
missing=()
for package in "$@"; do
  case "$package" in
    ffmpeg) commands=(ffmpeg ffprobe) ;;
    tesseract-ocr) commands=(tesseract) ;;
    espeak-ng) commands=(espeak-ng) ;;
    *) echo "Unsupported media package: $package" >&2; exit 2 ;;
  esac
  for command in "${commands[@]}"; do
    if ! command -v "$command" >/dev/null 2>&1; then
      missing+=("$package")
      break
    fi
  done
done
if [ "${#missing[@]}" -eq 0 ]; then exit 0; fi
apt_options=(-o Acquire::Retries=1 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20)
if timeout 180 sudo apt-get "${apt_options[@]}" install -y --no-install-recommends "${missing[@]}"; then
  exit 0
fi
source_options=()
if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
  source_options=(-o Dir::Etc::sourcelist=/etc/apt/sources.list.d/ubuntu.sources -o Dir::Etc::sourceparts=-)
fi
timeout 180 sudo apt-get "${apt_options[@]}" "${source_options[@]}" update
timeout 180 sudo apt-get "${apt_options[@]}" install -y --no-install-recommends "${missing[@]}"
