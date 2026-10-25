#!/usr/bin/env bash
# Use hosted tools first; recover from a stalled runner mirror using Ubuntu's signed archive.
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
media_ready() {
  local package tool
  for package in "$@"; do
    case "$package" in
      ffmpeg) tools=(ffmpeg ffprobe) ;;
      tesseract-ocr) tools=(tesseract) ;;
      espeak-ng) tools=(espeak-ng) ;;
    esac
    for tool in "${tools[@]}"; do command -v "$tool" >/dev/null 2>&1 || return 1; done
  done
}
apt_options=(-o Acquire::Retries=1 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20
             -o DPkg::Lock::Timeout=30)
if timeout 180 sudo apt-get "${apt_options[@]}" install -y --no-install-recommends "${missing[@]}"; then
  if media_ready "$@"; then exit 0; fi
fi
# Refreshing the runner's index still selects its same slow mirror. Use one temporary
# source list for BOTH update and install, without changing the runner's system sources.
[ "$(lsb_release -is)" = Ubuntu ] || { echo "Archive fallback requires Ubuntu" >&2; exit 1; }
codename=$(lsb_release -cs)
[[ "$codename" =~ ^[a-z]+$ ]] || exit 1
apt_dir=$(mktemp -d)
trap 'rm -rf "$apt_dir"' EXIT
chmod 755 "$apt_dir"
cat > "$apt_dir/ubuntu.list" <<EOF
deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://archive.ubuntu.com/ubuntu $codename main restricted universe multiverse
deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://archive.ubuntu.com/ubuntu $codename-updates main restricted universe multiverse
deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://security.ubuntu.com/ubuntu $codename-security main restricted universe multiverse
EOF
source_options=(-o "Dir::Etc::sourcelist=$apt_dir/ubuntu.list" -o Dir::Etc::sourceparts=-)
timeout 180 sudo apt-get "${apt_options[@]}" "${source_options[@]}" update
timeout 300 sudo apt-get "${apt_options[@]}" "${source_options[@]}" install -y --no-install-recommends "${missing[@]}"
media_ready "$@" || { echo "Media tools are still unavailable after installation" >&2; exit 1; }
