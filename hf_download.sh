#!/usr/bin/env bash
set -euo pipefail

# If invoked via `sh hf_download.sh`, re-exec with bash to avoid parse errors.
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

# Download required Hugging Face models into this project under models/...
# Assumes `hf` CLI is installed and authenticated.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

HF_BIN="${HF_BIN:-$(command -v hf || true)}"
LOG_HEARTBEAT_SECONDS="${LOG_HEARTBEAT_SECONDS:-15}"

if [[ -z "$HF_BIN" ]]; then
  echo "Error: Hugging Face CLI (hf) is not installed or not in PATH."
  echo "Install with: pip install -U huggingface_hub"
  exit 1
fi

# Work around occasional Xet backend crashes by forcing regular HTTP downloads.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

ensure_hf_auth() {
  if ! "$HF_BIN" auth whoami >/dev/null 2>&1; then
    echo "Error: hf CLI is not authenticated in this shell context."
    echo "Run: hf auth login"
    exit 1
  fi
}

mkdir -p models/flux/FLUX.1-dev
mkdir -p models/google/siglip-so400m-patch14-384
mkdir -p models/flux_ipa

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

write_manifest_for_dir() {
  local dir="$1"
  local manifest="$2"
  : > "$manifest"

  while IFS= read -r file; do
    local rel
    rel="${file#"$dir"/}"
    printf '%s\t%s\n' "$(sha256_file "$file")" "$rel" >> "$manifest"
  done < <(find "$dir" -type f ! -name "$(basename "$manifest")" ! -name "*.incomplete" | LC_ALL=C sort)
}

verify_manifest_for_dir() {
  local dir="$1"
  local manifest="$2"

  [[ -f "$manifest" ]] || return 1

  local lines
  lines="$(wc -l < "$manifest" | tr -d ' ')"
  [[ "$lines" -gt 0 ]] || return 1

  log "Verifying checksum manifest for $dir ($lines files)..."

  local idx=0

  while IFS=$'\t' read -r expected rel; do
    idx=$((idx + 1))
    if (( idx == 1 || idx % 5 == 0 || idx == lines )); then
      log "Verify progress: $idx/$lines ($rel)"
    fi

    [[ -n "$expected" && -n "$rel" ]] || return 1
    local file="$dir/$rel"
    [[ -f "$file" ]] || return 1
    local actual
    actual="$(sha256_file "$file")"
    [[ "$actual" == "$expected" ]] || return 1
  done < "$manifest"

  log "Checksum verification passed for $dir"
}

write_manifest_for_file() {
  local file="$1"
  local manifest="$2"
  printf '%s\t%s\n' "$(sha256_file "$file")" "$(basename "$file")" > "$manifest"
}

verify_manifest_for_file() {
  local file="$1"
  local manifest="$2"

  [[ -f "$file" && -f "$manifest" ]] || return 1

  local expected rel actual
  IFS=$'\t' read -r expected rel < "$manifest" || return 1
  [[ -n "$expected" ]] || return 1
  actual="$(sha256_file "$file")"
  [[ "$actual" == "$expected" ]]
}

download_if_needed_dir() {
  local repo_id="$1"
  local local_dir="$2"
  local manifest="$local_dir/.hf_download.sha256"

  if verify_manifest_for_dir "$local_dir" "$manifest"; then
    echo "Skip: $repo_id already present and checksum-verified."
    return
  fi

  ensure_hf_auth
  log "Downloading: $repo_id"

  (
    while true; do
      sleep "$LOG_HEARTBEAT_SECONDS"
      log "Still downloading $repo_id ..."
    done
  ) &
  local heartbeat_pid=$!

  set +e
  "$HF_BIN" download "$repo_id" \
    --repo-type model \
    --local-dir "$local_dir"
  local download_rc=$?
  set -e

  kill "$heartbeat_pid" >/dev/null 2>&1 || true
  wait "$heartbeat_pid" 2>/dev/null || true

  if [[ "$download_rc" -ne 0 ]]; then
    log "Download failed for $repo_id"
    exit "$download_rc"
  fi

  write_manifest_for_dir "$local_dir" "$manifest"
  log "Download complete and manifest updated for $repo_id"
}

download_if_needed_file() {
  local repo_id="$1"
  local filename="$2"
  local local_dir="$3"
  local local_file="$local_dir/$filename"
  local manifest="$local_dir/.${filename}.sha256"

  if verify_manifest_for_file "$local_file" "$manifest"; then
    echo "Skip: $repo_id/$filename already present and checksum-verified."
    return
  fi

  ensure_hf_auth
  log "Downloading: $repo_id/$filename"

  (
    while true; do
      sleep "$LOG_HEARTBEAT_SECONDS"
      log "Still downloading $repo_id/$filename ..."
    done
  ) &
  local heartbeat_pid=$!

  set +e
  "$HF_BIN" download "$repo_id" \
    --repo-type model \
    "$filename" \
    --local-dir "$local_dir"
  local download_rc=$?
  set -e

  kill "$heartbeat_pid" >/dev/null 2>&1 || true
  wait "$heartbeat_pid" 2>/dev/null || true

  if [[ "$download_rc" -ne 0 ]]; then
    log "Download failed for $repo_id/$filename"
    exit "$download_rc"
  fi

  write_manifest_for_file "$local_file" "$manifest"
  log "Download complete and manifest updated for $repo_id/$filename"
}

# FLUX base model (gated)
#download_if_needed_dir "black-forest-labs/FLUX.1-dev" "models/flux/FLUX.1-dev"

# SigLIP image encoder
#download_if_needed_dir "google/siglip-so400m-patch14-384" "models/google/siglip-so400m-patch14-384"

# Kijai IP-Adapter binary used by this project
download_if_needed_file "Kijai/FLUX.1-dev-IP-Adapter-safetensors" "instantx_flux1_dev_ip_adapter_bf16.safetensors"

log "Done. Required model files are available under models/..."
