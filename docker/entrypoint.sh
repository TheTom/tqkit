#!/usr/bin/env bash
# tqkit / vLLM-AMD container entrypoint.
#
# Prints the build identity + a quick KV-cache savings report, then
# delegates to vllm serve. Invoked as the Docker ENTRYPOINT so users
# always see the receipts before the model loads.

set -euo pipefail

vllm_commit="$(cat /workspace/.vllm-commit 2>/dev/null || echo unknown)"
echo "============================================================"
echo " vllm-turboquant-amd"
echo " vllm commit: ${vllm_commit}"
echo " tqkit:       $(python3 -c 'import tqkit;print(tqkit.__version__)' 2>/dev/null || echo unknown)"
echo " ROCm:        $(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
echo "============================================================"

# If the user passes --help or --version, just defer to vllm.
if [[ "${1:-}" == "--help" || "${1:-}" == "--version" || $# -eq 0 ]]; then
  exec vllm serve "$@"
fi

# Surface a default KV-cache-dtype if the user didn't pass one.
has_kv_flag=0
for arg in "$@"; do
  case "$arg" in
    --kv-cache-dtype|--kv-cache-dtype=*) has_kv_flag=1 ;;
  esac
done

if [[ $has_kv_flag -eq 0 ]]; then
  echo "[entrypoint] no --kv-cache-dtype passed; defaulting to turboquant_k8v4"
  echo "[entrypoint]   (TurboQuant+ asymmetric, K=8bit V=4bit, 62.5% KV savings)"
  echo "[entrypoint] override with --kv-cache-dtype auto to disable TurboQuant+"
  set -- "$@" --kv-cache-dtype turboquant_k8v4
fi

exec vllm serve "$@"
