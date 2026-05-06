# Docker — vLLM with TurboQuant+ for AMD ROCm

Pre-built Docker image of vLLM with TurboQuant+ KV-cache compression patched in. Ships [`TheTom/vllm@feature/turboquant-amd-noautotune`](https://github.com/TheTom/vllm/tree/feature/turboquant-amd-noautotune) on top of the canonical `rocm/vllm-dev:base` image (ROCm 7.2 vLLM dev environment).

## Pull (when published)

```bash
docker pull thetom/vllm-turboquant:rocm-7.2
```

## Serve a model

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 8000:8000 \
  thetom/vllm-turboquant:rocm-7.2 \
    --model Qwen/Qwen2.5-14B-Instruct-1M \
    --kv-cache-dtype turboquant_k8v4
```

The default `--kv-cache-dtype` is `turboquant_k8v4` (K=8bit, V=4bit — the headline 62.5% savings asymmetric layout). Override with `auto` to disable TurboQuant+ for an apples-to-apples baseline.

**Layout cheatsheet** (TQ+ presets encode bit widths in the name):

| `--kv-cache-dtype` | K bits | V bits | symmetric? | savings vs FP16 |
| ------------------ | ------ | ------ | ---------- | --------------- |
| `auto` / FP16 | 16 | 16 | n/a | — |
| `turboquant_4bit_nc` | 4 | 4 | yes | 75% |
| `turboquant_k8v4` | 8 | 4 | no (asym) | **62.5%** (headline) |
| `turboquant_k8v3` | 8 | 3 | no | 65.6% |
| `turboquant_k4v3_nc` | 4 | 3 | no | 78% |
| `turboquant_3bit_nc` | 3 | 3 | yes | 81% |
| `*_rv` variants | — | — | rotated V | same savings, better PPL |

## Verify the KV savings before you serve

```bash
pip install tqkit
tq report --model qwen2.5-14b-instruct-1m --ctx 1M --layout tq+asym
# → 72 GB KV cache vs 192 GB FP16 (62.5% savings). Fits on a single MI300X.
```

## Build locally

```bash
cd docker
docker buildx build \
  --platform linux/amd64 \
  --build-arg VLLM_COMMIT=$(git ls-remote https://github.com/TheTom/vllm.git refs/heads/feature/turboquant-amd-noautotune | cut -f1) \
  -t thetom/vllm-turboquant:rocm-7.2 \
  -f Dockerfile.vllm-amd .
```

The image build pulls the `feature/turboquant-amd-noautotune` branch at HEAD by default; pin a specific commit via `--build-arg VLLM_COMMIT=<sha>` for reproducibility.

## Hardware

| GPU | gfx | tested |
| --- | --- | ------ |
| MI300X | gfx942 | yes (192 GB HBM3) |
| MI250X | gfx90a | not yet |
| Radeon W7900 | gfx1100 | not yet |

## CI

`.github/workflows/docker-vllm-amd.yml` builds the image on every `docker-vllm-amd-*` tag and on manual workflow dispatch. Push happens automatically when Docker Hub credentials are configured as repo secrets `DOCKERHUB_USERNAME` + `DOCKERHUB_TOKEN`.
