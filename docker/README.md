# Docker — vLLM with TurboQuant+ for AMD ROCm

Pre-built Docker image of vLLM with TurboQuant+ KV-cache compression patched in. Ships [`TheTom/vllm@feature/turboquant-amd`](https://github.com/TheTom/vllm/tree/feature/turboquant-amd) on top of the canonical `rocm/vllm-dev:base_7.2` image.

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
    --kv-cache-dtype tq_asym
```

The default `--kv-cache-dtype` is `tq_asym` (K=q8_0, V=turbo4 — the headline 62.5% savings layout). Override with `fp16` to disable TurboQuant+ for an apples-to-apples baseline.

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
  --build-arg VLLM_COMMIT=$(git ls-remote https://github.com/TheTom/vllm.git refs/heads/feature/turboquant-amd | cut -f1) \
  -t thetom/vllm-turboquant:rocm-7.2 \
  -f Dockerfile.vllm-amd .
```

The image build pulls the `feature/turboquant-amd` branch at HEAD by default; pin a specific commit via `--build-arg VLLM_COMMIT=<sha>` for reproducibility.

## Hardware

| GPU | gfx | tested |
| --- | --- | ------ |
| MI300X | gfx942 | yes (192 GB HBM3) |
| MI250X | gfx90a | not yet |
| Radeon W7900 | gfx1100 | not yet |

## CI

`.github/workflows/docker-vllm-amd.yml` builds the image on every `docker-vllm-amd-*` tag and on manual workflow dispatch. Push happens automatically when Docker Hub credentials are configured as repo secrets `DOCKERHUB_USERNAME` + `DOCKERHUB_TOKEN`.
