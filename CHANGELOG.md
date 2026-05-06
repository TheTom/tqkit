# Changelog

## v0.3.0 (2026-05-06) — engine bridges + Qwen3.6 + correct vLLM TQ+ presets

### Added

- **`tq bench`**: dispatches the canonical workload across detected backends, parses each engine's output, emits a unified markdown table (KV cache size, decode tps per backend × layout).
- **`tqkit/engines/`**: 5 engine bridges sharing a common `Engine` base + `RunResult` dataclass: `LlamaCppEngine`, `VllmCudaEngine`, `VllmAmdEngine`, `MlxSwiftEngine`, `VllmSwiftEngine`.
- **`tq config`**: prints the pinned canonical_bench.yml so anyone can compare their runs to the reference table.
- **Models**: Qwen3.6-27B (16 full-attention / 64 total layers, hybrid linear+full, head_dim=256) and Qwen3.6-35B-A3B (10 full-attention / 40 total, MoE 256 experts).
- **`ModelArch.notes`**: free-form annotation flagging hybrid architectures so readers don't double-count linear-attention layers in KV math.
- **Docker**: `docker/Dockerfile.vllm-amd` + `entrypoint.sh` for `thetom/vllm-turboquant` ROCm image.
- **Integration docs**: `docs/integrate/{llama-cpp,vllm-cuda,vllm-amd,mlx-swift,vllm-swift}.md` — one-page recipe per backend.

### Fixed

- **vLLM `--kv-cache-dtype` preset names**: previous docs/code used placeholder names `tq_asym` / `tq_sym` that don't exist in the actual TheTom/vllm fork. Real names are bit-explicit: `turboquant_k8v4` (asym, K=8bit V=4bit, headline 62.5% savings) and `turboquant_4bit_nc` (symmetric 4-bit, no centroid). Full preset list in `docs/integrate/vllm-cuda.md`.
- **AMD branch name**: `feature/turboquant-amd-noautotune` is the real branch on TheTom/vllm; the previously-referenced `feature/turboquant-amd` does not exist.
- **Docker base tag**: `rocm/vllm-dev:base` (the actual canonical tag); `base_7.2` was a guess and doesn't exist on Docker Hub.
- **MLX-Swift recommendation**: route via TheTom's alpha branch (production-stable, drives vllm-swift), not Eric's upstream alpha.

### Test coverage

39 tests, 92% line coverage. Engine subprocess parsing covered with mocked llama.cpp + vllm + mlx + vllm-swift sample logs.

---

## v0.2.1 (2026-05-06) — `tq config` + canonical_bench.yml

- Pinned canonical benchmark config: model HF revision + prompt sha256 + per-backend per-layout expected results. The contract for `tq bench` reproducibility.
- `tq config` prints the shipped YAML.
- 18 tests, 89% coverage, 85% gate enforced via `pyproject.toml`.

---

## v0.2.0 (2026-05-06) — `tq report` + `tq table` (theoretical KV math)

- `tqkit/kv_math.py`: pinned `ModelArch` for Qwen2.5 7B/14B/32B, Qwen3-8B, Qwen3-Next-80B-A3B, Llama-3.1 8B/70B, Mistral-7B.
- `LAYOUT_BYTES_PER_ELEM`: 6 layouts (fp16, fp8, q8_0, tq+sym, tq+asym, turbo4) parametrize the KV cache math.
- `tq report --model X --ctx 32K --layout tq+asym`: per-token + total KV bytes, savings vs FP16.
- `tq table --model X`: full layout × ctx grid in markdown.
- Honest math: K=q8_0 + V=turbo4 = 62.5% savings vs FP16. (The widely-quoted 70%+ figure conflates aggressive 4-bit symmetric with K=8bit asymmetric.)

---

## v0.1.0 (2026-05-06) — initial release

- `tq backends`: autodetects llama.cpp, vLLM, MLX, vllm-swift on the host.
- `tq integrate <backend>`: prints install + serve recipe.
- 16 tests, 90% coverage.
