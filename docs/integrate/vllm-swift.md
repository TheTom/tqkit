# Integrating TurboQuant+ into vllm-swift

[vllm-swift](https://github.com/TheTom/vllm-swift) is the native Swift/Metal vLLM backend for Apple Silicon. It speaks the OpenAI Chat Completions API and serves any model that mlx-swift-lm can load. Backed by TheTom's [mlx-swift-lm@alpha](https://github.com/TheTom/mlx-swift-lm/tree/alpha) — the production-stable TurboQuant+ branch.

## Install

```bash
brew install thetom/vllm-swift/vllm-swift
```

Or build from source:

```bash
git clone https://github.com/TheTom/vllm-swift
cd vllm-swift
swift build -c release
```

## Serve with TurboQuant+ KV cache

```bash
vllm-swift serve \
    --model qwen2.5-14b-instruct-1m-4bit \
    --kv-layout tq+asym \
    --port 8000
```

### Layout flags

| `--kv-layout` | savings vs FP16 |
| ------------- | --------------- |
| `fp16` (default) | — |
| `tq+sym` | 50% |
| `tq+asym` | 62.5% (headline) |

## Verify the savings

```bash
pip install tqkit
tq report --model qwen2.5-14b-instruct-1m --ctx 32K --layout tq+asym
```

## Use it with your existing OpenAI-style code

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="anything",
)

response = client.chat.completions.create(
    model="qwen2.5-14b-instruct-1m-4bit",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Zero code change. Your existing scripts, agents (Cline, Aider, OpenCode), and SDKs all "just work" — `vllm-swift` is OpenAI-API-compatible, and the TurboQuant+ KV savings happen transparently inside the server.

## Caveats

- `--kv-layout tq+asym` flag is provided by the in-progress port (task #118). Until then, vllm-swift uses MLX-Swift's default cache.
- Apple Silicon only (M1 through M5, A18+). Not available on Intel Macs.
- For maximum throughput at long context, also pass `--max-num-seqs N` to keep the GPU fed; KV savings let you push N higher.
