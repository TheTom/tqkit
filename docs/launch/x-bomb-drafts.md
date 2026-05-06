# tqkit launch — X post drafts

Voice rules: lowercase, no em-dashes, "I" not "we", direct, data-first, no fluff.
DO NOT POST. Drafts only.

---

## Stage 1 — AMD 1M-context bomb (single tweet)

Drops AFTER:
- AMD KV reporter port lands in `tqkit`
- Docker image is published to ghcr.io with a real measured tok/s number
- `tq report --model qwen2.5-14b-instruct-1m --ctx 1M --layout tq+asym` returns clean numbers on the MI300X droplet

Required edits before posting: replace `<TOKS>` with the measured prefill or decode tok/s, replace `<IMG>` tag with the Docker pull line that actually works.

### Variant 1A — receipts-first

```
first 1M-token context on a single AMD GPU.

qwen2.5-14b, 1M ctx, mi300x.
fp16 kv cache: 192 gb. doesn't fit.
turboquant+ asym (k=q8_0, v=turbo4): 72 gb. fits with weights.
<TOKS> tok/s decode, +0.12 ppl vs fp16.

pip install tqkit
docker pull ghcr.io/thetom/tqkit-rocm:latest

verify the math yourself:
tq report --model qwen2.5-14b-instruct-1m --ctx 1M --layout tq+asym
```

### Variant 1B — short and punchy

```
1M tokens, 14b model, one mi300x.

fp16 kv = 192 gb, won't fit.
tq+ asym kv = 72 gb, fits, +0.12 ppl.
<TOKS> tok/s.

pip install tqkit
docker pull ghcr.io/thetom/tqkit-rocm

reproduce:
tq report --model qwen2.5-14b-instruct-1m --ctx 1M --layout tq+asym
```

### Variant 1C — thesis-first

```
14b at 1M context fits on a single gpu today. just not at fp16.

mi300x, qwen2.5-14b-1m, turboquant+ asym kv:
- kv cache: 72 gb (fp16 would be 192 gb, no gpu fits that)
- ppl delta: +0.12
- decode: <TOKS> tok/s

pip install tqkit
docker pull ghcr.io/thetom/tqkit-rocm
```

---

## Stage 2 — cross-platform unified table (3-tweet thread)

Hook: turboquant+ is a kv cache layout, not an architecture. ports per-engine in days. runs on every backend i benchmarked.

Implicit contrast with SubQ: closed-stack architecture moat vs open layout that drops into 5 engines.

### Variant 2A — single tweet (table compressed)

```
turboquant+ kv cache, 5 engines, same ~62.5% savings, vendor-agnostic:

llama.cpp:    fp16 ▸ tq+asym  (62.5% kv saved)
vllm cuda:    fp16 ▸ tq+asym  (62.5%)
vllm rocm:    fp16 ▸ tq+asym  (62.5%)
mlx-swift:    fp16 ▸ tq+asym  (62.5%)
vllm-swift:   fp16 ▸ tq+asym  (62.5%)

it's a layout, not an architecture.

pip install tqkit
github.com/TheTom/tqkit
```

### Variant 2B — 3-tweet thread

**tweet 1 / hook**
```
turboquant+ is a kv cache layout, not an architecture.

i ported it to 5 inference engines. same 62.5% kv savings on each. +0.05 to +0.20 ppl.

llama.cpp, vllm cuda, vllm rocm, mlx-swift, vllm-swift.

ports take days, not years.
```

**tweet 2 / table**
```
qwen2.5-7b, 32k ctx, kv cache size:

engine          fp16     tq+asym   saved
llama.cpp       16.0 gb   6.0 gb   62.5%
vllm cuda       16.0 gb   6.0 gb   62.5%
vllm rocm       16.0 gb   6.0 gb   62.5%
mlx-swift       16.0 gb   6.0 gb   62.5%
vllm-swift      16.0 gb   6.0 gb   62.5%

ppl delta: +0.05 to +0.20 across the matrix.
```

**tweet 3 / cta + jab**
```
the moat for long-context is supposed to be a custom subquadratic stack on closed weights. it isn't.

it's a kv layout on open weights, running on whatever silicon you already own.

pip install tqkit
github.com/TheTom/tqkit

reproduce every cell with: tq bench --layout tq+asym
```

### Variant 2C — single tweet, jab-forward

```
the long-context moat is supposed to be a custom architecture on closed weights.

it's not. it's a kv cache layout.

turboquant+ runs on llama.cpp, vllm (cuda + rocm), mlx-swift, vllm-swift. same 62.5% kv savings, +0.05 to +0.20 ppl, every engine.

pip install tqkit
github.com/TheTom/tqkit
```

---

## Pre-post checklist

- [ ] Stage 1: confirm `tq report` exits clean on MI300X droplet at 1M ctx
- [ ] Stage 1: replace `<TOKS>` with a real measured number, not a guess
- [ ] Stage 1: confirm `docker pull` line resolves on a clean machine (no auth)
- [ ] Stage 2: confirm all 5 engines produce a numeric row in `tq bench` at the same model/ctx
- [ ] Stage 2: confirm github.com/TheTom/tqkit README has the install + verify path before traffic lands
- [ ] both stages: char count under 280 per tweet (verify after any edit)
- [ ] both stages: lowercase, no em-dashes, "i" not "we" (re-scan after edits)
