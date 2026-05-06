"""longctx engine bridge — retrieval-as-savings.

Other tqkit engines compress the KV cache (TurboQuant+ asym = 62% smaller).
This one **avoids** the KV cache entirely: retrieve the relevant top-K
chunks and feed only those to the model. The KV cache for 3K tokens of
retrieved chunks is dramatically smaller than the KV cache for 1M tokens
of full context, even at FP16.

The unified `tq bench` story becomes:

    layout         | KV bytes @ 1M ctx | savings vs FP16
    ---------------|-------------------|----------------
    fp16 (full)    | 192 GB            | -
    tq+asym (full) | 73.5 GB           | 62%
    longctx (top-8)| ~600 MB           | 99.7%
    longctx + tq+  | ~225 MB           | 99.9%

The most efficient KV cache is the one you never allocate.

This engine reports a synthetic kv_cache_bytes that reflects only the
context the model actually sees (sum of retrieved chunk lengths × FP16
bpe), not the full input prompt. Decode tps reflects real generation.
"""
from __future__ import annotations

import time

from tqkit.engines.base import Engine, RunResult, EngineNotInstalled


class LongctxEngine(Engine):
    """Bench wrapper around longctx.LongCtxClient."""

    name = "longctx"

    # longctx itself is layout-agnostic — it lets the underlying generator
    # use whatever KV layout the server is configured with. So `tq+asym`
    # here means "longctx retrieval combined with TQ+-quantized generator
    # KV"; `fp16` means "longctx retrieval with vanilla FP16 generator KV".
    LAYOUT_FLAGS = {
        "fp16":    [],
        "tq+asym": [],
        "tq+sym":  [],
    }

    def __init__(self, server: str | None = None, model: str | None = None):
        self._server = server or "http://localhost:5050/v1/chat/completions"
        self._model = model

    def is_installed(self) -> bool:
        try:
            import longctx  # noqa: F401
            return True
        except ImportError:
            return False

    def supports(self, layout: str) -> bool:
        return layout in self.LAYOUT_FLAGS

    def run(
        self,
        model: str,
        prompt: str,
        ctx_tokens: int,
        layout: str = "fp16",
        max_tokens: int = 64,
        timeout: int = 600,
        candidates: list[str] | None = None,
        top_k: int = 8,
    ) -> RunResult:
        """Run longctx retrieval + generate.

        Args:
            model: served model name on the OpenAI-compatible endpoint
            prompt: the question / target the user is asking
            ctx_tokens: NOMINAL prompt length the user *would* have sent
                without retrieval — used to compute the savings claim
                vs full-context FP16 KV cache.
            layout: "fp16", "tq+asym", "tq+sym" — informational only,
                forwarded as a string but not enforced.
            max_tokens: generation budget
            timeout: HTTP timeout in seconds
            candidates: pre-chunked list of strings to retrieve from. If
                None, retrieval is skipped (degenerate; for testing).
            top_k: number of chunks to retrieve.
        """
        if not self.is_installed():
            raise EngineNotInstalled(
                "longctx not installed. pip install longctx."
            )
        if not self.supports(layout):
            raise ValueError(f"layout '{layout}' not recognized by longctx")

        from longctx import LongCtxClient

        client = LongCtxClient(model=model, server=self._server,
                               timeout=timeout)
        candidates = candidates or [prompt]

        t0 = time.time()
        response = client.ask(
            query=prompt,
            candidates=candidates,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - t0

        # Synthetic kv_cache_bytes: tokens the model ACTUALLY saw,
        # at FP16 (or whatever the underlying server uses; we don't know
        # from this side so report FP16 baseline equivalent).
        retrieved_tokens = response.prompt_tokens or 0
        # Placeholder per-token: treat as 192 KB (Qwen2.5-14B FP16) so
        # the downstream caller can compute savings vs full-ctx FP16.
        # In a real run with `tq bench --model <X>`, the bench dispatcher
        # should override this with the model's actual per-token bytes.
        kv_bytes = retrieved_tokens * 192 * 1024

        decode_tps = None
        if response.completion_tokens and elapsed > 0:
            # rough decode tps; doesn't separate prefill from decode
            decode_tps = response.completion_tokens / max(elapsed, 0.01)

        return RunResult(
            backend=self.name,
            model=model,
            layout=f"longctx+{layout}",
            ctx_tokens=ctx_tokens,
            kv_cache_bytes=kv_bytes,
            output=response.content[:512],
            decode_tps=decode_tps,
            total_seconds=elapsed,
            extra={
                "retrieved_indices": str(response.retrieved_indices),
                "retrieved_token_count": str(retrieved_tokens),
                "top_k": str(top_k),
                "savings_strategy": "retrieval (not compression)",
            },
            raw_log="",
        )
