"""Engine base class + RunResult + exceptions."""
from __future__ import annotations

from dataclasses import dataclass, field


class EngineNotInstalled(RuntimeError):
    """Raised when an engine binary or Python module isn't on the system."""


@dataclass
class RunResult:
    """Normalized result from one engine run.

    Every engine emits these fields by parsing its own log output. Fields
    that the engine doesn't expose are left None; downstream code (the
    bench printer) must handle missing values gracefully.
    """
    backend: str
    model: str
    layout: str
    ctx_tokens: int

    # Memory & cache (the headline numbers)
    kv_cache_bytes: int | None = None      # total, what fp16 vs tq+ math hangs on
    peak_memory_bytes: int | None = None   # peak GPU/RAM observed

    # Throughput
    prefill_tps: float | None = None       # tokens/sec during prefill
    decode_tps: float | None = None        # tokens/sec during decode
    total_seconds: float | None = None     # wall-clock end-to-end

    # Quality
    output: str | None = None              # generated text (truncated)
    ppl_wikitext2: float | None = None     # if separately measured

    # Diagnostics
    raw_log: str = ""                      # full engine stderr/stdout for debugging
    extra: dict[str, str] = field(default_factory=dict)


class Engine:
    """Abstract base for an inference engine that supports TurboQuant+.

    Subclasses must define `name` and implement `run()`. The run() method
    is responsible for: locating the binary, choosing flags for the
    requested layout, executing with the prompt, and parsing the engine's
    output into a RunResult.
    """

    name: str = "abstract"

    # Map from canonical layout name (matches tqkit.kv_math.LAYOUT_BYTES_PER_ELEM)
    # to engine-specific flags. Subclasses override.
    LAYOUT_FLAGS: dict[str, list[str]] = {}

    def supports(self, layout: str) -> bool:
        return layout in self.LAYOUT_FLAGS

    def is_installed(self) -> bool:
        """Cheap check: can we run anything? Override in subclasses."""
        return False

    def run(
        self,
        model: str,
        prompt: str,
        ctx_tokens: int,
        layout: str = "fp16",
        max_tokens: int = 64,
        timeout: int = 600,
    ) -> RunResult:
        raise NotImplementedError
