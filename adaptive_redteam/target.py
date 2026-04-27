"""
Target model abstraction for red-team evaluation.

Wraps safety_probe.BaseBackend to provide a simplified one-call generate()
interface optimized for red-teaming. The adaptive loop only needs to send
prompts and get text back — it does not need to manage backend lifecycle,
batching, or config grids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from safety_probe.backends.base import BaseBackend, GenerationConfig


class TargetModel:
    """
    Thin wrapper over safety_probe.BaseBackend.

    Use as a context manager to manage backend lifecycle:

        with TargetModel.from_openai("gpt-4o-mini") as target:
            response = target.generate("Is 7×8=54?")

    Note: generate() and generate_batch() assume the backend is already
    loaded (i.e., called inside a `with` block). Calling outside a context
    manager will raise a RuntimeError from the underlying backend.
    """

    def __init__(self, backend: BaseBackend) -> None:
        self._backend = backend

    @classmethod
    def from_openai(
        cls,
        model_id: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        **kwargs: Any,
    ) -> TargetModel:
        """Convenience constructor for any OpenAI-compatible API."""
        import os

        from safety_probe.backends.openai_backend import OpenAIBackend
        api_key = os.environ.get(api_key_env, "")
        return cls(OpenAIBackend(model_id=model_id, base_url=base_url, api_key=api_key, **kwargs))

    def __enter__(self) -> TargetModel:
        self._backend.load()
        return self

    def __exit__(self, *_: Any) -> None:
        self._backend.unload()

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        """Generate a single response for a prompt."""
        from safety_probe.backends.base import GenerationConfig as GC
        cfg = config or GC(temperature=0.7)
        results = self._backend.generate([prompt], cfg)
        return str(results[0].response)

    def generate_batch(self, prompts: list[str], config: GenerationConfig | None = None) -> list[str]:
        """Generate responses for a batch of prompts."""
        from safety_probe.backends.base import GenerationConfig as GC
        cfg = config or GC(temperature=0.7)
        results = self._backend.generate(prompts, cfg)
        return [r.response for r in results]

    def generate_messages(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a response from a chat message history when the backend supports it."""
        from safety_probe.backends.base import GenerationConfig as GC

        cfg = config or GC(temperature=0.7)
        if hasattr(self._backend, "generate_messages"):
            result = self._backend.generate_messages(messages, cfg)
            return str(result.response)

        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        return self.generate(transcript, cfg)
