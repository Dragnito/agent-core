"""Interfaz que todo proveedor de LLM debe cumplir."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agent_core.types import LLMResponse, Message


@runtime_checkable
class LLMProvider(Protocol):
    """Cualquier objeto con este método `complete` es un proveedor válido.

    No hace falta heredar de esta clase (tipado estructural).
    `runtime_checkable` permite `isinstance(x, LLMProvider)`, pero solo
    verifica que el método exista, no su firma.
    """

    def complete(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> LLMResponse: ...
