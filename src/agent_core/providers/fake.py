"""Proveedor falso para tests: devuelve respuestas predefinidas, sin red."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

from agent_core.types import LLMResponse, Message


class FakeProviderExhaustedError(RuntimeError):
    """Se llamó a `complete` más veces que respuestas configuradas."""


class FakeProvider:
    """Devuelve una `LLMResponse` por llamada, en orden (FIFO).

    Guarda en `calls` una copia de los mensajes recibidos en cada llamada,
    incluida la llamada que falla por falta de respuestas.
    """

    def __init__(self, responses: Iterable[LLMResponse]) -> None:
        self._responses: deque[LLMResponse] = deque(responses)
        self._configured = len(self._responses)
        self.calls: list[list[Message]] = []

    @property
    def remaining(self) -> int:
        """Cuántas respuestas quedan por devolver."""
        return len(self._responses)

    def complete(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        # Copia: si el llamador sigue modificando su lista, el registro no cambia.
        self.calls.append(list(messages))
        if not self._responses:
            raise FakeProviderExhaustedError(
                f"FakeProvider: llamada #{len(self.calls)} pero solo se "
                f"configuraron {self._configured} respuestas"
            )
        return self._responses.popleft()
