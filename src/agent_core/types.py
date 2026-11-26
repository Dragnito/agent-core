"""Tipos del contrato de agent-core.

Ver la sección "Contrato" de claude.md: no cambiar sin preguntar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, get_args

Role = Literal["system", "user", "assistant", "tool"]

VALID_ROLES: frozenset[str] = frozenset(get_args(Role))


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Pedido del LLM para ejecutar una herramienta."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Message:
    """Un mensaje de la conversación.

    `tool_calls` usa default_factory para que cada instancia tenga su propia
    lista (con `= []` todas compartirían la misma).
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        # Literal no se valida en ejecución; esto atrapa typos como "usr".
        if self.role not in VALID_ROLES:
            raise ValueError(
                f"role inválido: {self.role!r}. Debe ser uno de {sorted(VALID_ROLES)}"
            )


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Respuesta de un proveedor de LLM."""

    text: str
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int
