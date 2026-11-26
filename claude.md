# agent-core

Framework mínimo de agentes en Python 3.11+, hecho para aprender cómo funcionan por dentro.
Stack: src/ layout, pytest, httpx. Sin LangChain ni SDKs de proveedores.

## Reglas
- No agregar dependencias sin preguntar.
- Todo cambio incluye tests que corren SIN internet.
- Tipado en todas las funciones públicas.
- Explicar brevemente cada decisión de diseño al proponer cambios.
- El loop del agente (src/agent_core/agent.py) lo debe entender el humano, antes de crearlo debes explicarlo detalladamente.
- Nunca escribir claves o secretos en el código. Se leen desde variables de entorno (.env).

## Contrato (no cambiar sin preguntar)
- `ToolCall(id: str, name: str, arguments: dict)`
- `Message(role: "system"|"user"|"assistant" |"tool", content: str = "", tool_calls: list[ToolCall] = [], tool_call_id: str | None = None)`
- `LLMResponse(text: str, tool_calls: list[ToolCall], input_tokens: int, output_tokens: int)`
- `LLMProvider` (Protocol): `complete(messages: list[Message], tools: list[dict]) -> LLMResponse`