from dataclasses import FrozenInstanceError

import pytest

from agent_core.types import LLMResponse, Message, ToolCall


def test_tool_call_fields() -> None:
    tc = ToolCall(id="call_1", name="sumar", arguments={"a": 1, "b": 2})
    assert tc.id == "call_1"
    assert tc.name == "sumar"
    assert tc.arguments == {"a": 1, "b": 2}


def test_message_defaults() -> None:
    msg = Message(role="user")
    assert msg.content == ""
    assert msg.tool_calls == []
    assert msg.tool_call_id is None


def test_message_tool_calls_not_shared_between_instances() -> None:
    a = Message(role="assistant")
    b = Message(role="assistant")
    assert a.tool_calls is not b.tool_calls
    a.tool_calls.append(ToolCall(id="1", name="x", arguments={}))
    assert b.tool_calls == []


@pytest.mark.parametrize("role", ["system", "user", "assistant", "tool"])
def test_message_accepts_valid_roles(role: str) -> None:
    assert Message(role=role).role == role  # type: ignore[arg-type]


def test_message_rejects_invalid_role() -> None:
    with pytest.raises(ValueError, match="usr"):
        Message(role="usr")  # type: ignore[arg-type]


def test_message_is_frozen() -> None:
    msg = Message(role="user", content="hola")
    with pytest.raises(FrozenInstanceError):
        msg.content = "chau"  # type: ignore[misc]


def test_message_equality_by_value() -> None:
    assert Message(role="user", content="hola") == Message(role="user", content="hola")


def test_llm_response_fields() -> None:
    tc = ToolCall(id="1", name="x", arguments={})
    resp = LLMResponse(text="ok", tool_calls=[tc], input_tokens=10, output_tokens=5)
    assert resp.text == "ok"
    assert resp.tool_calls == [tc]
    assert resp.input_tokens == 10
    assert resp.output_tokens == 5


def test_llm_response_is_frozen() -> None:
    resp = LLMResponse(text="ok", tool_calls=[], input_tokens=0, output_tokens=0)
    with pytest.raises(FrozenInstanceError):
        resp.text = "otro"  # type: ignore[misc]
