import pytest

from agent_core.providers import FakeProvider, FakeProviderExhaustedError, LLMProvider
from agent_core.types import LLMResponse, Message


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], input_tokens=1, output_tokens=1)


def test_returns_responses_in_order() -> None:
    fake = FakeProvider([_resp("uno"), _resp("dos"), _resp("tres")])
    msgs = [Message(role="user", content="hola")]
    assert [fake.complete(msgs, []).text for _ in range(3)] == ["uno", "dos", "tres"]


def test_records_messages_per_call() -> None:
    fake = FakeProvider([_resp("a"), _resp("b")])
    first = [Message(role="user", content="1")]
    second = [Message(role="user", content="1"), Message(role="assistant", content="a")]
    fake.complete(first, [])
    fake.complete(second, [])
    assert fake.calls == [first, second]


def test_recorded_messages_are_a_copy() -> None:
    fake = FakeProvider([_resp("a")])
    history = [Message(role="user", content="hola")]
    fake.complete(history, [])
    history.append(Message(role="assistant", content="a"))
    assert fake.calls == [[Message(role="user", content="hola")]]


def test_raises_clear_error_when_exhausted() -> None:
    fake = FakeProvider([_resp("a"), _resp("b")])
    msgs = [Message(role="user", content="x")]
    fake.complete(msgs, [])
    fake.complete(msgs, [])
    with pytest.raises(FakeProviderExhaustedError, match=r"llamada #3.*2 respuestas"):
        fake.complete(msgs, [])
    # La llamada fallida también queda registrada, para poder depurar.
    assert len(fake.calls) == 3


def test_empty_responses_fails_on_first_call() -> None:
    fake = FakeProvider([])
    with pytest.raises(FakeProviderExhaustedError, match=r"llamada #1.*0 respuestas"):
        fake.complete([Message(role="user")], [])


def test_exhausted_error_is_runtime_error() -> None:
    assert issubclass(FakeProviderExhaustedError, RuntimeError)


def test_remaining_decreases() -> None:
    fake = FakeProvider([_resp("a"), _resp("b")])
    assert fake.remaining == 2
    fake.complete([], [])
    assert fake.remaining == 1
    fake.complete([], [])
    assert fake.remaining == 0


def test_accepts_generator() -> None:
    fake = FakeProvider(_resp(t) for t in ["a", "b"])
    assert fake.remaining == 2
    assert fake.complete([], []).text == "a"


def test_satisfies_llm_provider_protocol() -> None:
    assert isinstance(FakeProvider([]), LLMProvider)
