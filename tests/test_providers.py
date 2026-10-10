from types import SimpleNamespace

from agent_harness.providers import (
    OpenCodeProvider,
    ProviderError,
    ScriptedProvider,
    call_tool,
    say,
    scripted_run,
)
from agent_harness.types import Message


def test_scripted_provider_replays_turns():
    p = scripted_run(say("hello"), say("bye"))
    m1 = p.complete([], [])
    m2 = p.complete([], [])
    assert (m1.content, m2.content) == ("hello", "bye")


def test_scripted_provider_estimates_usage():
    p = scripted_run(say("x" * 100))
    msg = p.complete([say_to_msg("y" * 400)], [])
    assert msg.usage.input_tokens >= 100  # 400 chars / 4
    assert msg.usage.output_tokens >= 25


def say_to_msg(content):
    from agent_harness.types import Message

    return Message.user(content)


def test_script_exhaustion_raises():
    p = ScriptedProvider([])
    try:
        p.complete([], [])
    except ProviderError as e:
        assert "script exhausted" in str(e)
    else:
        raise AssertionError("expected ProviderError")


def test_opencode_provider_translates_tool_calls_and_usage():
    requests = []

    class Completions:
        def create(self, **request):
            requests.append(request)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="I will inspect it.",
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_1",
                                    function=SimpleNamespace(
                                        name="read_file",
                                        arguments='{"path":"notes.md"}',
                                    ),
                                )
                            ],
                        )
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
            )

    provider = OpenCodeProvider(
        model="test-model",
        api_key="test-key",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
    )
    message = provider.complete(
        [Message.system("rules"), Message.user("inspect notes")],
        [{"name": "read_file", "description": "read", "parameters": {"type": "object"}}],
    )

    assert message.content == "I will inspect it."
    assert message.tool_calls[0].name == "read_file"
    assert message.tool_calls[0].arguments == {"path": "notes.md"}
    assert message.usage.input_tokens == 11
    assert message.usage.output_tokens == 7
    assert requests[0]["tools"][0]["type"] == "function"
    assert requests[0]["tools"][0]["function"]["name"] == "read_file"


def test_opencode_provider_rejects_invalid_tool_json():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name="read_file", arguments="not-json"),
                        )
                    ],
                )
            )
        ],
        usage=None,
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: response),
        )
    )
    provider = OpenCodeProvider(model="test-model", api_key="test-key", client=client)
    try:
        provider.complete([], [])
    except ProviderError as exc:
        assert "invalid JSON arguments" in str(exc)
    else:
        raise AssertionError("expected invalid tool JSON to fail closed")


def test_callable_script_items_see_conversation():
    seen = {}

    def inspect(messages):
        seen["n"] = len(messages)
        return say("inspected")

    p = scripted_run(call_tool("noop", {}), inspect)
    p.complete([], [])
    p.complete([say_to_msg("a"), say_to_msg("b")], [])
    assert seen["n"] == 2
