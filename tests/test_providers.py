from types import SimpleNamespace

from agent_harness.providers import (
    OpenCodeProvider,
    ProviderError,
    ScriptedProvider,
    call_tool,
    say,
    scripted_run,
)
from agent_harness.types import Message, ToolCall


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


def test_opencode_go_defaults_and_normalizes_cli_model(monkeypatch):
    monkeypatch.delenv("OPENCODE_BASE_URL", raising=False)
    default = OpenCodeProvider(api_key="test-key", client=object())
    assert default.model == "mimo-v2.6-pro"
    assert default.base_url == "https://opencode.ai/zen/go/v1"
    provider = OpenCodeProvider(
        model="opencode-go/mimo-v2.6-pro",
        api_key="test-key",
        client=object(),
    )
    assert provider.model == default.model


def test_opencode_go_honest_headers_cover_turns_and_summaries():
    from agent_harness import __version__

    requests = []

    def create(**request):
        requests.append(request)
        return {"choices": [{"message": {"content": "done"}}]}

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = OpenCodeProvider(api_key="test-key", client=client)
    provider.complete([Message.user("task")], [])
    provider.complete([Message.user("continue")], [])
    provider.summarize([Message.user("summarize")])
    for request in requests:
        assert request["extra_headers"] == {
            "User-Agent": f"agent-harness/{__version__}",
            "x-opencode-session": provider.session_id,
        }
    # Repetitions must not share a conversation identity.
    another = OpenCodeProvider(api_key="test-key", client=client)
    assert another.session_id != provider.session_id
    explicit = OpenCodeProvider(api_key="test-key", client=client, session_id="eval-session-1")
    assert explicit.session_id == "eval-session-1"


def test_opencode_go_url_override_precedence(monkeypatch):
    monkeypatch.setenv("OPENCODE_BASE_URL", "https://env.example/v1")
    env = OpenCodeProvider(api_key="test-key", client=object())
    assert env.base_url == "https://env.example/v1"
    explicit = OpenCodeProvider(
        api_key="test-key", client=object(), base_url="https://explicit.example/v1"
    )
    assert explicit.base_url == "https://explicit.example/v1"


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


def test_anthropic_groups_parallel_tool_results_in_one_user_message():
    from agent_harness.providers.anthropic import _to_api_messages

    calls = [ToolCall("one", "read_file", {"path": "a"}), ToolCall("two", "read_file", {"path": "b"})]
    messages = [
        Message.user("read both"),
        Message(role="assistant", content="", tool_calls=calls),
        Message.tool_result(calls[0], "a"),
        Message.tool_result(calls[1], "b"),
        Message.user("continue"),
    ]

    translated = _to_api_messages(messages)
    assert len(translated) == 4
    assert translated[2]["role"] == "user"
    assert [block["tool_use_id"] for block in translated[2]["content"]] == ["one", "two"]
    assert translated[3] == {"role": "user", "content": "continue"}
