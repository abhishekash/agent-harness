from agent_harness.providers import ProviderError, ScriptedProvider, call_tool, say, scripted_run


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


def test_callable_script_items_see_conversation():
    seen = {}

    def inspect(messages):
        seen["n"] = len(messages)
        return say("inspected")

    p = scripted_run(call_tool("noop", {}), inspect)
    p.complete([], [])
    p.complete([say_to_msg("a"), say_to_msg("b")], [])
    assert seen["n"] == 2
