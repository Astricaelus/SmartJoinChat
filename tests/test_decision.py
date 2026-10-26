from SmartJoinChat.decision import (
    MAX_PERSONA_PROMPT_CHARS,
    ProbabilityGate,
    build_decision_prompt,
    normalize_settings,
    parse_decision,
)


def test_normalize_settings_clamps_and_defaults():
    settings = normalize_settings(
        {
            "smart_join_history_count": 500,
            "smart_join_gate_probability": -2,
            "smart_join_activeness": "unexpected",
            "smart_join_gate_mode": "unexpected",
            "smart_join_gate_enable": "yes",
            "smart_join_provider_id": "  provider-a  ",
        }
    )

    assert settings["smart_join_history_count"] == 100
    assert settings["smart_join_gate_probability"] == 0.0
    assert settings["smart_join_activeness"] == "medium"
    assert settings["smart_join_gate_mode"] == "raw"
    assert settings["smart_join_gate_enable"] is True
    assert settings["smart_join_provider_id"] == "provider-a"


def test_build_decision_prompt_contains_activeness_and_history():
    settings = normalize_settings({"smart_join_activeness": "high"})
    prompt = build_decision_prompt(
        ["[alice/12:00:00]: hello", "[bob/12:00:03]: question"],
        settings,
        bot_names=["helper", "large fish"],
        bot_id="bot-123",
        group_name="group-1",
    )

    assert "Activeness: HIGH" in prompt
    assert "Be eager but not disruptive" in prompt
    assert "[alice/12:00:00]: hello" in prompt
    assert "<transcript>" in prompt
    assert "helper" in prompt
    assert "large fish" in prompt
    assert "bot-123" in prompt
    assert "does not need to be named or mentioned" in prompt


def test_build_decision_prompt_includes_persona_and_limits_size():
    settings = normalize_settings({"smart_join_activeness": "medium"})
    prompt = build_decision_prompt(
        ["[alice/12:00:00]: 大肥鱼你在吗"],
        settings,
        bot_names=["大肥鱼"],
        persona_prompt="# 一、角色\n你是大肥鱼，一条好奇的鲸鱼。",
    )
    assert "<persona>" in prompt
    assert "你是大肥鱼" in prompt
    assert "<transcript>" in prompt

    truncated = build_decision_prompt(
        [],
        settings,
        persona_prompt="x" * (MAX_PERSONA_PROMPT_CHARS + 100),
    )
    assert "[persona prompt truncated]" in truncated


def test_parse_decision_accepts_json_and_fenced_json():
    assert parse_decision('{"join": true, "reason": "useful"}') == (
        True,
        "useful",
    )
    assert parse_decision('```json\n{"join": false, "reason": "no opening"}\n```') == (
        False,
        "no opening",
    )
    assert parse_decision('answer: {"participate": "yes"}') == (
        True,
        "model decision",
    )


def test_parse_decision_fails_closed():
    assert parse_decision("") == (False, "empty model response")
    assert parse_decision("I think maybe we can talk") == (
        False,
        "unparseable model response",
    )


def test_raw_gate_uses_independent_draws():
    draws = iter([0.1, 0.9, 0.2])
    gate = ProbabilityGate(lambda: next(draws))

    assert gate.should_continue("umo", enabled=True, probability=0.5, mode="raw")
    assert not gate.should_continue("umo", enabled=True, probability=0.5, mode="raw")
    assert gate.should_continue("umo", enabled=True, probability=0.5, mode="raw")


def test_uniform_gate_spreads_passes_evenly():
    gate = ProbabilityGate(lambda: 0.0)

    results = [
        gate.should_continue(
            "umo",
            enabled=True,
            probability=0.25,
            mode="uniform",
        )
        for _ in range(8)
    ]

    assert results == [False, False, False, True, False, False, False, True]


def test_gate_short_circuits_at_extremes():
    gate = ProbabilityGate(lambda: 0.0)

    assert gate.should_continue("umo", enabled=False, probability=0.0, mode="raw")
    assert not gate.should_continue("umo", enabled=True, probability=0.0, mode="raw")
    assert gate.should_continue("umo", enabled=True, probability=1.0, mode="raw")
