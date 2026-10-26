import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
from astrbot.api.message_components import Plain
from astrbot.api.platform import MessageType
from astrbot.builtin_stars.astrbot.group_chat_context import GroupChatContext
from astrbot.core.config.astrbot_config import AstrBotConfig
from SmartJoinChat.decision import DEFAULT_SETTINGS, METHOD
from SmartJoinChat.main import SmartJoinChat


class FakeEvent:
    def __init__(
        self,
        *,
        umo: str = "platform:group:g1",
        extras: dict | None = None,
        messages: list | None = None,
    ) -> None:
        self.unified_msg_origin = umo
        self.is_at_or_wake_command = False
        self._extras = extras or {}
        self._messages = messages or [Plain(text="hello")]

    def get_message_type(self):
        return MessageType.GROUP_MESSAGE

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def get_group_id(self):
        return "g1"

    def get_self_id(self):
        return "bot-1"

    def get_messages(self):
        return self._messages


class FakeGroupContext:
    def __init__(self, umo: str) -> None:
        self.raw_records = {
            umo: deque(
                [
                    "[alice/12:00:00]: first",
                    "[bob/12:00:02]: second",
                    "[alice/12:00:03]: third",
                ]
            )
        }
        self.formatted_events = []

    def cfg(self, event):
        return {
            "enable_active_reply": True,
            "ar_whitelist": [],
        }

    async def _format_message(self, event, cfg):
        self.formatted_events.append(event)
        return "[carol/12:00:05]: current"


class FakeProvider:
    def meta(self):
        return SimpleNamespace(id="default-provider")


class FakeContext:
    def __init__(self, config: dict, *, joins: bool = True) -> None:
        self.config = config
        self.llm_calls = []
        self._joins = joins
        self.conversation_manager = None

    def get_config(self, umo=None):
        return self.config

    async def get_using_provider_async(self, umo=None):
        return FakeProvider()

    async def llm_generate(self, **kwargs):
        self.llm_calls.append(kwargs)
        return SimpleNamespace(
            completion_text=json.dumps({"join": self._joins, "reason": "test decision"})
        )


def active_reply_config(**overrides):
    active_reply = {
        "enable": True,
        "method": METHOD,
        "possibility_reply": 0.1,
        "whitelist": [],
        **DEFAULT_SETTINGS,
    }
    active_reply.update(overrides)
    return {
        "provider_ltm_settings": {
            "active_reply": active_reply,
            "group_icl_enable": False,
        }
    }


@pytest.mark.asyncio
async def test_contextual_decision_uses_isolated_provider_call():
    config = active_reply_config(
        smart_join_provider_id="decision-model",
        smart_join_history_count=2,
        smart_join_activeness="low",
    )
    context = FakeContext(config, joins=True)
    plugin = SmartJoinChat(context, None)
    plugin._original_need_active_reply = _unexpected_original
    group_context = FakeGroupContext("platform:group:g1")
    event = FakeEvent()

    join = await plugin._need_active_reply(group_context, event)

    assert join is True
    assert len(context.llm_calls) == 1
    call = context.llm_calls[0]
    assert call["chat_provider_id"] == "decision-model"
    assert call["contexts"] == []
    assert call["session_id"]
    assert "Activeness: LOW" in call["prompt"]
    assert "[alice/12:00:03]: third" in call["prompt"]
    assert "[carol/12:00:05]: current" in call["prompt"]
    assert "[alice/12:00:00]: first" not in call["prompt"]
    assert "[bob/12:00:02]: second" not in call["prompt"]
    assert group_context.formatted_events == [event]
    assert context.conversation_manager is None


@pytest.mark.asyncio
async def test_contextual_decision_falls_back_to_current_provider():
    config = active_reply_config(smart_join_provider_id="")
    context = FakeContext(config, joins=False)
    plugin = SmartJoinChat(context, None)
    plugin._original_need_active_reply = _unexpected_original

    join = await plugin._need_active_reply(
        FakeGroupContext("platform:group:g1"),
        FakeEvent(),
    )

    assert join is False
    assert context.llm_calls[0]["chat_provider_id"] == "default-provider"


@pytest.mark.asyncio
async def test_other_methods_delegate_to_original():
    config = active_reply_config(method="possibility_reply")
    context = FakeContext(config)
    plugin = SmartJoinChat(context, None)

    async def original(group_context, event):
        return True

    plugin._original_need_active_reply = original

    assert await plugin._need_active_reply(FakeGroupContext("umo"), FakeEvent())
    assert context.llm_calls == []


@pytest.mark.asyncio
async def test_whitelist_blocks_before_llm_call():
    context = FakeContext(active_reply_config())
    plugin = SmartJoinChat(context, None)
    plugin._original_need_active_reply = _unexpected_original

    class WhitelistedGroupContext(FakeGroupContext):
        def cfg(self, event):
            return {
                "enable_active_reply": True,
                "ar_whitelist": ["another-group"],
            }

    result = await plugin._need_active_reply(
        WhitelistedGroupContext("platform:group:g1"),
        FakeEvent(),
    )

    assert result is False
    assert context.llm_calls == []


class CollectingGroupContext(GroupChatContext):
    def __init__(self):
        self.recorded = []

    async def handle_message(self, event):
        self.recorded.append(event)


@pytest.mark.asyncio
async def test_collector_reuses_group_context_when_injection_is_off():
    group_context = CollectingGroupContext()
    metadata = SimpleNamespace(
        star_cls=SimpleNamespace(group_chat_context=group_context)
    )
    context = FakeContext(active_reply_config())
    context.get_registered_star = lambda name: metadata
    plugin = SmartJoinChat(context, None)
    plugin._installed = True
    event = FakeEvent()

    await plugin.collect_group_history(event)

    assert group_context.recorded == [event]


@pytest.mark.asyncio
async def test_collector_skips_when_group_injection_is_enabled():
    group_context = CollectingGroupContext()
    metadata = SimpleNamespace(
        star_cls=SimpleNamespace(group_chat_context=group_context)
    )
    config = active_reply_config()
    config["provider_ltm_settings"]["group_icl_enable"] = True
    context = FakeContext(config)
    context.get_registered_star = lambda name: metadata
    plugin = SmartJoinChat(context, None)
    plugin._installed = True

    await plugin.collect_group_history(FakeEvent())

    assert group_context.recorded == []


def test_profile_settings_are_mirrored_into_plugin_config(tmp_path: Path):
    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
    plugin_config = AstrBotConfig(
        config_path=str(tmp_path / "plugin_config.json"),
        schema=schema,
    )
    profile = active_reply_config(
        smart_join_provider_id="decision-model",
        smart_join_history_count=33,
    )

    class Manager:
        def __init__(self):
            self.confs = {"default": profile}

    context = SimpleNamespace(astrbot_config_mgr=Manager())
    plugin = SmartJoinChat(context, plugin_config)
    plugin._installed = True

    plugin._capture_profile_settings(profile)

    snapshot = plugin_config["profile_snapshots"]["default"]
    assert snapshot["smart_join_provider_id"] == "decision-model"
    assert snapshot["smart_join_history_count"] == 33
    saved = json.loads(
        (tmp_path / "plugin_config.json").read_text(encoding="utf-8-sig")
    )
    assert saved["profile_snapshots"]["default"]["smart_join_history_count"] == 33


async def _unexpected_original(group_context, event):
    raise AssertionError("original method should not be called")
