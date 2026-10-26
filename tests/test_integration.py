import copy
from types import SimpleNamespace

import pytest
from astrbot.builtin_stars.astrbot.group_chat_context import GroupChatContext
from astrbot.core.config.default import (
    CONFIG_METADATA_2,
    CONFIG_METADATA_3,
    DEFAULT_CONFIG,
)
from astrbot.core.config.i18n_utils import ConfigMetadataI18n
from astrbot.dashboard.services.config_service import validate_config
from SmartJoinChat import integration
from SmartJoinChat import main as plugin_main
from SmartJoinChat.decision import METHOD
from SmartJoinChat.integration import ProfileSchemaBridge


@pytest.fixture
def isolated_metadata(monkeypatch):
    monkeypatch.setattr(
        integration,
        "CONFIG_METADATA_2",
        copy.deepcopy(CONFIG_METADATA_2),
    )
    monkeypatch.setattr(
        integration,
        "CONFIG_METADATA_3",
        copy.deepcopy(CONFIG_METADATA_3),
    )


def test_profile_bridge_adds_fields_and_restores_core_metadata(isolated_metadata):
    default_config = copy.deepcopy(DEFAULT_CONFIG)
    configs = {"default": copy.deepcopy(DEFAULT_CONFIG)}
    bridge = ProfileSchemaBridge(
        default_config,
        configs,
        {
            "default": {
                "smart_join_history_count": 42,
                "smart_join_activeness": "high",
            }
        },
        {},
    )

    bridge.install()

    profile_items = integration.find_profile_metadata_items()
    method_meta = profile_items["provider_ltm_settings.active_reply.method"]
    assert METHOD in method_meta["options"]
    assert method_meta["labels"] == ["Reply probability", "Smart context (LLM)"]
    profile_keys = list(profile_items)
    method_index = profile_keys.index("provider_ltm_settings.active_reply.method")
    assert profile_keys[method_index + 1] == (
        "provider_ltm_settings.active_reply.smart_join_provider_id"
    )
    assert (
        profile_items["provider_ltm_settings.active_reply.possibility_reply"][
            "condition"
        ]["provider_ltm_settings.active_reply.method"]
        == "possibility_reply"
    )

    history_meta = profile_items[
        "provider_ltm_settings.active_reply.smart_join_history_count"
    ]
    assert history_meta["description"] == "Recent Messages"
    assert history_meta["type"] == "int"

    gate_meta = profile_items[
        "provider_ltm_settings.active_reply.smart_join_gate_probability"
    ]
    assert gate_meta["condition"][
        "provider_ltm_settings.active_reply.smart_join_gate_enable"
    ]

    active_reply = configs["default"]["provider_ltm_settings"]["active_reply"]
    assert active_reply["smart_join_history_count"] == 42
    assert active_reply["smart_join_activeness"] == "high"
    errors, normalized = validate_config(
        copy.deepcopy(configs["default"]),
        integration.CONFIG_METADATA_2,
        is_core=True,
    )
    assert errors == []
    assert (
        normalized["provider_ltm_settings"]["active_reply"]["smart_join_provider_id"]
        == ""
    )

    converted = ConfigMetadataI18n.convert_to_i18n_keys(integration.CONFIG_METADATA_3)
    converted_items = converted["ext_group"]["metadata"]["ltm"]["items"]
    assert (
        converted_items["provider_ltm_settings.active_reply.smart_join_history_count"][
            "description"
        ]
        == "Recent Messages"
    )
    assert converted_items["provider_ltm_settings.active_reply.method"]["labels"] == [
        "Reply probability",
        "Smart context (LLM)",
    ]

    bridge.remove()

    assert (
        "provider_ltm_settings.active_reply.smart_join_history_count"
        not in integration.find_profile_metadata_items()
    )
    assert integration.find_profile_metadata_items()[
        "provider_ltm_settings.active_reply.method"
    ]["options"] == ["possibility_reply"]
    restored_items = integration.find_profile_metadata_items()
    assert list(restored_items)[:4] == [
        "provider_ltm_settings.group_icl_enable",
        "provider_ltm_settings.group_message_max_cnt",
        "provider_ltm_settings.image_caption",
        "provider_ltm_settings.image_caption_provider_id",
    ]
    assert restored_items["provider_ltm_settings.active_reply.possibility_reply"][
        "condition"
    ] == {"provider_ltm_settings.active_reply.enable": True}
    assert (
        "smart_join_history_count"
        not in configs["default"]["provider_ltm_settings"]["active_reply"]
    )


@pytest.mark.asyncio
async def test_plugin_initialize_and_terminate_restore_decision_hook(
    isolated_metadata,
    monkeypatch,
):
    monkeypatch.setattr(
        plugin_main,
        "DEFAULT_CONFIG",
        copy.deepcopy(DEFAULT_CONFIG),
    )

    class Manager:
        def __init__(self):
            self.confs = {"default": copy.deepcopy(DEFAULT_CONFIG)}

    context = SimpleNamespace(astrbot_config_mgr=Manager())
    plugin = plugin_main.SmartJoinChat(context, None)
    original = GroupChatContext.need_active_reply

    await plugin.initialize()

    try:
        assert GroupChatContext.need_active_reply is not original
        assert getattr(
            GroupChatContext.need_active_reply,
            "_smart_join_patched",
            False,
        )
        assert (
            METHOD
            in integration.find_profile_metadata_items()[
                "provider_ltm_settings.active_reply.method"
            ]["options"]
        )
    finally:
        await plugin.terminate()

    assert GroupChatContext.need_active_reply is original
    assert (
        METHOD
        not in integration.find_profile_metadata_items()[
            "provider_ltm_settings.active_reply.method"
        ]["options"]
    )

    await plugin.initialize()
    try:
        assert getattr(
            GroupChatContext.need_active_reply,
            "_smart_join_patched",
            False,
        )
    finally:
        await plugin.terminate()

    assert GroupChatContext.need_active_reply is original
