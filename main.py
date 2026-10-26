from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Mapping
from sys import maxsize
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Json, Plain
from astrbot.api.platform import MessageType
from astrbot.api.star import Context, Star, register
from astrbot.builtin_stars.astrbot.group_chat_context import GroupChatContext
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.config.default import DEFAULT_CONFIG

from .decision import (
    DEFAULT_SETTINGS,
    METHOD,
    SYSTEM_PROMPT,
    ProbabilityGate,
    build_decision_prompt,
    normalize_settings,
    parse_decision,
)
from .integration import (
    STRATEGY_KEYS,
    ProfileSchemaBridge,
    read_strategy_settings,
)


@register(
    "astrbot_plugin_smart_join_chat",
    "Astricaelus",
    "Let a model decide contextually whether the bot should join a group chat.",
    "1.0.0",
)
class SmartJoinChat(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context, config)
        self.plugin_config = config
        self._bridge: ProfileSchemaBridge | None = None
        self._installed = False
        self._decision_locks: dict[str, asyncio.Lock] = {}
        self._gate = ProbabilityGate()
        self._original_need_active_reply = None
        self._patched_need_active_reply = None
        self._original_save_config = None
        self._patched_save_config = None
        self._capture_guard = threading.RLock()
        self._capturing = False

    async def initialize(self) -> None:
        """Install the decision method and profile integration."""
        if self._installed:
            return

        configs = self.context.astrbot_config_mgr.confs
        snapshots = self._profile_snapshots()
        fallback = {
            key: self.plugin_config.get(key, DEFAULT_SETTINGS[key])
            if isinstance(self.plugin_config, Mapping)
            else DEFAULT_SETTINGS[key]
            for key in STRATEGY_KEYS
        }
        self._bridge = ProfileSchemaBridge(
            DEFAULT_CONFIG,
            configs,
            snapshots,
            fallback,
        )
        try:
            self._bridge.install()
            self._install_decision_patch()
            self._install_save_hook()
        except Exception:
            await self.terminate()
            raise
        self._installed = True
        logger.info(
            "[SmartJoinChat] Installed profile method %r and group-history collector.",
            METHOD,
        )

    async def terminate(self) -> None:
        """Restore all AstrBot extension points touched by this plugin."""
        if self._bridge is not None:
            self._bridge.remove()
            self._bridge = None

        if (
            self._patched_need_active_reply is not None
            and GroupChatContext.need_active_reply is self._patched_need_active_reply
            and self._original_need_active_reply is not None
        ):
            GroupChatContext.need_active_reply = self._original_need_active_reply

        if (
            self._patched_save_config is not None
            and AstrBotConfig.save_config is self._patched_save_config
            and self._original_save_config is not None
        ):
            AstrBotConfig.save_config = self._original_save_config

        self._original_need_active_reply = None
        self._patched_need_active_reply = None
        self._original_save_config = None
        self._patched_save_config = None
        self._decision_locks.clear()
        self._installed = False
        logger.info("[SmartJoinChat] Uninstalled profile integration.")

    def _install_decision_patch(self) -> None:
        original = GroupChatContext.need_active_reply
        if getattr(original, "_smart_join_patched", False):
            raise RuntimeError("Another SmartJoinChat instance is already active.")

        plugin = self

        async def need_active_reply(
            group_context: GroupChatContext,
            event: AstrMessageEvent,
        ) -> bool:
            return await plugin._need_active_reply(group_context, event)

        need_active_reply._smart_join_patched = True
        need_active_reply._smart_join_original = original
        self._original_need_active_reply = original
        self._patched_need_active_reply = need_active_reply
        GroupChatContext.need_active_reply = need_active_reply

    def _install_save_hook(self) -> None:
        original = AstrBotConfig.save_config
        if getattr(original, "_smart_join_patched", False):
            return
        plugin = self

        def save_config(
            config: AstrBotConfig,
            replace_config: dict | None = None,
            **kwargs: Any,
        ) -> Any:
            result = original(config, replace_config, **kwargs)
            plugin._capture_profile_settings(config)
            return result

        save_config._smart_join_patched = True
        save_config._smart_join_original = original
        self._original_save_config = original
        self._patched_save_config = save_config
        AstrBotConfig.save_config = save_config

    async def _need_active_reply(
        self,
        group_context: GroupChatContext,
        event: AstrMessageEvent,
    ) -> bool:
        original = self._original_need_active_reply
        config = self.context.get_config(umo=event.unified_msg_origin)
        active_reply = config.get("provider_ltm_settings", {}).get("active_reply", {})
        if not isinstance(active_reply, dict) or active_reply.get("method") != METHOD:
            return bool(await original(group_context, event))

        group_cfg = group_context.cfg(event)
        if not group_cfg["enable_active_reply"]:
            return False
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return False
        if event.is_at_or_wake_command:
            return False
        if not self._is_whitelisted(event, group_cfg["ar_whitelist"]):
            return False

        settings = normalize_settings(read_strategy_settings(active_reply))
        if not self._gate.should_continue(
            event.unified_msg_origin,
            enabled=settings["smart_join_gate_enable"],
            probability=settings["smart_join_gate_probability"],
            mode=settings["smart_join_gate_mode"],
        ):
            logger.debug(
                "[SmartJoinChat] Probability gate skipped %s",
                event.unified_msg_origin,
            )
            return False

        lock = self._decision_locks.setdefault(
            event.unified_msg_origin,
            asyncio.Lock(),
        )
        if lock.locked():
            logger.debug(
                "[SmartJoinChat] A decision is already running for %s",
                event.unified_msg_origin,
            )
            return False

        async with lock:
            try:
                history = await self._recent_history(
                    group_context,
                    event,
                    group_cfg,
                    settings["smart_join_history_count"],
                )
                provider_id = settings["smart_join_provider_id"]
                if not provider_id:
                    provider = await self.context.get_using_provider_async(
                        event.unified_msg_origin
                    )
                    if provider is None:
                        logger.warning(
                            "[SmartJoinChat] No chat provider is available for %s",
                            event.unified_msg_origin,
                        )
                        return False
                    provider_id = provider.meta().id

                prompt = build_decision_prompt(
                    history,
                    settings,
                    bot_name=event.get_self_id(),
                    group_name=event.get_group_id(),
                )
                response = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    contexts=[],
                    session_id=uuid.uuid4().hex,
                )
                join, reason = parse_decision(response.completion_text)
                logger.info(
                    "[SmartJoinChat] %s provider=%s join=%s reason=%s",
                    event.unified_msg_origin,
                    provider_id,
                    join,
                    reason[:160],
                )
                return join
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one failed decision must not stop chat
                logger.exception(
                    "[SmartJoinChat] Contextual decision failed for %s",
                    event.unified_msg_origin,
                )
                return False

    async def _recent_history(
        self,
        group_context: GroupChatContext,
        event: AstrMessageEvent,
        group_cfg: dict[str, Any],
        limit: int,
    ) -> list[str]:
        records = list(group_context.raw_records.get(event.unified_msg_origin, ()))
        if not event.get_extra("_group_context_record_id"):
            current = await group_context._format_message(event, group_cfg)
            if not records or records[-1] != current:
                records.append(current)
        return records[-limit:]

    @filter.event_message_type(
        filter.EventMessageType.GROUP_MESSAGE,
        priority=maxsize,
    )
    async def collect_group_history(self, event: AstrMessageEvent) -> None:
        """Collect messages for contextual decisions when ICL is disabled."""
        if not self._installed:
            return
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return
        config = self.context.get_config(umo=event.unified_msg_origin)
        ltm = config.get("provider_ltm_settings", {})
        active_reply = ltm.get("active_reply", {})
        if (
            not isinstance(active_reply, dict)
            or not active_reply.get("enable", False)
            or active_reply.get("method") != METHOD
            or ltm.get("group_icl_enable", False)
        ):
            return
        if event.get_extra("handlers_parsed_params", {}):
            return
        if not any(
            isinstance(component, Plain | Image | Json)
            for component in event.get_messages()
        ):
            return

        group_context = self._group_context()
        if group_context is None:
            return
        try:
            await group_context.handle_message(event)
        except Exception:  # noqa: BLE001 - history collection is best-effort
            logger.exception(
                "[SmartJoinChat] Failed to collect group history for %s",
                event.unified_msg_origin,
            )

    def _group_context(self) -> GroupChatContext | None:
        metadata = self.context.get_registered_star("astrbot")
        if metadata is None or metadata.star_cls is None:
            return None
        group_context = getattr(
            metadata.star_cls,
            "group_chat_context",
            None,
        )
        return group_context if isinstance(group_context, GroupChatContext) else None

    @staticmethod
    def _is_whitelisted(event: AstrMessageEvent, whitelist: list[str]) -> bool:
        if not whitelist:
            return True
        return event.unified_msg_origin in whitelist or bool(
            event.get_group_id() and event.get_group_id() in whitelist
        )

    def _profile_snapshots(self) -> dict[str, dict[str, Any]]:
        if not isinstance(self.plugin_config, Mapping):
            return {}
        snapshots = self.plugin_config.get("profile_snapshots", {})
        if not isinstance(snapshots, dict):
            return {}
        return {
            str(profile_id): dict(settings)
            for profile_id, settings in snapshots.items()
            if isinstance(settings, Mapping)
        }

    def _capture_profile_settings(self, saved_config: AstrBotConfig) -> None:
        if not self._installed or not isinstance(self.plugin_config, AstrBotConfig):
            return
        with self._capture_guard:
            if self._capturing:
                return
            self._capturing = True
            try:
                profile_id = None
                for (
                    candidate_id,
                    candidate,
                ) in self.context.astrbot_config_mgr.confs.items():
                    if candidate is saved_config:
                        profile_id = candidate_id
                        break
                if profile_id is None:
                    return
                active_reply = saved_config.get("provider_ltm_settings", {}).get(
                    "active_reply", {}
                )
                if (
                    not isinstance(active_reply, dict)
                    or active_reply.get("method") != METHOD
                ):
                    return

                settings = read_strategy_settings(active_reply)
                snapshots = self.plugin_config.get("profile_snapshots", {})
                if not isinstance(snapshots, dict):
                    snapshots = {}
                    self.plugin_config["profile_snapshots"] = snapshots
                if snapshots.get(profile_id) != settings:
                    snapshots[profile_id] = settings
                    self.plugin_config.save_config()
            except Exception:  # noqa: BLE001 - persistence must not break saves
                logger.exception(
                    "[SmartJoinChat] Failed to mirror profile settings for restart."
                )
            finally:
                self._capturing = False


# Keep these names available for tests and future AstrBot integration.
__all__ = [
    "METHOD",
    "SmartJoinChat",
]
