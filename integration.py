from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from astrbot.core.config.default import CONFIG_METADATA_2, CONFIG_METADATA_3
from astrbot.core.config.i18n_utils import ConfigMetadataI18n

from .decision import DEFAULT_SETTINGS, METHOD

STRATEGY_KEYS = tuple(DEFAULT_SETTINGS)

CONFIG_ITEM_META = {
    "smart_join_provider_id": {"type": "string"},
    "smart_join_history_count": {"type": "int"},
    "smart_join_activeness": {"type": "string"},
    "smart_join_gate_enable": {"type": "bool"},
    "smart_join_gate_probability": {"type": "float"},
    "smart_join_gate_mode": {"type": "string"},
}

PROFILE_ITEM_META = {
    "smart_join_provider_id": {
        "description": "Decision Model",
        "hint": (
            "Chat provider used only to judge whether the bot should join. "
            "Leave empty to use the bot's current chat model."
        ),
        "type": "string",
        "_special": "select_provider",
    },
    "smart_join_history_count": {
        "description": "Recent Messages",
        "hint": (
            "Number of recent group messages shown to the decision model. "
            "Messages are collected through AstrBot's group-context pipeline."
        ),
        "type": "int",
        "slider": {"min": 1, "max": 100, "step": 1},
    },
    "smart_join_activeness": {
        "description": "Activeness",
        "hint": "How willing the bot should be to join a conversation.",
        "type": "string",
        "options": ["low", "medium", "high"],
        "labels": ["Low", "Medium", "High"],
    },
    "smart_join_gate_enable": {
        "description": "Probability Gate",
        "hint": (
            "Run a lightweight probability gate before the model decision. "
            "This can reduce LLM calls and unsolicited interruptions."
        ),
        "type": "bool",
    },
    "smart_join_gate_probability": {
        "description": "Gate Probability",
        "hint": "Probability that a message reaches the model decision step.",
        "type": "float",
        "slider": {"min": 0, "max": 1, "step": 0.05},
        "condition": {
            "provider_ltm_settings.active_reply.enable": True,
            "provider_ltm_settings.active_reply.method": METHOD,
            "provider_ltm_settings.active_reply.smart_join_gate_enable": True,
        },
    },
    "smart_join_gate_mode": {
        "description": "Gate Sampling",
        "hint": (
            "Raw uses each random draw independently. Uniform spacing uses a "
            "randomized phase and spreads allowed messages evenly to reduce bursts."
        ),
        "type": "string",
        "options": ["raw", "uniform"],
        "labels": ["Raw RNG value", "Uniform spacing correction"],
        "condition": {
            "provider_ltm_settings.active_reply.enable": True,
            "provider_ltm_settings.active_reply.method": METHOD,
            "provider_ltm_settings.active_reply.smart_join_gate_enable": True,
        },
    },
}


def find_profile_metadata_items() -> dict[str, Any] | None:
    """Locate the `ltm` item mapping in the dashboard profile metadata.

    Returns:
        The item mapping, or None when the expected core structure is absent.
    """
    section = CONFIG_METADATA_3.get("ext_group", {}).get("metadata", {}).get("ltm")
    if isinstance(section, dict) and isinstance(section.get("items"), dict):
        return section["items"]
    return None


def find_validation_schema() -> dict[str, Any] | None:
    """Locate the active-reply item mapping in the validation metadata.

    Returns:
        The active-reply item mapping, or None when the core structure is absent.
    """
    stack = [CONFIG_METADATA_2]
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        ltm = current.get("provider_ltm_settings")
        if isinstance(ltm, dict):
            items = ltm.get("items", {})
            active_reply = items.get("active_reply")
            if isinstance(active_reply, dict) and isinstance(
                active_reply.get("items"), dict
            ):
                return active_reply["items"]
        stack.extend(value for value in current.values() if isinstance(value, dict))
    return None


def read_strategy_settings(active_reply: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read only SmartJoinChat keys that are present in a profile.

    Args:
        active_reply: The profile's active-reply configuration.

    Returns:
        A partial settings mapping containing keys that exist in the profile.
    """
    if not isinstance(active_reply, Mapping):
        return {}
    return {key: active_reply[key] for key in STRATEGY_KEYS if key in active_reply}


class ProfileSchemaBridge:
    """Expose strategy fields in AstrBot's profile configuration UI."""

    def __init__(
        self,
        default_config: dict[str, Any],
        configs: Mapping[str, dict[str, Any]],
        stored_settings: Mapping[str, Mapping[str, Any]],
        fallback_settings: Mapping[str, Any],
    ) -> None:
        self.default_config = default_config
        self.configs = configs
        self.stored_settings = stored_settings
        self.fallback_settings = dict(DEFAULT_SETTINGS)
        self.fallback_settings.update(fallback_settings)
        self._installed = False
        self._method_meta: dict[str, Any] | None = None
        self._original_method_options: list[str] | None = None
        self._original_method_labels: list[str] | None = None
        self._original_method_raw_text = False
        self._original_converter = None
        self._patched_converter = None
        self._converter_owned = False
        self._profile_items: dict[str, Any] | None = None
        self._validation_items: dict[str, Any] | None = None
        self._original_profile_key_order: list[str] | None = None
        self._possibility_meta: dict[str, Any] | None = None
        self._original_possibility_condition: dict[str, Any] | None = None

    def install(self) -> None:
        """Install profile defaults, selectors, and validation metadata."""
        if self._installed:
            return

        self._profile_items = find_profile_metadata_items()
        self._validation_items = find_validation_schema()
        if self._profile_items is None or self._validation_items is None:
            raise RuntimeError(
                "Unsupported AstrBot version: active-reply profile metadata not found."
            )
        self._original_profile_key_order = list(self._profile_items)

        default_active_reply = self.default_config["provider_ltm_settings"][
            "active_reply"
        ]
        for key, value in DEFAULT_SETTINGS.items():
            default_active_reply.setdefault(key, copy.deepcopy(value))

        default_settings = dict(self.fallback_settings)
        for profile_id, config in self.configs.items():
            active_reply = config.get("provider_ltm_settings", {}).get(
                "active_reply", {}
            )
            if not isinstance(active_reply, dict):
                active_reply = {}
                config.setdefault("provider_ltm_settings", {})["active_reply"] = (
                    active_reply
                )

            settings = default_settings
            if profile_id in self.stored_settings:
                settings = {
                    **default_settings,
                    **dict(self.stored_settings[profile_id]),
                }
            elif active_reply.get("method") == METHOD:
                settings = {
                    **default_settings,
                    **read_strategy_settings(active_reply),
                }
            for key, value in settings.items():
                if key in DEFAULT_SETTINGS:
                    active_reply.setdefault(key, copy.deepcopy(value))

        method_meta = self._profile_items.get(
            "provider_ltm_settings.active_reply.method"
        )
        if isinstance(method_meta, dict):
            self._method_meta = method_meta
            self._original_method_options = list(method_meta.get("options", []))
            self._original_method_labels = (
                list(method_meta["labels"]) if "labels" in method_meta else None
            )
            self._original_method_raw_text = bool(method_meta.get("raw_text"))
            options = list(method_meta.get("options", []))
            if METHOD not in options:
                options.append(METHOD)
            method_meta["options"] = options
            method_meta["labels"] = ["Reply probability", "Smart context (LLM)"]
            method_meta["_raw_text_attrs"] = ["labels"]

        possibility_meta = self._profile_items.get(
            "provider_ltm_settings.active_reply.possibility_reply"
        )
        if isinstance(possibility_meta, dict):
            self._possibility_meta = possibility_meta
            original_condition = possibility_meta.get("condition")
            self._original_possibility_condition = (
                copy.deepcopy(original_condition)
                if isinstance(original_condition, dict)
                else None
            )
            condition = dict(original_condition or {})
            condition["provider_ltm_settings.active_reply.method"] = "possibility_reply"
            possibility_meta["condition"] = condition

        for key, metadata in CONFIG_ITEM_META.items():
            self._validation_items[key] = copy.deepcopy(metadata)

        field_metas: dict[str, Any] = {}
        for key, metadata in PROFILE_ITEM_META.items():
            full_key = f"provider_ltm_settings.active_reply.{key}"
            meta = copy.deepcopy(metadata)
            meta["raw_text"] = True
            condition = dict(meta.get("condition", {}))
            condition["provider_ltm_settings.active_reply.enable"] = True
            condition["provider_ltm_settings.active_reply.method"] = METHOD
            meta["condition"] = condition
            field_metas[full_key] = meta

        ordered_items: dict[str, Any] = {}
        for key, value in self._profile_items.items():
            ordered_items[key] = value
            if key == "provider_ltm_settings.active_reply.method":
                ordered_items.update(field_metas)
        self._profile_items.clear()
        self._profile_items.update(ordered_items)

        self._install_text_converter()
        self._installed = True

    def remove(self) -> None:
        """Remove all profile UI and validation additions."""
        if not self._installed:
            return

        if self._method_meta is not None:
            if self._original_method_options is None:
                self._method_meta.pop("options", None)
            else:
                self._method_meta["options"] = self._original_method_options
            if self._original_method_labels is None:
                self._method_meta.pop("labels", None)
            else:
                self._method_meta["labels"] = self._original_method_labels
            if self._original_method_raw_text:
                self._method_meta["raw_text"] = True
            else:
                self._method_meta.pop("raw_text", None)
            self._method_meta.pop("_raw_text_attrs", None)

        if self._profile_items is not None:
            for key in PROFILE_ITEM_META:
                self._profile_items.pop(
                    f"provider_ltm_settings.active_reply.{key}", None
                )
            if self._original_profile_key_order is not None:
                ordered_items = {
                    key: self._profile_items[key]
                    for key in self._original_profile_key_order
                    if key in self._profile_items
                }
                for key, value in self._profile_items.items():
                    ordered_items.setdefault(key, value)
                self._profile_items.clear()
                self._profile_items.update(ordered_items)
        if self._validation_items is not None:
            for key in CONFIG_ITEM_META:
                self._validation_items.pop(key, None)
        if self._possibility_meta is not None:
            if self._original_possibility_condition is None:
                self._possibility_meta.pop("condition", None)
            else:
                self._possibility_meta["condition"] = copy.deepcopy(
                    self._original_possibility_condition
                )

        active_reply = self.default_config.get("provider_ltm_settings", {}).get(
            "active_reply", {}
        )
        for key in STRATEGY_KEYS:
            active_reply.pop(key, None)
        for config in self.configs.values():
            active_reply = config.get("provider_ltm_settings", {}).get(
                "active_reply", {}
            )
            if isinstance(active_reply, dict):
                for key in STRATEGY_KEYS:
                    active_reply.pop(key, None)

        self._remove_text_converter()
        self._installed = False

    def _install_text_converter(self) -> None:
        if getattr(
            ConfigMetadataI18n.convert_to_i18n_keys,
            "_smart_join_patched",
            False,
        ):
            return
        self._converter_owned = True
        self._original_converter = ConfigMetadataI18n.convert_to_i18n_keys
        bridge = self

        def converted(metadata: dict[str, Any]) -> dict[str, Any]:
            result = bridge._original_converter(metadata)
            return bridge._restore_raw_text(result)

        converted._smart_join_patched = True
        converted._smart_join_original = self._original_converter
        self._patched_converter = converted
        ConfigMetadataI18n.convert_to_i18n_keys = staticmethod(converted)

    def _remove_text_converter(self) -> None:
        if not self._converter_owned:
            return
        current = ConfigMetadataI18n.convert_to_i18n_keys
        if current is self._patched_converter:
            ConfigMetadataI18n.convert_to_i18n_keys = staticmethod(
                current._smart_join_original
            )
        self._original_converter = None
        self._patched_converter = None
        self._converter_owned = False

    @staticmethod
    def _restore_raw_text(result: dict[str, Any]) -> dict[str, Any]:
        try:
            result_items = result["ext_group"]["metadata"]["ltm"]["items"]
            source_items = find_profile_metadata_items() or {}
        except (KeyError, TypeError):
            return result

        for key, source_meta in source_items.items():
            if not isinstance(source_meta, dict):
                continue
            attrs = list(source_meta.get("_raw_text_attrs", ()))
            if source_meta.get("raw_text"):
                attrs.extend(("description", "hint", "labels", "subtitles", "name"))
            if not attrs:
                continue
            target_meta = result_items.get(key)
            if not isinstance(target_meta, dict):
                continue
            target_meta.pop("raw_text", None)
            target_meta.pop("_raw_text_attrs", None)
            for attr in attrs:
                if attr in source_meta:
                    target_meta[attr] = source_meta[attr]
        return result
