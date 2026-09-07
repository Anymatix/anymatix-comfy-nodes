"""
Bridge AnymatixFetcher STRING (Chatterbox pack name) into upstream Chatterbox COMBO inputs
without patching ComfyUI-Chatterbox.

The fetcher must run first so files exist under ComfyUI/models/tts/chatterbox/<pack>.
"""

from __future__ import annotations

import importlib.util
import os
from typing import List


def _handler_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "ComfyUI-Chatterbox", "modules", "chatterbox_handler.py")
    )


def _chatterbox_combo_options() -> List[str]:
    hp = _handler_path()
    if not os.path.isfile(hp):
        return ["resembleai_default_voice"]
    try:
        spec = importlib.util.spec_from_file_location("_chatterbox_handler_anymatix_bridge", hp)
        if spec is None or spec.loader is None:
            return ["resembleai_default_voice"]
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        names = mod.get_chatterbox_model_pack_names()
        out = list(names) if names else ["resembleai_default_voice"]
        return out
    except Exception:
        return ["resembleai_default_voice"]


# COMBO output widgets use a list of allowed values as the first RETURN_TYPES entry.
RETURN_COMBO_OPTIONS = _chatterbox_combo_options()


class AnymatixChatterboxPackFromFetchedName:
    """
    Input: STRING from AnymatixFetcher output slot 0 (e.g. resembleai_default_voice).
    Output: same value with COMBO typing compatible with Chatterbox TTS/VC pack widgets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fetched_pack_name": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Wire AnymatixFetcher output (chatterbox_model_pack) here.",
                    },
                )
            }
        }

    RETURN_TYPES = (RETURN_COMBO_OPTIONS,)
    RETURN_NAMES = ("model_pack_name",)
    FUNCTION = "apply"
    CATEGORY = "Anymatix/audio"
    DESCRIPTION = "Converts AnymatixFetcher STRING pack id to Chatterbox COMBO input (no ComfyUI-Chatterbox edits)."

    def apply(self, fetched_pack_name: str):
        name = (fetched_pack_name or "").strip()
        if not name:
            raise ValueError("AnymatixChatterboxPackFromFetchedName: empty pack name from fetcher.")
        allowed = set(_chatterbox_combo_options())
        if name not in allowed:
            raise ValueError(
                f"Pack {name!r} is not among Chatterbox folders under models/tts/chatterbox: {sorted(allowed)!r}. "
                "Ensure AnymatixFetcher ran successfully for this URL."
            )
        return (name,)
