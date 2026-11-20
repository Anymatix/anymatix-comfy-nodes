#!/usr/bin/env python3
"""
THE WHISPER NODE: ITS PACK IS FETCHED AT A PINNED REVISION, AND WHAT IT
RETURNS IS TEXT A PERSON AND A SCRIPT CAN BOTH READ.

The fetcher module imports ComfyUI, so its pack table and url builder are
lifted from the source by `ast` (as in test_chatterbox_pack_url_revision.py);
the language list and the timings formatter live in a plain module.

    python3 -m pytest tests/test_whisper_pack_and_text.py -q
"""
import ast
import importlib.util
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_FETCHER = os.path.join(_ROOT, "anymatix_checkpoint_fetcher.py")
_NAMES = ("_CHATTERBOX_HF_PACK_RE", "_chatterbox_pack_file_url", "CHATTERBOX_PACKS", "WHISPER_PACKS", "MODEL_PACKS", "dirmap")


def _lift():
    with open(_FETCHER, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    wanted = [
        node
        for node in tree.body
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in _NAMES for t in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name in _NAMES)
    ]
    namespace = {"re": re}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), _FETCHER, "exec"), namespace)
    return namespace


NS = _lift()
spec = importlib.util.spec_from_file_location("anymatix_whisper_text", os.path.join(_ROOT, "anymatix_whisper_text.py"))
text = importlib.util.module_from_spec(spec)
spec.loader.exec_module(text)

SHA = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
SHIPPED = f"https://huggingface.co/openai/whisper-large-v3-turbo/resolve/{SHA}"


def test_the_whisper_pack_is_a_known_fetcher_type():
    assert "whisper_pack" in NS["MODEL_PACKS"]
    assert NS["dirmap"]["whisper_pack"] == "whisper_pack"
    # The Chatterbox table is untouched: its own test counts its eleven files.
    assert "whisper_pack" not in NS["CHATTERBOX_PACKS"]


def test_every_whisper_file_is_fetched_at_the_pinned_revision():
    files = NS["WHISPER_PACKS"]["whisper_pack"]["files"]
    assert "model.safetensors" in files and "tokenizer.json" in files and "preprocessor_config.json" in files
    for name in files:
        assert NS["_chatterbox_pack_file_url"](SHIPPED, name) == f"{SHIPPED}/{name}"


def test_a_whisper_url_without_revision_is_refused():
    with pytest.raises(ValueError, match="revision"):
        NS["_chatterbox_pack_file_url"]("https://huggingface.co/openai/whisper-large-v3-turbo", "config.json")


def test_language_options_round_trip_to_whisper_codes():
    assert text.LANGUAGE_OPTIONS[0] == text.AUTO_LANGUAGE
    assert len(text.LANGUAGE_OPTIONS) == 101
    assert text.language_code(text.AUTO_LANGUAGE) is None
    assert text.language_code("Italian (it)") == "it"
    assert text.language_code("Cantonese (yue)") == "yue"
    for option in text.LANGUAGE_OPTIONS[1:]:
        assert text.language_code(option)


@pytest.mark.parametrize("option", ["Italian", "Klingon (tlh)", "it", ""])
def test_an_unknown_language_is_refused(option):
    with pytest.raises(ValueError):
        text.language_code(option)


def test_timings_are_one_line_per_word():
    chunks = [
        {"text": " Ciao,", "timestamp": (0.0, 0.42)},
        {"text": " come", "timestamp": (0.5, 0.71)},
        {"text": " ", "timestamp": (0.71, 0.72)},
        {"text": " stai?", "timestamp": (0.8, None)},
    ]
    assert text.format_timings(chunks) == "0.00-0.42 Ciao,\n0.50-0.71 come\n0.80-0.80 stai?"


def test_no_chunks_is_no_timings():
    assert text.format_timings([]) == ""
