#!/usr/bin/env python3
"""
A BACKGROUND REMOVAL WEIGHT CAN BE FETCHED, AND ITS LOADER TAKES THE PATH.

ComfyUI's LoadBackgroundRemovalModel only accepts a file already in
models/background_removal at validation time; the twin takes the path
AnymatixFetcher returns. The fetcher module imports ComfyUI, so the directory
map and the class's input declaration are lifted from the source by `ast`,
as in test_whisper_pack_and_text.py.

    python3 -m pytest tests/test_background_removal_loader.py -q
"""
import ast
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_FETCHER = os.path.join(_ROOT, "anymatix_checkpoint_fetcher.py")
_INIT = os.path.join(_ROOT, "__init__.py")


def _tree(path):
    with open(path, "r", encoding="utf-8") as f:
        return ast.parse(f.read())


def _dirmap():
    for node in _tree(_FETCHER).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "dirmap" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("no dirmap")


def _class(name):
    for node in _tree(_FETCHER).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"no class {name}")


def test_background_removal_is_a_known_fetcher_type():
    assert _dirmap()["background_removal"] == "background_removal"


def test_the_loader_takes_a_string_and_returns_the_core_type():
    cls = _class("AnymatixLoadBackgroundRemovalModel")
    source = ast.unparse(cls)
    assert "'bg_removal_name': ('STRING', {})" in source
    assert "RETURN_TYPES = ('BACKGROUND_REMOVAL',)" in source
    assert "from comfy.bg_removal_model import load" in source


def test_the_loader_is_registered():
    source = open(_INIT, "r", encoding="utf-8").read()
    assert '"AnymatixLoadBackgroundRemovalModel": AnymatixLoadBackgroundRemovalModel' in source
