#!/usr/bin/env python3
"""
CLIPSEG IS FETCHED FROM A PINNED REVISION, NOT A MOVING `main`.

`bugs/create-mask-fetches-clipseg-from-moving-main`: `anymatix_clipseg.py`
built its resolve url with a literal `main`, fetched for `Create Mask` and the
three `is:dev` inpainting cards. No library url exists for it (the node takes
no url input), so the standing rule
(`todos/shipped-model-urls-name-branch-not-commit`, 2026-09-16 -- "pin them
all") is applied as a pinned constant in the node itself.

The module under test imports `folder_paths`, which exists only inside a
running ComfyUI, so the two constants are lifted from the shipped source by
`ast`, as the Chatterbox and DWPose revision tests do.

    python3 -m pytest tests/test_clipseg_url_revision.py -q
"""
import ast
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(os.path.dirname(_HERE), "anymatix_clipseg.py")
_NAMES = ("CLIPSEG_MODEL_ID", "CLIPSEG_REVISION", "CLIPSEG_BASE_URL")


def _load_under_test():
    with open(_SOURCE, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in _NAMES for t in node.targets)
    ]
    assert len(wanted) == 3, f"expected {_NAMES} in {_SOURCE}, found {len(wanted)}"
    namespace = {}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), _SOURCE, "exec"), namespace)
    return namespace["CLIPSEG_MODEL_ID"], namespace["CLIPSEG_REVISION"], namespace["CLIPSEG_BASE_URL"]


def test_base_url_names_a_commit_not_main():
    model_id, revision, base_url = _load_under_test()
    assert model_id == "CIDAS/clipseg-rd64-refined"
    # A 40-char hex commit, never the literal branch name.
    assert re.fullmatch(r"[0-9a-f]{40}", revision), revision
    assert revision != "main"
    assert base_url == f"https://huggingface.co/{model_id}/resolve/{revision}"
    assert "/resolve/main" not in base_url
