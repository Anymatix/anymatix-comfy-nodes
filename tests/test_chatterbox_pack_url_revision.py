#!/usr/bin/env python3
"""
A CHATTERBOX PACK URL NAMES ITS REVISION, AND ONE THAT DOES NOT IS REFUSED.

Until 2026-09-18 `download_chatterbox_hf_pack` appended a literal
`/resolve/main/` to a bare repository url, so the three Voice cards (Speak
text, Cleanup voice, Clone voice) fetched eleven files from a moving branch
(`bugs/the-three-voice-cards-fetch-moving-main`). The pack url is now the
resolve prefix `https://huggingface.co/<org>/<repo>/resolve/<revision>`, the
same shape as every other pinned url the library ships.

The module imports ComfyUI, so the regex and the url builder are lifted from
the shipped source by `ast`, as in test_dwpose_aux_url_revision.py.

    python3 -m pytest tests/test_chatterbox_pack_url_revision.py -q
"""
import ast
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(os.path.dirname(_HERE), "anymatix_checkpoint_fetcher.py")
_NAMES = ("_CHATTERBOX_HF_PACK_RE", "_chatterbox_pack_file_url", "CHATTERBOX_PACKS")


def _load_under_test():
    with open(_SOURCE, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in _NAMES for t in node.targets
        ):
            wanted.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in _NAMES:
            wanted.append(node)
    assert len(wanted) == 3, f"expected {_NAMES} in {_SOURCE}, found {len(wanted)}"
    namespace = {"re": re}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), _SOURCE, "exec"), namespace)
    return namespace["_chatterbox_pack_file_url"], namespace["CHATTERBOX_PACKS"]


file_url, PACKS = _load_under_test()

SHA = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
# Exactly what the three shipped Voice cards carry since 2026-09-18.
SHIPPED = f"https://huggingface.co/ResembleAI/chatterbox/resolve/{SHA}"


@pytest.mark.parametrize("pack_type", sorted(PACKS))
def test_every_file_of_every_pack_is_fetched_at_the_pinned_revision(pack_type):
    for name in PACKS[pack_type]["files"]:
        assert file_url(SHIPPED, name) == (
            f"https://huggingface.co/ResembleAI/chatterbox/resolve/{SHA}/{name}"
        )


def test_the_eleven_files_are_eleven_pinned_urls():
    urls = [file_url(SHIPPED, f) for p in PACKS.values() for f in p["files"]]
    assert len(urls) == 11
    assert all(f"/resolve/{SHA}/" in u for u in urls)
    assert not any("/resolve/main/" in u for u in urls)


def test_a_trailing_slash_is_the_same_url():
    assert file_url(SHIPPED + "/", "ve.pt") == file_url(SHIPPED, "ve.pt")


@pytest.mark.parametrize(
    "url",
    [
        # The shape all three cards shipped before 2026-09-18: no revision.
        "https://huggingface.co/ResembleAI/chatterbox",
        "https://huggingface.co/ResembleAI/chatterbox/",
        # A page url, not a resolve url.
        f"https://huggingface.co/ResembleAI/chatterbox/tree/{SHA}",
        # A FILE url is not a pack prefix: appending would name a path twice.
        f"https://huggingface.co/ResembleAI/chatterbox/resolve/{SHA}/ve.pt",
        "https://example.com/ResembleAI/chatterbox/resolve/main",
        "http://huggingface.co/ResembleAI/chatterbox/resolve/main",
        "",
        None,
    ],
)
def test_a_url_without_a_revision_is_refused(url):
    with pytest.raises(ValueError, match="revision"):
        file_url(url, "ve.pt")
