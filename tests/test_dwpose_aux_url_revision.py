#!/usr/bin/env python3
"""
THE DWPOSE URL CARRIES A REVISION, AND THE REVISION IS NOT ALWAYS `main`.

`_destination_path_for_dwpose_aux_url` decides WHERE an annotator checkpoint
goes, and it is reached before any cache check -- so when its pattern refuses a
url, the card dies on every machine, including one that already holds the
weights.

On 2026-09-16 `a8f13a50b` repointed all 75 shipped Hugging Face urls from
`/resolve/main/` to `/resolve/<commit>/` (`todos/shipped-model-urls-name-branch-not-commit`:
"pin them all"). The pattern here still hard-coded `main`, so on 2026-09-17
both ControlNet cards -- `SD 1.5 ControlNet` 7c1f2a44 and `SDXL ControlNet`
2c9480c3 -- refused to run with

    dwpose_aux URL must look like https://huggingface.co/<org>/<repo>/resolve/main/<filename>

which also taught the reader to expect the one shape we no longer ship.

These tests pin BOTH halves of the rule: any revision is accepted, and a url
that is not a Hugging Face resolve url is still refused. The first two cases
are the exact urls `app/library` ships today.

The module under test imports ComfyUI (`comfy`, `nodes`, `spandrel`, ...),
which exists only inside a running ComfyUI, so the two definitions are lifted
from the real source file by `ast` and executed here. That keeps the test
against the SHIPPED text rather than a copy of it.

    python3 -m pytest tests/test_dwpose_aux_url_revision.py -q
"""
import ast
import os
import re
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(os.path.dirname(_HERE), "anymatix_checkpoint_fetcher.py")

_RE_NAME = "_DWPOSE_AUX_HF_RE"
_FN_NAME = "_destination_path_for_dwpose_aux_url"


def _load_under_test():
    """Lift the regex and the destination function out of the shipped file."""
    with open(_SOURCE, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == _RE_NAME for t in node.targets
        ):
            wanted.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == _FN_NAME:
            wanted.append(node)
    assert len(wanted) == 2, (
        f"expected {_RE_NAME} and {_FN_NAME} in {_SOURCE}, found {len(wanted)}"
    )
    namespace = {"re": re, "os": os}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), _SOURCE, "exec"), namespace)
    return namespace[_FN_NAME]


destination_path = _load_under_test()

ROOT = os.path.join("/tmp", "annotator_ckpts")

# The two urls every ControlNet card fetches, exactly as `app/library` ships
# them since the 2026-09-16 repoint.
SHIPPED_YOLOX = (
    "https://huggingface.co/yzd-v/DWPose/resolve/"
    "1a7144101628d69ee7a3768d1ee3a094070dc388/yolox_l.onnx"
)
SHIPPED_DWLL = (
    "https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5/resolve/"
    "359d662a9b33b73f6d0f21732baf8845f17bb4be/dw-ll_ucoco_384_bs5.torchscript.pt"
)


@pytest.mark.parametrize(
    "url,filename",
    [
        # A 40-hex commit: the shape we ship. This is the case that was broken.
        (SHIPPED_YOLOX, "yolox_l.onnx"),
        (SHIPPED_DWLL, "dw-ll_ucoco_384_bs5.torchscript.pt"),
        # A short sha, a tag and a branch are all revisions Hugging Face serves.
        (
            "https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5/resolve/"
            "8bf6f0f/dw-ll_ucoco_384.onnx",
            "dw-ll_ucoco_384.onnx",
        ),
        ("https://huggingface.co/yzd-v/DWPose/resolve/v1.0/yolox_l.onnx", "yolox_l.onnx"),
        # `main` must keep working: nothing about the fix unships a branch url.
        ("https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx", "yolox_l.onnx"),
    ],
)
def test_any_revision_is_accepted_and_the_file_is_named_by_its_filename(url, filename):
    assert destination_path(url, ROOT) == os.path.join(ROOT, filename)


def test_the_revision_never_reaches_the_path():
    """Two revisions of one filename are ONE file on disk.

    comfyui_controlnet_aux looks the checkpoint up by NAME under
    AUX_ANNOTATOR_CKPTS_PATH, so putting the revision in the layout would hide
    the file from the node that consumes it.
    """
    pinned = destination_path(SHIPPED_YOLOX, ROOT)
    on_main = destination_path(
        "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx", ROOT
    )
    assert pinned == on_main
    assert "1a7144101628d69ee7a3768d1ee3a094070dc388" not in pinned


def test_nested_layout_keeps_org_and_repo_and_drops_the_revision():
    """custom_hf_download's own layout is <ckpts>/<org>/<repo>/<filename>."""
    assert destination_path(SHIPPED_YOLOX, ROOT, nested=True) == os.path.join(
        ROOT, "yzd-v", "DWPose", "yolox_l.onnx"
    )


@pytest.mark.parametrize(
    "url",
    [
        # Not Hugging Face at all -- the reason the check exists.
        "https://example.com/foo.onnx",
        "https://cdn.example.com/yzd-v/DWPose/resolve/main/yolox_l.onnx",
        # A repo page url, not a resolve url: `blob` renders HTML.
        "https://huggingface.co/yzd-v/DWPose/blob/main/yolox_l.onnx",
        # A bare repository url names no file.
        "https://huggingface.co/yzd-v/DWPose",
        # A resolve url with a revision and no filename.
        "https://huggingface.co/yzd-v/DWPose/resolve/main/",
        # http, not https.
        "http://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
        "",
        None,
    ],
)
def test_a_url_that_is_not_a_hugging_face_resolve_url_is_refused(url):
    with pytest.raises(ValueError) as excinfo:
        destination_path(url, ROOT)
    message = str(excinfo.value)
    # The message must name what is wrong with THE URL IN HAND ...
    assert "not a Hugging Face resolve url" in message
    if url:
        assert url in message
    # ... and the shape it prescribes must not teach `main`, which is what sent
    # the reader looking for the one url we no longer ship. The echoed url is
    # removed first: it is the user's text, and it may say anything.
    prescription = message.replace(str(url), "")
    assert "/resolve/main/" not in prescription
    assert "/resolve/<revision>/" in prescription


def test_the_shipped_urls_are_the_ones_the_library_actually_carries():
    """If `app/library` ever stops shipping these two, this test is stale.

    It is a reminder, not a coupling: the node pack cannot read `app/library`
    from its own repository, so the urls above are transcribed. They came from
    `TRACKERS/BUGS/both-controlnet-cards-refuse-run-dwpose-fetcher`.
    """
    for url in (SHIPPED_YOLOX, SHIPPED_DWLL):
        assert re.match(r"^https://huggingface\.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/", url)
