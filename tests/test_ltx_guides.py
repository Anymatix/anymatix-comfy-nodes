"""
`AnymatixAddGuides`: a list of (image, frame) pairs, each applied as one
`LTXVAddGuide` in order, at its exact frame.

ComfyUI is not imported: `comfy_extras.nodes_lt` and `nodes` are stood in by
fakes that record what each guide was called with, which is the whole
contract of this node — the guide arithmetic itself is ComfyUI's.
"""

import importlib.util
import json
import os
import sys
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location("anymatix_ltx_guides", os.path.join(_HERE, "..", "anymatix_ltx_guides.py"))
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
parse_keyframes = _MODULE.parse_keyframes
AnymatixAddGuides = _MODULE.AnymatixAddGuides


class _NodeOutput:
    def __init__(self, *args):
        self.args = args

    @property
    def result(self):
        return self.args


@pytest.fixture
def calls(monkeypatch):
    recorded = []

    class LTXVAddGuide:
        @classmethod
        def execute(cls, positive, negative, vae, latent, image, frame_idx, strength):
            recorded.append((image, frame_idx, strength, positive))
            return _NodeOutput(positive + [frame_idx], negative + [frame_idx], {"samples": latent["samples"] + 1})

    class LoadImage:
        def load_image(self, image):
            return (f"pixels of {image}", None)

    nodes_lt = types.ModuleType("comfy_extras.nodes_lt")
    nodes_lt.LTXVAddGuide = LTXVAddGuide
    monkeypatch.setitem(sys.modules, "comfy_extras", types.ModuleType("comfy_extras"))
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_lt", nodes_lt)
    nodes = types.ModuleType("nodes")
    nodes.LoadImage = LoadImage
    monkeypatch.setitem(sys.modules, "nodes", nodes)
    return recorded


def test_applies_every_pair_in_order_at_its_exact_frame(calls):
    pairs = [{"image": f"k{i}.png", "frame": f} for i, f in enumerate([0, 18, 24, 26, 48, 138, 165, 182, 186, 200])]
    positive, negative, latent = AnymatixAddGuides().add([], [], "vae", {"samples": 0}, json.dumps(pairs), 0.7)
    assert [c[1] for c in calls] == [0, 18, 24, 26, 48, 138, 165, 182, 186, 200]
    assert [c[0] for c in calls] == [f"pixels of k{i}.png" for i in range(10)]
    assert all(c[2] == 0.7 for c in calls)
    # Each guide reads what the one before it wrote.
    assert calls[3][3] == [0, 18, 24]
    assert positive == [0, 18, 24, 26, 48, 138, 165, 182, 186, 200]
    assert latent["samples"] == 10


def test_takes_more_than_any_fixed_count(calls):
    pairs = [{"image": "a.png", "frame": f} for f in range(40)]
    AnymatixAddGuides().add([], [], "vae", {"samples": 0}, json.dumps(pairs), 1.0)
    assert len(calls) == 40


@pytest.mark.parametrize(
    "text,message",
    [
        ("not json", "not JSON"),
        ("[]", "non-empty list"),
        ('{"image": "a.png", "frame": 0}', "non-empty list"),
        ('[["a.png", 0]]', "keyframe 1 is not"),
        ('[{"frame": 0}]', "keyframe 1 names no image"),
        ('[{"image": "a.png", "frame": 0}, {"image": "b.png", "frame": 1.5}]', "keyframe 2's frame"),
        ('[{"image": "a.png", "frame": true}]', "keyframe 1's frame"),
    ],
)
def test_refuses_what_is_not_a_list_of_pairs(text, message):
    with pytest.raises(ValueError, match=message):
        parse_keyframes(text)
