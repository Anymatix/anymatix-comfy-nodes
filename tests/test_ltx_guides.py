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

    class LTXVPreprocess:
        @classmethod
        def execute(cls, image, img_compression):
            return _NodeOutput(f"{image} compressed {img_compression}")

    nodes_lt = types.ModuleType("comfy_extras.nodes_lt")
    nodes_lt.LTXVAddGuide = LTXVAddGuide
    nodes_lt.LTXVPreprocess = LTXVPreprocess
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


def test_preprocesses_every_keyframe_when_asked(calls):
    pairs = [{"image": "a.png", "frame": 0}, {"image": "b.png", "frame": 64}]
    AnymatixAddGuides().add([], [], "vae", {"samples": 0}, json.dumps(pairs), 0.7, img_compression=18)
    assert [c[0] for c in calls] == ["pixels of a.png compressed 18", "pixels of b.png compressed 18"]


def test_passes_pictures_through_without_compression(calls):
    pairs = [{"image": "a.png", "frame": 0}]
    AnymatixAddGuides().add([], [], "vae", {"samples": 0}, json.dumps(pairs), 0.7)
    assert [c[0] for c in calls] == ["pixels of a.png"]


def test_takes_more_than_any_fixed_count(calls):
    pairs = [{"image": "a.png", "frame": f} for f in range(40)]
    AnymatixAddGuides().add([], [], "vae", {"samples": 0}, json.dumps(pairs), 1.0)
    assert len(calls) == 40


def test_a_key_with_its_own_strength_is_guided_with_it(calls):
    pairs = [
        {"image": "a.png", "frame": 0},
        {"image": "b.png", "frame": 18, "strength": 0.25},
        {"image": "c.png", "frame": 40, "strength": 1},
        {"image": "d.png", "frame": 64},
    ]
    AnymatixAddGuides().add([], [], "vae", {"samples": 0}, json.dumps(pairs), 0.7)
    # Each key only its own value, the node's own strength for a key with none; nothing between keys.
    assert [c[2] for c in calls] == [0.7, 0.25, 1.0, 0.7]


def test_parse_returns_the_strength_only_where_a_pair_has_one():
    text = '[{"image": "a.png", "frame": 0}, {"image": "b.png", "frame": 9, "strength": 0.5}]'
    assert parse_keyframes(text) == [("a.png", 0, None), ("b.png", 9, 0.5)]


def test_input_types_are_unchanged():
    types_ = AnymatixAddGuides.INPUT_TYPES()
    assert list(types_["required"]) == ["positive", "negative", "vae", "latent", "keyframes", "strength"]
    assert types_["required"]["strength"] == ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
    assert list(types_["optional"]) == ["img_compression"]


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
        ('[{"image": "a.png", "frame": 0, "strength": "0.5"}]', "keyframe 1's strength"),
        ('[{"image": "a.png", "frame": 0, "strength": true}]', "keyframe 1's strength"),
        ('[{"image": "a.png", "frame": 0, "strength": null}]', "keyframe 1's strength"),
        ('[{"image": "a.png", "frame": 0}, {"image": "b.png", "frame": 2, "strength": NaN}]', "keyframe 2's strength"),
    ],
)
def test_refuses_what_is_not_a_list_of_pairs(text, message):
    with pytest.raises(ValueError, match=message):
        parse_keyframes(text)
