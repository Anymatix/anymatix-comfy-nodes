"""
`AnymatixH3AddGuides`: a list of (image, frame) pairs, each applied as one
`MiniMaxH3AddGuide` in order, at its exact frame, after the first `skip`.

ComfyUI is not imported: `comfy_extras.nodes_minimax_h3` and `nodes` are stood
in by fakes that record what each guide was called with, which is the whole
contract of this node — the guide itself is ComfyUI's.
"""

import importlib.util
import json
import os
import sys
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location("anymatix_h3_guides", os.path.join(_HERE, "..", "anymatix_h3_guides.py"))
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
AnymatixH3AddGuides = _MODULE.AnymatixH3AddGuides


class _NodeOutput:
    def __init__(self, *args):
        self.args = args

    @property
    def result(self):
        return self.args


@pytest.fixture
def calls(monkeypatch):
    recorded = []

    class MiniMaxH3AddGuide:
        @classmethod
        def execute(cls, positive, latent, frame_idx, vae=None, audio_vae=None, image=None, audio=None):
            recorded.append({"image": image, "frame": frame_idx, "vae": vae, "latent": latent, "positive": positive})
            return _NodeOutput(positive + [frame_idx])

    class LoadImage:
        def load_image(self, image):
            return (f"pixels of {image}", None)

    nodes_h3 = types.ModuleType("comfy_extras.nodes_minimax_h3")
    nodes_h3.MiniMaxH3AddGuide = MiniMaxH3AddGuide
    monkeypatch.setitem(sys.modules, "comfy_extras", types.ModuleType("comfy_extras"))
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_minimax_h3", nodes_h3)
    nodes = types.ModuleType("nodes")
    nodes.LoadImage = LoadImage
    monkeypatch.setitem(sys.modules, "nodes", nodes)
    return recorded


FRAMES = [0, 8, 19, 30, 41, 52, 63, 74, 85, 96, 107, 118, 129, 140, 151, 162, 173, 191]


def pairs_at(frames):
    return json.dumps([{"image": f"k{i}.png", "frame": f} for i, f in enumerate(frames)])


def test_applies_every_pair_in_order_at_its_exact_frame(calls):
    latent = {"samples": "av latent"}
    (positive,) = AnymatixH3AddGuides().add([], latent, "video vae", pairs_at(FRAMES))
    assert [c["frame"] for c in calls] == FRAMES
    assert [c["image"] for c in calls] == [f"pixels of k{i}.png" for i in range(len(FRAMES))]
    # Every guide gets the video VAE and reads the one latent, unchanged.
    assert all(c["vae"] == "video vae" and c["latent"] is latent for c in calls)
    # Each guide reads the conditioning the one before it wrote.
    assert calls[3]["positive"] == [0, 8, 19]
    assert positive == FRAMES


def test_skip_leaves_out_the_first_pairs(calls):
    (positive,) = AnymatixH3AddGuides().add([], {"samples": 0}, "vae", pairs_at(FRAMES), skip=1)
    assert [c["frame"] for c in calls] == FRAMES[1:]
    assert calls[0]["image"] == "pixels of k1.png"
    assert positive == FRAMES[1:]


def test_skip_past_the_end_adds_nothing(calls):
    (positive,) = AnymatixH3AddGuides().add(["as it came"], {"samples": 0}, "vae", pairs_at([0]), skip=1)
    assert calls == []
    assert positive == ["as it came"]


def test_takes_more_than_any_fixed_count(calls):
    AnymatixH3AddGuides().add([], {"samples": 0}, "vae", pairs_at(range(40)))
    assert len(calls) == 40


def test_refuses_what_the_ltx_node_refuses(calls):
    with pytest.raises(ValueError, match="non-empty list"):
        AnymatixH3AddGuides().add([], {"samples": 0}, "vae", "[]")
    with pytest.raises(ValueError, match="keyframe 2's frame"):
        AnymatixH3AddGuides().add([], {"samples": 0}, "vae", '[{"image": "a.png", "frame": 0}, {"image": "b.png", "frame": 1.5}]')
    assert calls == []


def test_declares_one_positive_output_and_an_optional_skip():
    types_ = AnymatixH3AddGuides.INPUT_TYPES()
    assert list(types_["required"]) == ["positive", "latent", "vae", "keyframes"]
    assert types_["optional"]["skip"] == ("INT", {"default": 0, "min": 0, "max": 9999})
    assert AnymatixH3AddGuides.RETURN_TYPES == ("CONDITIONING",)
    assert AnymatixH3AddGuides.RETURN_NAMES == ("positive",)
