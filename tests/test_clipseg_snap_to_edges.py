"""What `snap_to_edges` is allowed to do to a CLIPSeg heat map.

The claim: **a coarse heat map whose ramp crosses the object's edge at a
slant is re-drawn so that `> threshold` cuts along the edge of the PICTURE**,
not along the ramp. That ramp is what left a pale collar of background around
the shipped Create Mask pumpkin (`TRACKERS/BUGS/create-mask-returns-poor-mask-card-as`
in the app repository): 2,569 px of wall and floor inside the mask.

Every claim is checked against the same map with the snap OFF, so a test that
passes cannot be one that never discriminated anything.
"""

import importlib.util
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_clipseg():
    if "folder_paths" not in sys.modules:
        stub = types.ModuleType("folder_paths")
        stub.models_dir = os.path.join(_HERE, "_no_such_models_dir")
        sys.modules["folder_paths"] = stub
    spec = importlib.util.spec_from_file_location(
        "anymatix_clipseg_snap", os.path.join(_HERE, "..", "anymatix_clipseg.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


clipseg = _load_clipseg()
THRESHOLD = 0.4
SIZE = 256
EDGE = 128  # the object (orange) is every column < EDGE; the rest is pale grey
ORANGE = (0.85, 0.45, 0.10)
GREY = (0.75, 0.75, 0.74)


def a_picture():
    img = torch.empty(SIZE, SIZE, 3)
    img[:, :EDGE] = torch.tensor(ORANGE)
    img[:, EDGE:] = torch.tensor(GREY)
    return img


def a_coarse_heatmap(centre):
    """A bilinear-looking ramp from 1 to 0, 40 px wide, centred off the true edge."""
    x = torch.arange(SIZE).float()
    row = ((centre + 20 - x) / 40).clamp(0, 1)
    return row.unsqueeze(0).expand(SIZE, SIZE).contiguous()


def mask_right_edge(mask):
    """Column of the first pixel NOT in the mask, on the middle row."""
    row = mask[SIZE // 2]
    outside = (~row).nonzero()
    return int(outside[0]) if len(outside) else SIZE


@pytest.mark.parametrize("centre", [EDGE + 12, EDGE - 12])
def test_the_cut_lands_on_the_edge_of_the_picture(centre):
    img = a_picture()
    heat = a_coarse_heatmap(centre)
    raw_edge = mask_right_edge(heat > THRESHOLD)
    # A window wide enough to span the 40 px ramp, as the node's own radius
    # spans most of one 46 px CLIP token at 1024.
    snapped = clipseg.snap_heatmap_to_edges(heat, img, 48)
    snapped_edge = mask_right_edge(snapped > THRESHOLD)
    # Without the snap the cut follows the ramp, off the edge by several px...
    assert abs(raw_edge - EDGE) >= 4
    # ...with it, the cut is on the edge.
    assert abs(snapped_edge - EDGE) <= 1


def test_a_map_over_one_surface_is_left_as_it_is():
    """No edge in the picture: nothing to snap to, and the frame border is not eaten."""
    img = torch.full((SIZE, SIZE, 3), 0.5)
    heat = torch.full((SIZE, SIZE), 0.9)
    snapped = clipseg.snap_heatmap_to_edges(heat, img, clipseg.snap_radius(SIZE, SIZE))
    assert torch.allclose(snapped, heat, atol=1e-4)


def test_radius_follows_the_token_size():
    assert clipseg.snap_radius(1024, 1024) == 32
    assert clipseg.snap_radius(512, 768) == 24
    assert clipseg.snap_radius(8, 8) == 1


def test_the_node_declares_the_input_off_by_default():
    optional = clipseg.AnymatixCLIPSeg.INPUT_TYPES()["optional"]
    assert optional["snap_to_edges"] == ("BOOLEAN", {"default": False})
