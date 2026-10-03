"""Whether a Create Mask mask reaches the edge of the frame an object crosses.

The claim these tests defend: **the mask of an object that crosses the frame
is non-zero on the crossed edge**, at Mask Blur 0 as at the shipped 6. It was
not: it stopped up to 15 px short (`TRACKERS/BUGS/create-mask-stops-11-15-px-short`
in the app repository).

Where the margin came from, in the order it was looked for:

* Everything AFTER the model -- the resize to the picture, the blur, the
  threshold, the dilation -- was cleared first:
  `test_a_heatmap_reaching_the_border_gives_a_mask_reaching_the_border` held
  on the node as it shipped, and still holds. It is kept so none of those
  steps ever brings a margin back.
* The model's own heat map sags over the outer 16 px of its 352-px grid, on
  every side, whatever the picture shows there. The cause is the one padded
  convolution in CLIPSeg -- a 3x3 `Conv2d(padding=1)` over the token grid,
  padding with zeros -- and `pad_convolutions_by_replication` is the fix (its
  docstring has the measurements).

`test_an_object_crossing_the_frame_reaches_the_crossed_edges` runs `segment()`
end to end against a stand-in with that same padded convolution, and FAILS on
the node as it shipped at PIN 0d5d728: the stand-in's zero padding cuts the
outer block of the heat map, as the real model's did.
`test_without_the_fix_the_frame_is_cut` keeps that failure executable, so the
main test cannot quietly stop discriminating. Pure tensor code: no model, no
download, no GPU.

`test_the_pinned_clipseg_reaches_the_crossed_edges` runs the REAL pinned model
on CPU when its weights are already on disk (`ANYMATIX_CLIPSEG_DIR`, or the
HuggingFace cache), and is skipped otherwise. It never downloads anything.

The module under test imports `folder_paths`, which exists only inside a
running ComfyUI, so it is loaded against a stub, as in test_clipseg_blur.py.
"""

import importlib.util
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
import torch.nn.functional as F  # noqa: E402  (after the skip)

_HERE = os.path.dirname(os.path.abspath(__file__))

# The values the shipped `Create Mask` card runs with.
SHIPPED_BLUR = 6
SHIPPED_THRESHOLD = 0.4
SHIPPED_DILATION = 0

MODEL_INPUT = 352  # CLIPSegProcessor's size for the pinned weights
PATCH = 16  # ViT-B/16: one token is a 16x16 block of the heat map

TEAL = (40, 150, 150)
ORANGE = (240, 130, 20)
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
TEAL_RED = (TEAL[0] / 255 - 0.485) / 0.229  # the red channel as the model receives it
ORANGE_RED = (ORANGE[0] / 255 - 0.485) / 0.229


def _load_clipseg():
    """A fresh `anymatix_clipseg`, with `folder_paths` stubbed out.

    Fresh per test, because some tests replace its module-level functions.
    """
    if "folder_paths" not in sys.modules:
        stub = types.ModuleType("folder_paths")
        stub.models_dir = os.path.join(_HERE, "_no_such_models_dir")
        sys.modules["folder_paths"] = stub
    spec = importlib.util.spec_from_file_location(
        "anymatix_clipseg_frame_edge", os.path.join(_HERE, "..", "anymatix_clipseg.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clipseg():
    return _load_clipseg()


def a_picture(h, w, object_rows, object_cols):
    """A ComfyUI IMAGE: teal, with an orange object over the given rows and columns."""
    picture = torch.empty(1, h, w, 3)
    picture[...] = torch.tensor(TEAL) / 255
    picture[0, object_rows, object_cols] = torch.tensor(ORANGE) / 255
    return picture


class _Output:
    def __init__(self, logits):
        self.logits = logits


class ResizingProcessor:
    """Stands in for CLIPSegProcessor: resize to 352, rescale, normalise, as ViTImageProcessor does."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, text, images, return_tensors):
        picture = torch.from_numpy(np.asarray(images[0])).permute(2, 0, 1).float()[None] / 255.0
        picture = F.interpolate(
            picture, size=(MODEL_INPUT, MODEL_INPUT), mode="bilinear", align_corners=False, antialias=True
        )
        return {
            "pixel_values": (picture - MEAN) / STD,
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
        }


class CertainEverywhere:
    """A model sure of the object on every pixel of its heat map, the border included."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def eval(self):
        return self

    def modules(self):
        return iter(())

    def __call__(self, pixel_values, **_):
        size = pixel_values.shape[-1]
        return _Output(torch.logit(torch.full((1, size, size), 0.9)))  # (1, S, S), as transformers 5.x


class ZeroPaddedHead(torch.nn.Module):
    """CLIPSeg's spatial path in miniature, its one padded convolution included.

    An unpadded 16x16/stride-16 patch embedding (CLIPSeg's is unpadded too),
    then the head of `clipseg-rd64-refined`'s decoder layer for layer: a 3x3
    `Conv2d(padding=1)` over the token grid, ReLU, and two 4x4/stride-4
    transposed convolutions that make each token a 16x16 block of the heat
    map. One channel instead of 64, and fixed weights: a token is 1 on the
    object and 0 off it, the 3x3 convolution averages its neighbourhood, and
    the logit is chosen so a token sure of the object reads 0.90. With zero
    padding a border token misses a third of its neighbourhood and reads 0.23,
    under the 0.4 threshold -- the real model's sag, coarser. It is a stand-in
    for the MECHANISM, not for the model's boundaries inside the picture,
    which it blurs by a token; the tests read the frame only.
    """

    SCALE, BIAS = 10.2, -8.0

    def __init__(self):
        super().__init__()
        nn = torch.nn
        self.patch_embedding = nn.Conv2d(1, 1, PATCH, stride=PATCH, bias=False)
        self.decoder = nn.Module()
        self.decoder.transposed_convolution = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 4, stride=4, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 4, stride=4, bias=False),
        )
        with torch.no_grad():
            self.patch_embedding.weight.fill_(1.0 / PATCH**2)
            self.decoder.transposed_convolution[0].weight.fill_(1.0 / 9)
            self.decoder.transposed_convolution[2].weight.fill_(1.0)
            self.decoder.transposed_convolution[4].weight.fill_(1.0)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def forward(self, pixel_values, **_):
        red = self.patch_embedding(pixel_values[:, :1])
        tokens = ((red - TEAL_RED) / (ORANGE_RED - TEAL_RED)).clamp(0, 1)
        heat = self.decoder.transposed_convolution(tokens)
        return _Output(self.SCALE * heat[:, 0] + self.BIAS)  # (1, S, S), as transformers 5.x


def _inject(monkeypatch, clipseg, model_class):
    """`segment()` runs as shipped, with stand-ins where it imports `transformers`."""
    fake = types.ModuleType("transformers")
    fake.CLIPSegProcessor = ResizingProcessor
    fake.CLIPSegForImageSegmentation = model_class
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(clipseg, "ensure_clipseg_model", lambda: "no model directory is read")
    return clipseg


@pytest.fixture
def certain_model(monkeypatch, clipseg):
    return _inject(monkeypatch, clipseg, CertainEverywhere)


@pytest.fixture
def zero_padded_model(monkeypatch, clipseg):
    return _inject(monkeypatch, clipseg, ZeroPaddedHead)


def _segment(node_module, picture, blur):
    (mask,) = node_module.AnymatixCLIPSeg().segment(
        picture, "orange", blur=blur, threshold=SHIPPED_THRESHOLD, dilation_factor=SHIPPED_DILATION
    )
    return mask


def _edge(mask, axis, index, span):
    return mask[index, span] if axis == "row" else mask[span, index]


@pytest.mark.parametrize("blur", [0, SHIPPED_BLUR])
@pytest.mark.parametrize("dilation", [0, 5])
@pytest.mark.parametrize("h,w", [(1024, 1024), (768, 1280), (1000, 1000)])
def test_a_heatmap_reaching_the_border_gives_a_mask_reaching_the_border(certain_model, blur, dilation, h, w):
    """Everything after the model: resize, blur, threshold, dilation.

    A heat map certain up to its border must give a mask that fills the
    picture. This held on the node as it shipped -- the bilinear resize maps
    outer pixel to outer pixel, the blur divides by real pixels only
    (bugs/mask-blur-eats-mask-frame-border), and max_pool2d pads with -inf --
    and it is pinned so none of them brings a margin back.
    """
    (mask,) = certain_model.AnymatixCLIPSeg().segment(
        torch.rand(1, h, w, 3), "anything", blur=blur, threshold=SHIPPED_THRESHOLD, dilation_factor=dilation
    )
    assert mask.shape == (h, w)
    assert mask.min().item() == 1.0


# (name, h, w, object rows, object cols, crossed edges as (axis, index, span))
# axis "row" + index 0 = the top row, read over `span` columns; and so on. The
# spans stop short of the object's own side inside the picture, which the
# stand-in blurs by a token.
CROSSING = [
    ("frame-filling", 1024, 1024, slice(None), slice(None),
     [("row", 0, slice(None)), ("row", -1, slice(None)), ("col", 0, slice(None)), ("col", -1, slice(None))]),
    ("top-left quadrant", 1024, 1024, slice(0, 512), slice(0, 512),
     [("row", 0, slice(0, 400)), ("col", 0, slice(0, 400))]),
    ("right strip, not square", 768, 1280, slice(None), slice(900, None),
     [("col", -1, slice(None)), ("row", 0, slice(960, None)), ("row", -1, slice(960, None))]),
]


@pytest.mark.parametrize("blur", [0, SHIPPED_BLUR])
@pytest.mark.parametrize("name,h,w,rows,cols,edges", CROSSING, ids=[c[0] for c in CROSSING])
def test_an_object_crossing_the_frame_reaches_the_crossed_edges(zero_padded_model, name, h, w, rows, cols, edges, blur):
    """The whole crossed edge is in the mask, corners included -- at Blur 0 as at 6."""
    mask = _segment(zero_padded_model, a_picture(h, w, rows, cols), blur)
    assert mask.shape == (h, w)
    for axis, index, span in edges:
        assert _edge(mask, axis, index, span).min().item() == 1.0, f"{name}: {axis} {index} stops short of the frame"


def test_the_fix_invents_no_object_on_an_edge_it_does_not_cross(zero_padded_model):
    """The negative half: padding the grid with its own border must not grow the mask where the object is not."""
    mask = _segment(zero_padded_model, a_picture(1024, 1024, slice(0, 512), slice(0, 512)), SHIPPED_BLUR)
    assert mask[-1, :].max().item() == 0.0, "the bottom edge is not crossed"
    assert mask[:, -1].max().item() == 0.0, "the right edge is not crossed"
    assert mask[0, 560:].max().item() == 0.0, "the top edge past the object"
    assert mask[560:, 0].max().item() == 0.0, "the left edge past the object"


def test_without_the_fix_the_frame_is_cut(monkeypatch, zero_padded_model):
    """The defect as it shipped at PIN 0d5d728, kept executable.

    No fix, so the stand-in's convolution pads with zeros, its border tokens
    sag under the threshold, and the mask of a frame-filling object touches
    none of the four edges. If this ever stops holding, the main test above
    has stopped discriminating and proves nothing.
    """
    monkeypatch.setattr(zero_padded_model, "pad_convolutions_by_replication", lambda model: 0)
    mask = _segment(zero_padded_model, a_picture(1024, 1024, slice(None), slice(None)), 0)
    assert mask[0, :].max().item() == 0.0, "the top row used to be cut -- that was the bug"
    assert mask[-1, :].max().item() == 0.0
    assert mask[:, 0].max().item() == 0.0
    assert mask[:, -1].max().item() == 0.0
    rows = torch.nonzero(mask.any(dim=1)).flatten()
    assert rows[0].item() >= PATCH, f"first mask row {rows[0].item()}"


def test_only_the_padded_convolution_changes_and_only_on_the_outer_ring_of_tokens(clipseg):
    """What the fix touches: the padded convolution, nothing else, and the heat map's outer 16 px only."""
    head = ZeroPaddedHead()
    torch.manual_seed(0)
    tokens = torch.rand(1, 1, MODEL_INPUT // PATCH, MODEL_INPUT // PATCH)
    with torch.no_grad():
        before = head.decoder.transposed_convolution(tokens)

    assert clipseg.pad_convolutions_by_replication(head) == 1
    assert head.decoder.transposed_convolution[0].padding_mode == "replicate"
    assert head.patch_embedding.padding_mode == "zeros", "unpadded: nothing to change"

    with torch.no_grad():
        after = head.decoder.transposed_convolution(tokens)
    assert after.shape == before.shape == (1, 1, MODEL_INPUT, MODEL_INPUT)
    assert torch.equal(after[..., PATCH:-PATCH, PATCH:-PATCH], before[..., PATCH:-PATCH, PATCH:-PATCH])
    assert not torch.equal(after[..., :PATCH, :], before[..., :PATCH, :]), "the outer ring is what moves"


def _pinned_weights_on_disk(module):
    """The pinned CLIPSeg directory, if it is already on this machine; never a download."""
    candidates = []
    if os.environ.get("ANYMATIX_CLIPSEG_DIR"):
        candidates.append(os.environ["ANYMATIX_CLIPSEG_DIR"])
    candidates.append(os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "models--CIDAS--clipseg-rd64-refined", "snapshots", module.CLIPSEG_REVISION,
    ))
    for directory in candidates:
        has_config = os.path.isfile(os.path.join(directory, "config.json"))
        has_weights = any(os.path.isfile(os.path.join(directory, w)) for w in module.CLIPSEG_WEIGHT_FILES)
        if has_config and has_weights:
            return directory
    return None


# (name, object rows, object cols, crossed edges, edges it does not reach)
REAL_PICTURES = [
    ("top-left quadrant", slice(0, 512), slice(0, 512),
     [("row", 0, slice(0, 500)), ("col", 0, slice(0, 500))],
     [("row", -1, slice(None)), ("col", -1, slice(None)), ("row", 0, slice(540, None)), ("col", 0, slice(540, None))]),
    ("bottom half", slice(512, None), slice(None),
     [("row", -1, slice(None)), ("col", 0, slice(540, None)), ("col", -1, slice(540, None))],
     [("row", 0, slice(None)), ("col", 0, slice(0, 480)), ("col", -1, slice(0, 480))]),
]


@pytest.mark.parametrize("name,rows,cols,crossed,clear", REAL_PICTURES, ids=[p[0] for p in REAL_PICTURES])
def test_the_pinned_clipseg_reaches_the_crossed_edges(monkeypatch, clipseg, name, rows, cols, crossed, clear):
    """The pinned model itself, on CPU, at the shipped card's values.

    Measured when this was written (torch 2.11, transformers 5.6): without the
    fix the top-left block's mask missed the whole top row and the bottom
    half's missed most of the bottom row at Blur 0; with it every crossed edge
    is in the mask over the object, nothing is on the edges it does not reach,
    and the heat map more than one token inside the border is bit-identical.
    """
    pytest.importorskip("transformers")
    from PIL import Image
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    weights = _pinned_weights_on_disk(clipseg)
    if weights is None:
        pytest.skip("the pinned CLIPSeg weights are not on this machine (set ANYMATIX_CLIPSEG_DIR)")

    picture = a_picture(1024, 1024, rows, cols)

    # The heat map, before and after, on the same loaded model.
    processor = CLIPSegProcessor.from_pretrained(weights, local_files_only=True)
    model = CLIPSegForImageSegmentation.from_pretrained(weights, local_files_only=True).eval()
    inputs = processor(text=["orange"], images=[Image.fromarray((picture[0].numpy() * 255).astype(np.uint8))],
                       return_tensors="pt")
    with torch.no_grad():
        before = model(**inputs).logits.squeeze()
    assert clipseg.pad_convolutions_by_replication(model) == 1
    with torch.no_grad():
        after = model(**inputs).logits.squeeze()
    assert torch.equal(after[PATCH:-PATCH, PATCH:-PATCH], before[PATCH:-PATCH, PATCH:-PATCH])

    # The mask, end to end through segment().
    monkeypatch.setattr(clipseg, "ensure_clipseg_model", lambda: weights)
    for blur in (0, SHIPPED_BLUR):
        mask = _segment(clipseg, picture, blur)
        for axis, index, span in crossed:
            assert _edge(mask, axis, index, span).min().item() == 1.0, f"{name}, blur {blur}: {axis} {index} stops short"
        for axis, index, span in clear:
            assert _edge(mask, axis, index, span).max().item() == 0.0, f"{name}, blur {blur}: {axis} {index} invented"

    monkeypatch.setattr(clipseg, "pad_convolutions_by_replication", lambda model: 0)
    without = _segment(clipseg, picture, 0)
    assert any(_edge(without, axis, index, span).min().item() == 0.0 for axis, index, span in crossed), (
        "without the fix the real model used to stop short of a crossed edge"
    )
