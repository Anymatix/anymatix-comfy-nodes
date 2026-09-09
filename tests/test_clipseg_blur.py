"""What the blur is allowed to do to a mask that reaches the frame.

The claim these tests defend: **blurring the heat map must not invent pixels
outside the image.** `F.avg_pool2d` pads with zeros and, with PyTorch's default
`count_include_pad=True`, counts those zeros in the denominator — so a heat map
the model was certain about is dragged towards 0 within `kernel_size // 2` of
every edge, and the threshold on the next line cuts it away. Measured on the
shipped `Create Mask` settings (`blur = 6`, `threshold = 0.4`): a certain 0.9
arrived at the threshold as 0.261 in a corner and 0.485 mid-edge.

Every test here is written to FAIL under the old keyword — see
`test_old_behaviour_would_have_eaten_the_border`, which reproduces it
explicitly so the fix cannot quietly regress into "the test never discriminated
anything".

The module under test imports `folder_paths`, which exists only inside a
running ComfyUI, so it is loaded here against a stub. Nothing in
`blur_heatmap()` touches it.
"""

import importlib.util
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402  (after the skip)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_clipseg():
    """Import `anymatix_clipseg` with `folder_paths` stubbed out."""
    if "folder_paths" not in sys.modules:
        stub = types.ModuleType("folder_paths")
        stub.models_dir = os.path.join(_HERE, "_no_such_models_dir")
        sys.modules["folder_paths"] = stub
    spec = importlib.util.spec_from_file_location(
        "anymatix_clipseg", os.path.join(_HERE, "..", "anymatix_clipseg.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


blur_heatmap = _load_clipseg().blur_heatmap

# The values the shipped `Create Mask` card runs with.
SHIPPED_BLUR = 6
SHIPPED_THRESHOLD = 0.4
CERTAIN = 0.9  # what CLIPSeg gives where it is sure


def a_certain_heatmap(size=64, value=CERTAIN):
    """A plateau of certainty covering the whole frame, edges included."""
    return torch.full((size, size), float(value))


def old_blur(preds, blur):
    """The blur exactly as it shipped at PIN `88eb8ad9`: zeros in the denominator."""
    kernel_size = int(blur) * 2 + 1
    return F.avg_pool2d(
        preds.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        count_include_pad=True,
    ).squeeze()


@pytest.mark.parametrize("blur", [2, SHIPPED_BLUR, 10])
def test_a_plateau_touching_the_frame_survives_the_blur(blur):
    """Certainty at the corner must still read as certainty after blurring."""
    out = blur_heatmap(a_certain_heatmap(), blur)

    corner = out[0, 0].item()
    mid_edge = out[0, out.shape[1] // 2].item()
    centre = out[out.shape[0] // 2, out.shape[1] // 2].item()

    assert corner == pytest.approx(CERTAIN, abs=1e-5)
    assert mid_edge == pytest.approx(CERTAIN, abs=1e-5)
    assert centre == pytest.approx(CERTAIN, abs=1e-5)
    assert out.min().item() == pytest.approx(CERTAIN, abs=1e-5)


def test_the_whole_border_stays_above_the_shipped_threshold():
    """Every one of the four borders, at the settings the card actually ships."""
    out = blur_heatmap(a_certain_heatmap(), SHIPPED_BLUR)
    mask = (out > SHIPPED_THRESHOLD).float()

    assert mask[0, :].min().item() == 1.0, "top row was cut"
    assert mask[-1, :].min().item() == 1.0, "bottom row was cut"
    assert mask[:, 0].min().item() == 1.0, "left column was cut"
    assert mask[:, -1].min().item() == 1.0, "right column was cut"
    assert mask.mean().item() == 1.0


def test_old_behaviour_would_have_eaten_the_border():
    """The measurement that opened the bug, kept executable.

    If this ever stops holding, the test above has stopped discriminating and
    proves nothing.
    """
    out = old_blur(a_certain_heatmap(), SHIPPED_BLUR)

    corner = out[0, 0].item()
    mid_edge = out[0, out.shape[1] // 2].item()

    assert corner == pytest.approx(0.261, abs=0.002)
    assert mid_edge == pytest.approx(0.485, abs=0.002)
    assert corner < SHIPPED_THRESHOLD, "the corner used to be discarded — that was the bug"
    assert out[out.shape[0] // 2, out.shape[1] // 2].item() == pytest.approx(CERTAIN, abs=1e-5)


def test_the_interior_is_untouched_by_the_change():
    """Away from the border the two agree — the fix moves the edge and nothing else."""
    torch.manual_seed(0)
    preds = torch.rand(64, 64)
    pad = SHIPPED_BLUR  # kernel 13 -> padding 6

    new = blur_heatmap(preds, SHIPPED_BLUR)
    old = old_blur(preds, SHIPPED_BLUR)

    interior = (slice(pad, -pad), slice(pad, -pad))
    assert torch.allclose(new[interior], old[interior], atol=1e-6)
    assert not torch.allclose(new[0, :], old[0, :], atol=1e-3), "the border must differ"


def test_blur_zero_is_a_no_op():
    preds = a_certain_heatmap()
    assert blur_heatmap(preds, 0) is preds


def test_the_mask_is_binary_after_the_threshold():
    """No feathered boundary: `segment()` thresholds AFTER the blur.

    A partial edge would show up here as a value strictly between 0 and 1.
    """
    size = 64
    preds = torch.zeros(size, size)
    preds[: size // 2, :] = CERTAIN  # a soft-edged half-frame plateau
    preds = blur_heatmap(preds, SHIPPED_BLUR)

    mask = (preds > SHIPPED_THRESHOLD).float()
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert 0.0 < mask.mean().item() < 1.0, "the fixture should produce both values"


def test_dilation_does_not_invent_zeros_at_the_frame():
    """The other pooling call in `segment()` is safe, and this pins that.

    `F.max_pool2d` pads with -inf, not 0, so a mask touching the frame is
    dilated outwards rather than shaved back. If that ever changes, the same
    class of bug reappears one line lower.
    """
    size = 64
    dilation_factor = 5
    mask = torch.zeros(size, size)
    mask[:, : size // 2] = 1.0  # touches the top, bottom and left edges

    out = F.max_pool2d(
        mask.unsqueeze(0).unsqueeze(0),
        kernel_size=2 * dilation_factor + 1,
        stride=1,
        padding=dilation_factor,
    ).squeeze()

    assert out.shape == mask.shape
    assert out[0, 0].item() == 1.0
    assert out[-1, 0].item() == 1.0
    assert out[:, 0].min().item() == 1.0, "the left column was eaten by the dilation"
    assert out.sum().item() > mask.sum().item(), "dilation must grow the mask, not shrink it"
