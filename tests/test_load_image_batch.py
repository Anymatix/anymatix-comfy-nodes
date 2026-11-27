"""
AnymatixLoadImageBatch: uploaded frames back into the IMAGE batch the consumer reads
(app/specifications/multi-machine-runs-v1.md §3.3; anymatix_load_image_batch.py).

The claims: a list of content-addressed PNGs becomes [N, H, W, 3] float32 in [0, 1], in the order given, each
frame decoded as ComfyUI's LoadImage decodes it (a grey depth frame becomes three equal channels); and the node
refuses, naming the frame, what would make the batch mean other bytes: a missing upload, bytes that are not the
digest they are named by, frames of two sizes, a name that is a path.

Loaded by file, so neither ComfyUI nor the pack's __init__.py is imported.
"""

import hashlib
import importlib.util
import os

import numpy as np
import pytest
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "anymatix_load_image_batch", os.path.join(_HERE, "..", "anymatix_load_image_batch.py")
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

FrameBatchError = _MODULE.FrameBatchError
frame_names = _MODULE.frame_names
load_image_batch = _MODULE.load_image_batch
AnymatixLoadImageBatch = _MODULE.AnymatixLoadImageBatch


def _write_png(folder, array, mode):
    """Writes the frame under its content address, as the app uploads it."""
    tmp = os.path.join(folder, "tmp.png")
    Image.fromarray(array, mode).save(tmp)
    with open(tmp, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    name = f"{digest}.png"
    os.replace(tmp, os.path.join(folder, name))
    return name


def _colour(value, w=4, h=3):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[..., 0] = value
    a[..., 1] = 255 - value
    a[0, 0, 2] = 7
    return a


def test_frames_become_one_batch_in_order(tmp_path):
    names = [_write_png(str(tmp_path), _colour(v), "RGB") for v in (10, 200, 30)]
    batch = load_image_batch([str(tmp_path / n) for n in names], names)
    assert batch.shape == (3, 3, 4, 3)
    assert batch.dtype == np.float32
    assert [round(float(batch[i, 1, 1, 0]) * 255) for i in range(3)] == [10, 200, 30]
    assert float(batch[0, 0, 0, 2]) == pytest.approx(7 / 255)
    assert float(batch.min()) >= 0.0 and float(batch.max()) <= 1.0


def test_a_grey_depth_frame_is_three_equal_channels(tmp_path):
    grey = np.arange(12, dtype=np.uint8).reshape(3, 4) * 20
    name = _write_png(str(tmp_path), grey, "L")
    batch = load_image_batch([str(tmp_path / name)], [name])
    assert batch.shape == (1, 3, 4, 3)
    assert np.array_equal(batch[0, ..., 0], batch[0, ..., 1])
    assert np.array_equal(batch[0, ..., 1], batch[0, ..., 2])
    assert np.allclose(batch[0, ..., 0], grey.astype(np.float32) / 255.0)


def test_a_missing_upload_names_the_frame(tmp_path):
    name = _write_png(str(tmp_path), _colour(1), "RGB")
    missing = "f" * 64 + ".png"
    with pytest.raises(FrameBatchError, match=r"frame 2 \(f{64}\.png\) was not uploaded"):
        load_image_batch([str(tmp_path / name), str(tmp_path / missing)], [name, missing])


def test_bytes_that_are_not_their_name_are_refused(tmp_path):
    name = _write_png(str(tmp_path), _colour(1), "RGB")
    wrong = "0" * 64 + ".png"
    os.replace(tmp_path / name, tmp_path / wrong)
    with pytest.raises(FrameBatchError, match="frame 1 .* holds other bytes"):
        load_image_batch([str(tmp_path / wrong)], [wrong])


def test_frames_of_two_sizes_are_refused(tmp_path):
    a = _write_png(str(tmp_path), _colour(1, w=4, h=3), "RGB")
    b = _write_png(str(tmp_path), _colour(2, w=5, h=3), "RGB")
    with pytest.raises(FrameBatchError, match=r"frame 2 .* is 5x3, frame 1 is 4x3"):
        load_image_batch([str(tmp_path / a), str(tmp_path / b)], [a, b])


def test_the_list_is_one_name_per_line_and_never_a_path():
    assert frame_names("a.png\n\n  b.png \n") == ["a.png", "b.png"]
    with pytest.raises(FrameBatchError, match="no frames"):
        frame_names("\n  \n")
    for bad in ("../x.png", "sub/x.png", "..", "a\\b.png"):
        with pytest.raises(FrameBatchError, match="is not a file name"):
            frame_names(f"ok.png\n{bad}")


def test_the_node_declares_what_the_app_dump_says():
    """The entry the app types it by (app/src/machines/ComfyUI/comfy_object_info.json) is written from these."""
    assert AnymatixLoadImageBatch.INPUT_TYPES() == {
        "required": {"files": ("STRING", {"default": "", "multiline": True})}
    }
    assert AnymatixLoadImageBatch.RETURN_TYPES == ("IMAGE",)
    assert AnymatixLoadImageBatch.RETURN_NAMES == ("images",)
    assert AnymatixLoadImageBatch.FUNCTION == "load"
    assert AnymatixLoadImageBatch.CATEGORY == "Anymatix"
