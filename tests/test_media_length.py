"""
The arithmetic that decides how long a clip is.

`snap_frame_count` is pure on purpose: LTX refuses a frame count that is not
8n+1, so a count that is one off is not a slightly wrong video, it is no video
at all, and that has to be provable without starting ComfyUI.
"""

import importlib.util
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "anymatix_media_length", os.path.join(_HERE, "..", "anymatix_media_length.py")
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
snap_frame_count = _MODULE.snap_frame_count


def snap(duration, frame_rate=24.0, multiple_of=8, offset=1, minimum=9, maximum=257):
    return snap_frame_count(duration, frame_rate, multiple_of, offset, minimum, maximum)


@pytest.mark.parametrize(
    "duration,frame_rate,expected",
    [
        (5.0, 24.0, 121),        # 120 raw -> the nearest 8n+1 above
        (5.041666, 24.0, 121),   # the card's shipped length, unchanged
        (10.0, 24.0, 241),
        (6.7, 30.0, 201),        # 201 is exactly 8*25+1
        (4.0, 25.0, 97),
    ],
)
def test_lands_on_a_count_the_model_accepts(duration, frame_rate, expected):
    assert snap(duration, frame_rate) == expected


@pytest.mark.parametrize("duration", [0.0, 0.01, 3.3, 7.7, 12.9, 500.0])
def test_every_result_is_8n_plus_1(duration):
    frames = snap(duration)
    assert (frames - 1) % 8 == 0


def test_clamps_into_the_family_not_past_it():
    """
    A clamp that returns the bound itself can return a count the model refuses:
    257 is 8n+1, but 250 is not. Both ends must land inside the family.
    """
    assert snap(0.0, minimum=10, maximum=250) == 17     # nearest 8n+1 at or above 10
    assert snap(1000.0, minimum=10, maximum=250) == 249  # nearest 8n+1 at or below 250
    assert (snap(0.0, minimum=10, maximum=250) - 1) % 8 == 0
    assert (snap(1000.0, minimum=10, maximum=250) - 1) % 8 == 0


def test_a_negative_duration_is_the_floor_not_a_crash():
    assert snap(-3.0) == 9


def test_refuses_a_frame_rate_that_cannot_produce_frames():
    with pytest.raises(ValueError):
        snap(5.0, frame_rate=0.0)


def test_multiple_of_one_means_no_snapping():
    assert snap(5.0, multiple_of=1, offset=0, minimum=1, maximum=10000) == 120
