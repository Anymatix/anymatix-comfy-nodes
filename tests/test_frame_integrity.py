"""
What the MP4 writer does with frames that are not a picture.

The claim these tests defend: **a run that produced no image data fails and
says so.** It used to paint the frames black and report success -- on
2026-09-10, twice on `fmt-5000`, a 1920x1056 x 177-frame render arrived at the
writer with 6082560 of 6082560 values non-finite in every frame, and what came
back was a 383 KB file that was almost entirely its audio track, presented as a
finished video. (`bugs/a-run-produced-no-picture-painted-black`.)

`test_the_old_behaviour_painted_it_black` reproduces the old line explicitly,
so these tests cannot quietly stop discriminating anything.

The module under test imports neither `comfy` nor `folder_paths`, which is why
it exists as its own module: `anymatix_save_animated_mp4.py` cannot be imported
outside a running ComfyUI, and anything kept inside it cannot be tested at all.
The one claim that IS about that file is made by reading its source.
"""

import importlib.util
import os
import re

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "anymatix_frame_integrity", os.path.join(_HERE, "..", "anymatix_frame_integrity.py")
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

AGENT_NOTE_MARKER = _MODULE.AGENT_NOTE_MARKER
FramesAreNotAPicture = _MODULE.FramesAreNotAPicture
nonfinite_count = _MODULE.nonfinite_count
describe = _MODULE.describe_frames_that_are_not_a_picture

# The shape that actually failed, scaled down: the ratio of bad values to total
# is what the message reports, and 3 channels is what makes it a frame.
FRAME_SHAPE = (4, 5, 3)


def a_picture():
    """A frame with numbers in it, in the range the writer expects."""
    return np.full(FRAME_SHAPE, 0.5, dtype=np.float32)


def no_picture():
    """The measured failure: every channel of every pixel non-finite."""
    return np.full(FRAME_SHAPE, np.nan, dtype=np.float32)


class TestWhatCountsAsAPicture:
    def test_a_frame_with_numbers_in_it_is_a_picture(self):
        assert nonfinite_count(a_picture()) == 0

    def test_an_all_nan_frame_is_bad_in_every_value(self):
        arr = no_picture()
        assert nonfinite_count(arr) == arr.size == 60

    def test_infinities_count_too_not_just_nan(self):
        arr = a_picture()
        arr[0, 0, 0] = np.inf
        arr[1, 2, 1] = -np.inf
        assert nonfinite_count(arr) == 2

    def test_there_is_no_threshold_one_bad_value_is_reported(self):
        """
        Deliberate, and the reason is in `anymatix_frame_integrity.py`: 362 of
        364 logged writes were clean and 2 were entirely non-finite, so the
        measured distribution has no middle to calibrate a threshold against --
        and a threshold would itself be a silent repair of everything below it.
        """
        arr = a_picture()
        arr[3, 4, 2] = np.nan
        assert nonfinite_count(arr) == 1


class TestTheOldBehaviourThisReplaces:
    def test_the_old_behaviour_painted_it_black(self):
        """
        The line that shipped at PIN `4247a596a`, run on the measured frame. It
        is here so that "the writer must refuse" is a claim with something to
        be false about: this is what success used to look like.
        """
        painted = np.nan_to_num(no_picture(), nan=0.0, posinf=1.0, neginf=0.0)
        as_bytes = np.rint(255.0 * np.clip(painted, 0.0, 1.0)).astype(np.uint8)
        assert as_bytes.max() == 0  # every pixel black
        assert nonfinite_count(painted) == 0  # and indistinguishable from a picture


class TestWhatTheRunSays:
    def message(self):
        return describe(
            frame_index=0,
            bad_count=6082560,
            total_values=6082560,
            total_frames=177,
            shape=(1056, 1920, 3),
            encoder="libx264",
        )

    def halves(self):
        person, _, agent = self.message().partition(AGENT_NOTE_MARKER)
        return person.strip(), agent.strip()

    def test_the_marker_is_the_one_the_app_splits_on(self):
        """
        Pinned as a literal because the app's copy lives in another repository
        (`app/src/lib/errorAgentNote.ts`). A rename on either side has to break
        a test here rather than drop a paragraph of diagnostics onto a card.
        """
        assert AGENT_NOTE_MARKER == "[ANYMATIX_AGENT_NOTE]"
        assert AGENT_NOTE_MARKER in self.message()

    def test_the_person_reads_a_state_and_an_action(self):
        person, _ = self.halves()
        assert "produced no picture" in person
        assert "Run the card again" in person

    def test_the_person_does_not_read_numbers_paths_or_identifiers(self):
        person, _ = self.halves()
        assert not re.search(r"\d", person)
        assert "bugs/" not in person
        assert "libx264" not in person
        assert "NaN" not in person and "non-finite" not in person
        assert "ANYMATIX_SAVE_MP4" not in person

    def test_the_driver_gets_the_counts_and_where_to_look(self):
        _, agent = self.halves()
        assert "6082560 of 6082560" in agent
        assert "Frame 0 of 177" in agent
        assert "(1056, 1920, 3)" in agent
        assert "libx264" in agent
        assert "FRAME_NONFINITE" in agent
        assert "bugs/a-run-produced-no-picture-painted-black" in agent

    def test_the_traceback_comfyui_appends_lands_in_the_drivers_half(self):
        """
        ComfyUI reports `str(exception)` as `exception_message` and the app
        concatenates the traceback after it, so anything the app appends falls
        on the far side of the marker by construction.
        """
        whole = self.message() + "\n" + 'Traceback (most recent call last):\n  File "x", line 1'
        person, _, agent = whole.partition(AGENT_NOTE_MARKER)
        assert "Traceback" not in person
        assert "Traceback" in agent

    def test_it_is_raised_as_its_own_type_so_the_encoder_loop_lets_it_through(self):
        assert issubclass(FramesAreNotAPicture, RuntimeError)


class TestTheWriterItself:
    """
    One claim about `anymatix_save_animated_mp4.py`, read from its source
    because the module cannot be imported without ComfyUI.
    """

    def source(self):
        with open(os.path.join(_HERE, "..", "anymatix_save_animated_mp4.py")) as f:
            return f.read()

    def test_the_writer_no_longer_repairs_frames_it_knows_are_broken(self):
        assert "nan_to_num" not in self.source()

    def test_the_writer_raises_instead(self):
        assert "raise FramesAreNotAPicture(" in self.source()
