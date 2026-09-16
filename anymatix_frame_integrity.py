"""
A FRAME THAT IS NOT A PICTURE IS A FAILED RUN, NOT A BLACK PICTURE.

The MP4 writer used to notice that the frames handed to it were non-finite,
write a line about it to the log, replace them with zeros and encode the file
anyway (`np.nan_to_num(_arr, nan=0.0, ...)` -- `nan=0.0` IS the black). The run
then finished as a SUCCESS holding a clip of nothing, with the audio muxed onto
it so that it even played. On 2026-09-10 that is exactly what happened twice on
`fmt-5000`: 6082560 of 6082560 values non-finite in every frame of a
1920x1056 x 177-frame render, a 383 KB file that was almost entirely its audio
track, and a person asking what was wrong with it.
(`bugs/a-run-produced-no-picture-painted-black`.)

This module holds the two things that decision needs and that the save node
cannot hold, because the save node imports `folder_paths` and `comfy.utils` and
so cannot be imported outside a running ComfyUI. Here there is nothing but
numpy, which is why `tests/test_frame_integrity.py` can read it.

WHY THERE IS NO THRESHOLD, AND WHY THAT IS A DECISION RATHER THAN AN OVERSIGHT.

A tempting refinement is to fail only when "enough" of a frame is bad -- a few
stray pixels being a different event from a frame with no numbers in it at all.
It is refused here for two reasons, and the second is the one that decides it:

 1. There is nothing to calibrate against. 364 video writes are logged on
    `fmt-5000`. 362 were clean (`min=0 max=1 all_finite=True`) and 2 were
    entirely non-finite, every channel of every pixel of every frame. The
    measured distribution has no middle, so any number put here would be
    invented and would then be quoted as if it had been measured.
 2. A threshold is itself a silent repair of everything below it -- the precise
    defect this module exists to remove. "Mostly a picture" would still be
    handed back as a success, and the person would still have no way to tell.

So: any non-finite value, in any frame, fails the run. If a real
few-bad-pixels case ever shows up, it arrives here as a reported failure with
its count in the message, which is a fact somebody can act on. It does not
arrive as a clip nobody can explain.
"""

import numpy as np

# The same marker as `app/src/lib/errorAgentNote.ts`, which is where the app's
# side of this contract lives: everything after it is for whoever is DRIVING
# the app, and the card draws only what comes before it. It is duplicated
# rather than shared because the two live in different repositories and this
# one may not import from the app; `tests/test_frame_integrity.py` pins the
# literal so a rename on either side is a red test rather than a paragraph of
# diagnostics landing on a person's screen.
AGENT_NOTE_MARKER = "[ANYMATIX_AGENT_NOTE]"


class FramesAreNotAPicture(RuntimeError):
    """
    Raised when the frames reaching the encoder carry no image data.

    It is its own type so that the save node's encoder-retry loop lets it
    through: trying a second encoder cannot put numbers back into a tensor that
    has none, and catching this with the encoder failures is how it would
    quietly become a black file again.
    """


def nonfinite_count(arr):
    """
    How many values of this frame are not finite. 0 means it is a picture.

    One pass, and the caller already holds the array, so this costs what
    `np.isfinite` costs and nothing more. There is deliberately no pre-flight
    pass over every frame before encoding starts: it would double the
    tensor-to-numpy conversion for the whole clip to answer a question the
    encode loop answers frame by frame anyway.
    """
    return int(np.count_nonzero(~np.isfinite(arr)))


def describe_frames_that_are_not_a_picture(
    frame_index,
    bad_count,
    total_values,
    total_frames,
    shape,
    encoder,
):
    """
    The message the raised `FramesAreNotAPicture` carries.

    Two halves, split by `AGENT_NOTE_MARKER`. The first is what the card shows:
    a state and an action, no identifier, no traceback, nothing that only a
    person with a shell could act on. The action is `Run the card again`
    because it is the one that is actually supported by what was measured --
    the same card, the same machine and the same session produced clean frames
    between the two failures, so a second run genuinely can succeed.

    The second half is for whoever is driving the app, and it carries the
    numbers. ComfyUI appends the traceback after `str(exception)`, so it lands
    inside this half by construction and never on the card.
    """
    person = "This render produced no picture, so no video was saved. Run the card again."
    agent = (
        f"Frame {frame_index} of {total_frames} reached the MP4 writer with "
        f"{bad_count} of {total_values} values non-finite (shape={shape}, "
        f"encoder={encoder!r}).\n\n"
        "The frames were already NaN when the writer received them, so the "
        "encoder is not implicated and neither is the container: the "
        "generation produced no numbers. Grep the ComfyUI log for "
        "'[ANYMATIX_SAVE_MP4] FRAME_NONFINITE' to see every affected frame, "
        "and 'FRAME0_STATS' for the first frame's range.\n\n"
        "bugs/a-run-produced-no-picture-painted-black"
    )
    return f"{person}\n\n{AGENT_NOTE_MARKER}\n{agent}"
