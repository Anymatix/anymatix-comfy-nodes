"""
HOW LONG THE CLIP IS, ASKED OF THE FILE INSTEAD OF THE PERSON.

A workflow that takes audio has three candidate lengths — the audio, a guide
video, and whatever number the card asks for — and only one of them can win.
Asking the user is the worst of the three: the file already knows, and every
mismatch between the number typed and the number that is true is a clip that
ends early or runs on into silence.

Nothing in the node pack could answer it. `TrimAudioDuration` *takes* a
duration; no node *reports* one. So the length had to be typed, and
`TRACKERS/TODOS/ia2v-video-should-last-as-long-as` blocked on exactly this:
"establish where the length can be read before the graph runs".

These two nodes are that. They are deliberately not one node: measuring a file
and choosing a frame count are different questions, and the second one has no
audio in it at all — a guide video's length goes through `AnymatixFrameCount`
by the same road.
"""

import math


class AnymatixAudioDuration:
    """
    How many seconds of audio there are.

    ComfyUI's AUDIO is `{"waveform": tensor[..., samples], "sample_rate": int}`,
    so the duration is a division — but it is a division nobody could reach
    from inside a graph, which is the whole point of exposing it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The audio to measure."}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT")
    RETURN_NAMES = ("duration", "sample_rate", "samples")
    FUNCTION = "measure"
    CATEGORY = "Anymatix"
    DESCRIPTION = "Report the length of an audio clip in seconds, so a workflow can derive its frame count from the file instead of asking for it."

    def measure(self, audio):
        if not isinstance(audio, dict):
            raise ValueError(f"AUDIO must be a dict, got {type(audio).__name__}")

        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")

        if waveform is None:
            raise ValueError("AUDIO has no waveform")
        if not sample_rate:
            raise ValueError("AUDIO has no sample_rate")

        samples = int(waveform.shape[-1])
        sample_rate = int(sample_rate)
        return (samples / sample_rate, sample_rate, samples)


def snap_frame_count(
    duration: float,
    frame_rate: float,
    multiple_of: int,
    offset: int,
    minimum: int,
    maximum: int,
) -> int:
    """
    The nearest frame count of the form `multiple_of * n + offset`, clamped.

    Pure, and separate from the node, so the arithmetic can be tested without a
    ComfyUI process. LTX wants 8n+1 — 97, 121, 161 — and a count that is not of
    that shape is not a slightly wrong video, it is a refused one.
    """
    if frame_rate <= 0:
        raise ValueError(f"frame_rate must be positive, got {frame_rate}")
    if multiple_of < 1:
        raise ValueError(f"multiple_of must be at least 1, got {multiple_of}")
    if minimum > maximum:
        raise ValueError(f"minimum {minimum} is above maximum {maximum}")

    raw = max(0.0, duration) * frame_rate
    steps = int(math.floor(((raw - offset) / multiple_of) + 0.5))
    frames = multiple_of * max(0, steps) + offset

    # Clamp INTO the family, not past it: the nearest valid count at or inside
    # each end, so a clamp can never hand back something the model refuses.
    lo = multiple_of * int(math.ceil((minimum - offset) / multiple_of)) + offset
    hi = multiple_of * int(math.floor((maximum - offset) / multiple_of)) + offset
    if hi < lo:
        hi = lo
    return min(max(frames, lo), hi)


class AnymatixFrameCount:
    """
    A duration in seconds becomes a frame count the model will accept.

    Also returns the duration that count actually represents, which is NOT the
    one that came in: the snap moves it. Feeding that number back into the trim
    is what keeps the audio and the video the same length instead of nearly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 3600.0, "step": 0.01, "tooltip": "Length in seconds."}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "multiple_of": ("INT", {"default": 8, "min": 1, "max": 1024, "tooltip": "Frame counts must be multiple_of * n + offset. LTX wants 8n+1."}),
                "offset": ("INT", {"default": 1, "min": 0, "max": 1024}),
                "minimum": ("INT", {"default": 9, "min": 1, "max": 100000}),
                "maximum": ("INT", {"default": 257, "min": 1, "max": 100000}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("frames", "duration")
    FUNCTION = "compute"
    CATEGORY = "Anymatix"
    DESCRIPTION = "Turn a duration into a frame count the model accepts, and report the duration that count really is."

    def compute(self, duration, frame_rate, multiple_of, offset, minimum, maximum):
        frames = snap_frame_count(duration, frame_rate, multiple_of, offset, minimum, maximum)
        return (frames, frames / frame_rate)
