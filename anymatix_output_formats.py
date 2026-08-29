"""
How a result is written down — the machine's half of the renderer's
`OutputFormats.ts`.

The renderer names a rung ("TIFF 16-bit", "ProRes 422 HQ", "WAV 24-bit") and
turns it into save inputs; this module is what those inputs mean in bytes. It
is a module of its own, and it imports neither `comfy` nor `folder_paths`, for
one reason: the save nodes cannot be imported outside ComfyUI, so anything
kept inside them cannot be tested. Every format decision that can be made
without a running server is made here, and `tests/test_output_formats.py`
exercises it directly.

WHAT IS NOT NEGOTIABLE HERE

  A rung that writes fewer bits than it claims is worse than a rung that
  refuses. The address says "16-bit"; if the file is 8-bit the two have
  disagreed and nothing downstream can tell. So a format this build cannot
  write raises, and the run fails loudly, rather than falling back to PNG.

WHY OPENCV AND NOT PILLOW

  Pillow cannot write a 16-bit RGB PNG at all — its `I;16` mode is single
  channel. It cannot write OpenEXR, and its AVIF support depends on a plugin
  we do not ship. OpenCV writes all four, and it is already a declared
  dependency. Pillow stays for the 8-bit formats, which it writes well and
  which every existing card uses.
"""

import os

# OpenCV reads this at import time and never again: set after `import cv2` and
# every EXR write returns False with no explanation.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - opencv is in requirements.txt
    CV2_AVAILABLE = False

from PIL import Image


# ---------------------------------------------------------------- images ----

#: The extensions the image save node accepts. `png` and `tiff` also take a
#: bit depth; the rest are 8-bit by definition.
IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "gif", "tiff", "webp", "bmp", "exr", "avif"]

#: Extensions that need OpenCV whatever the depth.
_CV2_ONLY = ("exr", "avif")


def srgb_to_linear(image):
    """
    The transfer function EXR expects.

    A ComfyUI IMAGE is 0..1 in display space. An OpenEXR file whose rung is
    called "half float, linear" and which holds sRGB values is mislabelled, and
    it lands in a compositor that will treat it as linear and grade it wrong.
    This is the one place where writing the tensor unchanged would be the bug.
    """
    image = np.asarray(image, dtype=np.float32)
    low = image / 12.92
    high = ((image + 0.055) / 1.055) ** 2.4
    return np.where(image <= 0.04045, low, high).astype(np.float32)


def _to_cv2_channel_order(image):
    """OpenCV writes BGR(A); a ComfyUI IMAGE is RGB(A)."""
    if image.ndim == 2 or image.shape[2] == 1:
        return image
    if image.shape[2] == 4:
        return image[:, :, [2, 1, 0, 3]]
    return image[:, :, ::-1]


def write_image(
    image,
    path,
    extension="png",
    quality=100,
    lossless_webp=False,
    bit_depth=8,
    fast=False,
):
    """
    Write one frame, already float 0..1, in the format the rung asked for.

    `image` is HxW, HxWx1, HxWx3 or HxWx4. `fast` trades compression for time
    and is set when a batch is large enough that the difference is felt; it
    never changes which format is written.

    Raises `RuntimeError` when the build cannot write the format. See the
    module docstring for why that is not a fallback.
    """
    image = np.asarray(image, dtype=np.float32)
    extension = extension.lower().lstrip(".")

    if extension == "exr":
        _require_cv2("OpenEXR")
        linear = _to_cv2_channel_order(srgb_to_linear(image))
        params = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]
        _cv2_write(path, linear, params, "OpenEXR half")
        return

    if extension == "avif":
        _require_cv2("AVIF")
        eight = _to_cv2_channel_order(_quantize(image, 8))
        _cv2_write(path, eight, [cv2.IMWRITE_AVIF_QUALITY, int(quality)], "AVIF")
        return

    if int(bit_depth) == 16:
        if extension not in ("png", "tiff"):
            raise RuntimeError(
                f"16-bit was asked for and `{extension}` cannot carry it; "
                "only PNG and TIFF can."
            )
        _require_cv2("16-bit")
        deep = _to_cv2_channel_order(_quantize(image, 16))
        _cv2_write(path, deep, [], f"16-bit {extension.upper()}")
        return

    _write_eight_bit(image, path, extension, quality, lossless_webp, fast)


def _quantize(image, bits):
    peak = (1 << bits) - 1
    dtype = np.uint16 if bits == 16 else np.uint8
    return np.clip(image * peak + 0.5, 0, peak).astype(dtype)


def _require_cv2(what):
    if not CV2_AVAILABLE:
        raise RuntimeError(
            f"{what} needs OpenCV, and this build has none. "
            "It is in requirements.txt: the install is incomplete."
        )


def _cv2_write(path, array, params, what):
    if not cv2.imwrite(path, array, params):
        raise RuntimeError(
            f"OpenCV refused to write {what} to {path}. "
            "The build is missing the codec; nothing was written."
        )


def _write_eight_bit(image, path, extension, quality, lossless_webp, fast):
    array = _quantize(image, 8)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    img = Image.fromarray(array)
    if extension in ("jpg", "jpeg"):
        img.save(path, quality=quality, optimize=not fast)
    elif extension == "webp":
        img.save(path, quality=quality, lossless=lossless_webp)
    elif extension == "png":
        # 0=none, 1=fast, 6=default, 9=max. Level 1 is ~32x faster than 9 and
        # is what a batch gets, because a batch is waiting on this loop.
        img.save(path, compress_level=1 if fast else 6)
    elif extension == "bmp":
        img.save(path)
    elif extension == "tiff":
        img.save(path, quality=quality, optimize=not fast)
    else:
        img.save(path, optimize=not fast)


# ---------------------------------------------------------------- video -----

#: What each rung is, as ffmpeg arguments and a container.
#:
#: `container` is the fact the renderer needs: ProRes and DNxHR are QuickTime,
#: FFV1 is Matroska, and only H.264/H.265 stay in an MP4. The node writes
#: `<prefix>.<container>` and the type the renderer declared has to agree — a
#: file at an address nobody asks for is the same as no file.
VIDEO_PRESETS = {
    # The four that existed before rungs did. Their names and arguments are
    # frozen: they are inside addresses that are already computed.
    "web_compatible": {
        "codec": "libx264",
        "profile": "baseline",
        "level": "3.0",
        "pix_fmt": "yuv420p",
        "crf": "23",
        "container": "mp4",
    },
    "high_quality": {
        "codec": "libx264",
        "profile": "high",
        "level": "4.0",
        "pix_fmt": "yuv420p",
        "crf": "18",
        "container": "mp4",
    },
    "fast_encode": {
        "codec": "libx264",
        "profile": "baseline",
        "level": "3.0",
        "pix_fmt": "yuv420p",
        "crf": "28",
        "preset": "ultrafast",
        "container": "mp4",
    },
    "high quality": {
        "codec": "libx264",
        "profile": "high",
        "level": "4.0",
        "pix_fmt": "yuv420p",
        "crf": "15",
        "preset": "slow",
        "keyint": "24",
        "min_keyint": "12",
        "sc_threshold": "0",
        "g": "24",
        "container": "mp4",
    },
    # The rungs. `extra` is passed through verbatim after the codec.
    "prores4444": {
        "codec": "prores_ks",
        "pix_fmt": "yuva444p10le",
        "container": "mov",
        "extra": ["-profile:v", "4444", "-vendor", "apl0", "-bits_per_mb", "8000"],
    },
    "prores422hq": {
        "codec": "prores_ks",
        "pix_fmt": "yuv422p10le",
        "container": "mov",
        "extra": ["-profile:v", "3", "-vendor", "apl0"],
    },
    "dnxhr_hqx": {
        "codec": "dnxhd",
        "pix_fmt": "yuv422p10le",
        "container": "mov",
        "extra": ["-profile:v", "dnxhr_hqx"],
    },
    "ffv1": {
        "codec": "ffv1",
        "pix_fmt": "yuv444p",
        "container": "mkv",
        "extra": ["-level", "3", "-g", "1", "-slices", "4", "-slicecrc", "1"],
    },
    "h265_crf24": {
        "codec": "libx265",
        "pix_fmt": "yuv420p10le",
        "container": "mp4",
        "extra": ["-crf", "24", "-preset", "medium", "-tag:v", "hvc1"],
    },
}


def video_container(quality):
    """The file extension a video rung writes. Unknown rungs are MP4."""
    preset = VIDEO_PRESETS.get(quality)
    return preset["container"] if preset else "mp4"


def video_codec_args(quality):
    """
    The ffmpeg arguments for a rung's video stream, codec first.

    Separate from the container so that the caller can decide, per attempt,
    whether to try a hardware encoder first — which it may only do for the
    H.264 rungs, where the result is still H.264. Substituting a hardware
    encoder for ProRes would write something else under the rung's name.
    """
    preset = VIDEO_PRESETS.get(quality)
    if preset is None:
        raise RuntimeError(
            f"No such video rung: {quality!r}. "
            f"Known: {', '.join(sorted(VIDEO_PRESETS))}"
        )
    args = ["-c:v", preset["codec"]]
    if "extra" in preset:
        args += list(preset["extra"])
    else:
        args += [
            "-profile:v",
            preset["profile"],
            "-level",
            preset["level"],
            "-crf",
            preset["crf"],
        ]
    return args


def video_allows_hardware_encoder(quality):
    """
    True only where a hardware H.264 encoder writes the same codec the rung
    names. Never for a master rung: ProRes encoded as H.264 is not ProRes.
    """
    preset = VIDEO_PRESETS.get(quality)
    return bool(preset) and preset["codec"] == "libx264"


# ---------------------------------------------------------------- audio -----

#: `codec` and `container` as ffmpeg names; `sample_fmt` where the codec does
#: not decide it. The renderer sends `format`, and MP3 stays the default so an
#: untouched card writes what it has always written.
AUDIO_FORMATS = {
    "mp3": {"codec": "libmp3lame", "container": "mp3", "extension": "mp3"},
    # No resampling: a master is written at the rate it arrived at. The rung
    # is named for its depth alone for the same reason.
    "wav": {
        "codec": "pcm_s24le",
        "container": "wav",
        "extension": "wav",
        "sample_fmt": "s32",
    },
    "flac": {
        "codec": "flac",
        "container": "flac",
        "extension": "flac",
        "sample_fmt": "s32",
    },
    "aac": {"codec": "aac", "container": "ipod", "extension": "m4a"},
}


def audio_format(name):
    """The encoder for an audio rung. Unknown names raise rather than guess."""
    spec = AUDIO_FORMATS.get(name)
    if spec is None:
        raise RuntimeError(
            f"No such audio rung: {name!r}. Known: {', '.join(sorted(AUDIO_FORMATS))}"
        )
    return spec
