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

try:
    from .anymatix_atomic_write import cleanup_temp, publish, temp_path_for
    from .anymatix_ai_disclosure import mark_file
    from .anymatix_c2pa import sign_file
except ImportError:
    # Loaded standalone (e.g. by tests via spec_from_file_location), with no
    # package context for a relative import to resolve against.
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from anymatix_atomic_write import cleanup_temp, publish, temp_path_for
    from anymatix_ai_disclosure import mark_file
    from anymatix_c2pa import sign_file


# ---------------------------------------------------------------- images ----

#: The extensions the image save node accepts. `png` and `tiff` also take a
#: bit depth; the rest are 8-bit by definition.
IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "gif", "tiff", "webp", "bmp", "exr", "avif"]

#: Extensions that need OpenCV whatever the depth.
_CV2_ONLY = ("exr", "avif")

#: What the pixel values of the incoming IMAGE mean. `sRGB` is every card
#: before HDR and stays the default, so an untouched card writes the bytes it
#: always wrote. `HDR` is BT.2100 HLG on Rec.2020 primaries and `HDR PQ` is
#: BT.2100 PQ on Rec.2020: the encodings ComfyUI's `ImageSDRToHDR` and
#: `ImageColorSpace` produce, still 0..1 but no longer display sRGB.
COLOR_SPACES = ["sRGB", "HDR", "HDR PQ"]

#: The CICP code points an HDR AVIF declares (ITU-T H.273): primaries,
#: transfer, matrix. Without them a viewer decodes the file as sRGB and the
#: highlights the card made are drawn as a washed-out SDR picture.
_AVIF_HDR_CICP = {
    "HDR": {"primaries": 9, "transfer": 18, "matrix": 9},  # BT.2020, ARIB STD-B67 (HLG), BT.2020 NCL
    "HDR PQ": {"primaries": 9, "transfer": 16, "matrix": 9},  # BT.2020, SMPTE ST 2084 (PQ), BT.2020 NCL
}


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
    color_space="sRGB",
):
    """
    Write one frame, already float 0..1, in the format the rung asked for.

    `color_space` says what those values mean (see `COLOR_SPACES`). Only AVIF
    carries an HDR space today; any other format asked to write HDR values
    raises, because writing them as if they were sRGB would mislabel the file.

    `image` is HxW, HxWx1, HxWx3 or HxWx4. `fast` trades compression for time
    and is set when a batch is large enough that the difference is felt; it
    never changes which format is written.

    Raises `RuntimeError` when the build cannot write the format. See the
    module docstring for why that is not a fallback.

    `path` never sees a partial file: the frame is written to a hidden temp
    name in the same directory and `os.replace()`d into place only once it is
    complete, so a reader polling `path` by URL never observes a half-written
    image. See `anymatix_atomic_write.py`.

    The staged file is labelled as AI-generated before it is published
    (AI Act art. 50(2); `anymatix_ai_disclosure.py`), then signed with a
    self-signed C2PA manifest (`anymatix_c2pa.py`). Both
    are best-effort: they never fail the write and never alter the pixels.
    """
    temp_path = temp_path_for(path)
    try:
        _write_image_to(
            temp_path, image, extension, quality, lossless_webp, bit_depth, fast, color_space
        )
        mark_file(temp_path, extension)
        sign_file(temp_path, extension)
        publish(temp_path, path)
    except BaseException:
        cleanup_temp(temp_path)
        raise


def _write_image_to(
    path, image, extension, quality, lossless_webp, bit_depth, fast, color_space="sRGB"
):
    image = np.asarray(image, dtype=np.float32)
    extension = extension.lower().lstrip(".")

    if color_space not in COLOR_SPACES:
        raise RuntimeError(
            f"No such colour space: {color_space!r}. Known: {', '.join(COLOR_SPACES)}"
        )
    if color_space != "sRGB":
        if extension != "avif":
            raise RuntimeError(
                f"`{extension}` cannot be written from {color_space} values yet; "
                "only AVIF carries HDR. Writing them as sRGB would mislabel the file."
            )
        _write_hdr_avif(path, image, quality, color_space)
        return

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


def _hdr_avif_crf(quality):
    """
    The AV1 rate factor for an AVIF quality. The renderer states quality as
    1..100 for every lossy rung; AV1 counts the other way, 0..63. 90 maps to
    18, the rate ComfyUI's own AVIF save uses by default.
    """
    quality = max(1, min(100, int(quality)))
    return int(round(63 - 0.5 * quality))


def _write_hdr_avif(path, image, quality, color_space):
    """
    A 10-bit 4:2:0 AVIF, with the HDR colour space DECLARED in the file.

    OpenCV cannot write the CICP colour description, so this goes through
    PyAV (already a dependency, and the library ComfyUI writes video with).
    The values arrive already encoded in the HDR transfer (HLG or PQ): they
    are quantised to 16 bits, converted to Y'CbCr with the BT.2020 matrix and
    stored at 10 bits. No tone mapping and no transfer is applied here.

    No alpha: the AV1 encoders PyAV ships do not encode it, and dropping it
    without a word would be the silent downgrade this module refuses.
    """
    try:
        import av
    except ImportError as error:  # pragma: no cover - av is in requirements.txt
        raise RuntimeError(
            "HDR AVIF needs PyAV, and this build has none. "
            "It is in requirements.txt: the install is incomplete."
        ) from error

    if image.ndim == 3 and image.shape[2] == 4:
        raise RuntimeError("HDR AVIF has no alpha channel; the image has one.")
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    height, width = image.shape[:2]
    if height % 2 or width % 2:
        # 4:2:0 halves the chroma; an odd edge is padded by repeating it.
        image = np.pad(image, ((0, height % 2), (0, width % 2), (0, 0)), mode="edge")

    cicp = _AVIF_HDR_CICP[color_space]
    deep = _quantize(image, 16)
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(deep), format="rgb48le")
    frame = frame.reformat(format="yuv420p10le", dst_colorspace=cicp["matrix"])

    last_error = None
    for encoder in ("libsvtav1", "libaom-av1"):
        try:
            _encode_avif(av, path, frame, encoder, _hdr_avif_crf(quality), cicp)
            return
        except (av.error.FFmpegError, ValueError) as error:
            last_error = error
            if os.path.exists(path):
                os.remove(path)
    raise RuntimeError(
        f"No AV1 encoder in this build could write the HDR AVIF: {last_error}"
    )


def _encode_avif(av, path, frame, encoder, crf, cicp):
    with av.open(path, mode="w", format="avif") as container:
        stream = container.add_stream(encoder, rate=1)
        stream.width = frame.width
        stream.height = frame.height
        stream.pix_fmt = "yuv420p10le"
        stream.options = {"crf": str(crf)}
        context = stream.codec_context
        context.color_primaries = cicp["primaries"]
        context.color_trc = cicp["transfer"]
        context.colorspace = cicp["matrix"]
        context.color_range = 1  # limited ("MPEG") range, as the matrix expects
        frame.color_primaries = cicp["primaries"]
        frame.color_trc = cicp["transfer"]
        frame.colorspace = cicp["matrix"]
        frame.color_range = 1
        frame.pts = 0
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


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
