"""
What a format rung means in bytes.

The claim these tests defend is narrow and it is the whole point of the
feature: **the file is what the rung says it is**. A 16-bit rung that writes
eight bits is not a smaller mistake than a crash — the address says one thing,
the file says another, and nothing downstream can tell. So every image test
reads the file back and looks at the dtype, and the video and audio tests pin
the codec and the container that the renderer independently derives from the
declared type.

The module under test imports neither `comfy` nor `folder_paths`, which is why
it exists: the save nodes cannot be imported outside a running ComfyUI, and
anything kept inside them cannot be tested at all.
"""

import importlib.util
import os

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "anymatix_output_formats", os.path.join(_HERE, "..", "anymatix_output_formats.py")
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

write_image = _MODULE.write_image
srgb_to_linear = _MODULE.srgb_to_linear
VIDEO_PRESETS = _MODULE.VIDEO_PRESETS
video_container = _MODULE.video_container
video_codec_args = _MODULE.video_codec_args
video_allows_hardware_encoder = _MODULE.video_allows_hardware_encoder
audio_format = _MODULE.audio_format
AUDIO_FORMATS = _MODULE.AUDIO_FORMATS

cv2 = pytest.importorskip("cv2")


def a_gradient(height=32, width=32, channels=3):
    """Something with real values in it: a flat colour compresses to nothing."""
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    planes = [(y * x), (y * (1 - x)), ((1 - y) * x)]
    return np.stack(planes[:channels], axis=-1)


# --------------------------------------------------------------- images ----


def test_png_8_bit_is_the_default_and_reads_back_as_bytes(tmp_path):
    path = str(tmp_path / "a.png")
    write_image(a_gradient(), path, extension="png")
    back = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert back.dtype == np.uint8
    assert back.shape == (32, 32, 3)


@pytest.mark.parametrize("extension", ["png", "tiff"])
def test_sixteen_bit_writes_sixteen_bits(tmp_path, extension):
    """The claim in the rung's name, checked against the file."""
    path = str(tmp_path / f"deep.{extension}")
    write_image(a_gradient(), path, extension=extension, bit_depth=16)
    back = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert back.dtype == np.uint16, "the rung says 16-bit and the file must be"
    assert back.shape == (32, 32, 3)


def test_sixteen_bit_keeps_distinctions_eight_bit_loses(tmp_path):
    """
    Depth is not decoration. Two values a 256-step ramp cannot tell apart come
    back different from the 16-bit file and identical from the 8-bit one.
    """
    image = np.zeros((1, 2, 3), dtype=np.float32)
    image[0, 0, :] = 0.5000
    image[0, 1, :] = 0.5010  # a quarter of an 8-bit step apart

    deep = str(tmp_path / "deep.png")
    shallow = str(tmp_path / "shallow.png")
    write_image(image, deep, extension="png", bit_depth=16)
    write_image(image, shallow, extension="png", bit_depth=8)

    d = cv2.imread(deep, cv2.IMREAD_UNCHANGED)
    s = cv2.imread(shallow, cv2.IMREAD_UNCHANGED)
    assert d[0, 0, 0] != d[0, 1, 0]
    assert s[0, 0, 0] == s[0, 1, 0]


def test_sixteen_bit_is_refused_where_it_cannot_be_carried(tmp_path):
    with pytest.raises(RuntimeError, match="cannot carry it"):
        write_image(a_gradient(), str(tmp_path / "x.jpg"), extension="jpg", bit_depth=16)


def test_exr_is_float_and_linear(tmp_path):
    """
    A `.exr` labelled "half float, linear" that holds display-space values is
    mislabelled: a compositor treats it as linear and grades it wrong.
    """
    image = np.full((4, 4, 3), 0.5, dtype=np.float32)
    path = str(tmp_path / "m.exr")
    write_image(image, path, extension="exr")
    back = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert back.dtype == np.float32
    # sRGB 0.5 is 0.2140 in linear light; 0.5 would mean nothing was converted.
    assert back[0, 0, 0] == pytest.approx(0.2140, abs=0.002)


def test_exr_is_written_as_half_not_full(tmp_path):
    """Half is two bytes a channel; full float would be about twice the file."""
    path = str(tmp_path / "half.exr")
    write_image(a_gradient(64, 64), path, extension="exr")
    ceiling = 64 * 64 * 3 * 2 * 1.25  # + header and EXR's own overhead
    assert os.path.getsize(path) < ceiling


def test_channel_order_survives_the_round_trip(tmp_path):
    """
    OpenCV writes BGR and a ComfyUI IMAGE is RGB. Get this wrong and every
    16-bit master comes out with red and blue swapped — which looks like a
    working feature until somebody opens one.
    """
    image = np.zeros((2, 2, 3), dtype=np.float32)
    image[..., 0] = 1.0  # pure red
    path = str(tmp_path / "red.png")
    write_image(image, path, extension="png", bit_depth=16)
    blue, green, red = cv2.imread(path, cv2.IMREAD_UNCHANGED)[0, 0]
    assert (red, green, blue) == (65535, 0, 0)


def test_alpha_survives_where_the_format_has_it(tmp_path):
    image = np.zeros((2, 2, 4), dtype=np.float32)
    image[..., 0] = 1.0
    image[..., 3] = 0.5
    path = str(tmp_path / "rgba.png")
    write_image(image, path, extension="png", bit_depth=16)
    back = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert back.shape[2] == 4
    assert back[0, 0, 3] == pytest.approx(32768, abs=200)


def test_avif_writes_and_is_smaller_than_the_png(tmp_path):
    png = str(tmp_path / "big.png")
    avif = str(tmp_path / "small.avif")
    image = a_gradient(128, 128)
    write_image(image, png, extension="png")
    write_image(image, avif, extension="avif", quality=90)
    assert os.path.getsize(avif) < os.path.getsize(png)


def test_srgb_to_linear_fixes_its_ends():
    assert srgb_to_linear(np.float32(0.0)) == pytest.approx(0.0)
    assert srgb_to_linear(np.float32(1.0)) == pytest.approx(1.0)


# ---------------------------------------------------------------- video ----


@pytest.mark.parametrize(
    "rung,container",
    [
        ("prores4444", "mov"),
        ("prores422hq", "mov"),
        ("dnxhr_hqx", "mov"),
        ("ffv1", "mkv"),
        ("h265_crf24", "mp4"),
        ("high_quality", "mp4"),
        ("high quality", "mp4"),
    ],
)
def test_the_container_is_the_rungs(rung, container):
    """
    The renderer derives the same extension from the type it declared. If the
    two ever disagree the file exists at an address nobody asks for, which is
    indistinguishable from no file.
    """
    assert video_container(rung) == container


def test_prores_asks_for_prores():
    args = video_codec_args("prores4444")
    assert args[:2] == ["-c:v", "prores_ks"]
    assert "4444" in args


def test_ffv1_makes_every_frame_a_keyframe():
    """Archival footage is scrubbed and cut; a 24-frame GOP is not that."""
    args = video_codec_args("ffv1")
    assert args[args.index("-g") + 1] == "1"


def test_a_master_rung_never_gets_the_hardware_encoder():
    """
    h264_videotoolbox is tried first because it writes H.264. Offering it for
    ProRes would put H.264 bytes at a ProRes address, silently.
    """
    assert video_allows_hardware_encoder("high_quality") is True
    assert video_allows_hardware_encoder("prores4444") is False
    assert video_allows_hardware_encoder("ffv1") is False


def test_an_unknown_rung_raises_rather_than_picking_one():
    with pytest.raises(RuntimeError, match="No such video rung"):
        video_codec_args("mezzanine")


def test_the_four_presets_that_predate_rungs_are_unchanged():
    """
    These names are inside addresses that are already computed. Changing what
    one of them encodes changes the bytes under an address that says they did
    not change.
    """
    assert VIDEO_PRESETS["web_compatible"]["crf"] == "23"
    assert VIDEO_PRESETS["high_quality"]["crf"] == "18"
    assert VIDEO_PRESETS["fast_encode"]["crf"] == "28"
    assert VIDEO_PRESETS["high quality"]["crf"] == "15"
    for name in ("web_compatible", "high_quality", "fast_encode", "high quality"):
        assert VIDEO_PRESETS[name]["codec"] == "libx264"
        assert VIDEO_PRESETS[name]["container"] == "mp4"


# ---------------------------------------------------------------- audio ----


@pytest.mark.parametrize(
    "rung,codec,extension",
    [
        ("mp3", "libmp3lame", "mp3"),
        ("wav", "pcm_s24le", "wav"),
        ("flac", "flac", "flac"),
        ("aac", "aac", "m4a"),
    ],
)
def test_each_audio_rung_names_its_codec_and_its_file(rung, codec, extension):
    spec = audio_format(rung)
    assert spec["codec"] == codec
    assert spec["extension"] == extension


def test_mp3_is_still_the_default_shape():
    """An untouched card keeps the file name it has always written."""
    assert AUDIO_FORMATS["mp3"]["extension"] == "mp3"


def test_an_unknown_audio_rung_raises():
    with pytest.raises(RuntimeError, match="No such audio rung"):
        audio_format("dat")


def test_wav_does_not_resample():
    """
    A master is written at the rate it arrived at. Silently resampling one is
    the kind of helpfulness that loses information nobody agreed to lose.
    """
    assert "sample_rate" not in AUDIO_FORMATS["wav"]


# ------------------------------------------------- the encoders themselves --
#
# The tests above pin what we ASK ffmpeg for. These run it. A build without
# prores_ks would pass every one of them and then fail on the machine, at the
# end of a generation somebody waited minutes for.


def _ffmpeg():
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    return imageio_ffmpeg.get_ffmpeg_exe()


def _encode(tmp_path, rung, frames=4, size=(256, 256)):
    import subprocess

    exe = _ffmpeg()
    width, height = size
    os.makedirs(str(tmp_path), exist_ok=True)
    out = str(tmp_path / f"clip.{video_container(rung)}")
    command = [
        exe, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", "24", "-i", "-",
        "-pix_fmt", VIDEO_PRESETS[rung]["pix_fmt"],
    ] + video_codec_args(rung) + [out]
    payload = (np.random.rand(frames, height, width, 3) * 255).astype(np.uint8)
    done = subprocess.run(command, input=payload.tobytes(), capture_output=True)
    assert done.returncode == 0, done.stderr.decode(errors="replace")[-1500:]
    return out


def _stream_codec(path):
    import json as _json
    import subprocess

    exe = _ffmpeg().replace("ffmpeg-", "ffprobe-")
    if not os.path.exists(exe):
        # imageio-ffmpeg ships no ffprobe; ffmpeg itself names the codec.
        done = subprocess.run([_ffmpeg(), "-i", path], capture_output=True)
        return done.stderr.decode(errors="replace")
    done = subprocess.run(
        [exe, "-v", "quiet", "-print_format", "json", "-show_streams", path],
        capture_output=True,
    )
    return _json.dumps(_json.loads(done.stdout))


@pytest.mark.parametrize(
    "rung,expect",
    [
        ("prores4444", "prores"),
        ("prores422hq", "prores"),
        ("dnxhr_hqx", "dnxhd"),
        ("ffv1", "ffv1"),
        ("h265_crf24", "hevc"),
    ],
)
def test_the_master_rungs_actually_encode(tmp_path, rung, expect):
    path = _encode(tmp_path, rung)
    assert os.path.getsize(path) > 0
    assert expect in _stream_codec(path).lower()


def test_a_lossless_rung_is_very_much_larger_than_a_delivery_one(tmp_path):
    """
    The size is the only externally visible sign that the rung took effect,
    and it is the one a user checks. Noise is the fair test: it defeats
    inter-frame prediction, so the difference here is the codec's.
    """
    archival = os.path.getsize(_encode(tmp_path / "a", "ffv1"))
    delivery = os.path.getsize(_encode(tmp_path / "b", "h265_crf24"))
    assert archival > delivery * 3
