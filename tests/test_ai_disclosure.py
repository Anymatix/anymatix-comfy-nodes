"""
Every result the save nodes write says, in a machine-readable way, that AI
made it -- AI Act art. 50(2). See `anymatix_ai_disclosure.py`.

The claims these tests defend:

  1. every markable format carries the IPTC DigitalSourceType URI, written
     through the SAME code path a run takes (`write_image`, the audio node's
     `save_audio`, the video node's `save_video`);
  2. nothing of the job leaks into the file -- not the path, not the file
     name, and so not a prompt that a caller put in either;
  3. the label never costs the picture: the pixels, samples and frames decode
     exactly as they do without it, and a label that fails leaves the file
     whole and the save successful.

Where `exiftool` is on PATH it is used as the independent reader: it parses
each container's metadata structure itself, which is the proof that the label
sits where other software looks for it, not merely somewhere in the bytes.
"""

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


disclosure = _load("anymatix_ai_disclosure")
formats = _load("anymatix_output_formats")

URI = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
SENTENCE = "Created by Anymatix using AI generators"

#: What a caller might have put in a path or a file name. It must never reach
#: the file's bytes.
SECRET = "SECRET-a-red-car-on-the-moon-prompt"

cv2 = pytest.importorskip("cv2")
EXIFTOOL = shutil.which("exiftool")


def a_gradient(height=32, width=48, channels=3):
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    planes = [(y * x), (y * (1 - x)), ((1 - y) * x), (0.25 + 0.5 * y * x)]
    return np.stack(planes[:channels], axis=-1)


def exiftool_tags(path):
    """Every tag exiftool finds, as {"Group:Name": value}. Skips without it."""
    if EXIFTOOL is None:
        pytest.skip("exiftool is not on PATH: no independent reader")
    done = subprocess.run(
        [EXIFTOOL, "-j", "-a", "-G1", "-struct", path], capture_output=True, check=True
    )
    return json.loads(done.stdout)[0]


def assert_labelled_per_exiftool(path, xmp=True):
    tags = exiftool_tags(path)
    source_types = {
        key: value
        for key, value in tags.items()
        if key.split(":")[-1].replace("_", "").lower() == "digitalsourcetype"
    }
    assert URI in source_types.values(), f"no DigitalSourceType in {sorted(tags)}"
    if xmp:
        assert tags.get("XMP-iptcExt:DigitalSourceType") == URI
        assert tags.get("XMP-xmp:CreatorTool") == "Anymatix"
        assert tags.get("XMP-dc:Description") == SENTENCE
    # What exiftool read FROM the file, not what it says about the file
    # system entry (`SourceFile`, `System:FileName`, `System:Directory`).
    inside = {k: v for k, v in tags.items() if k != "SourceFile" and not k.startswith("System:")}
    assert SECRET not in json.dumps(inside)


def assert_nothing_leaked(path):
    with open(path, "rb") as f:
        data = f.read()
    assert SECRET.encode() not in data
    assert os.path.dirname(path).encode() not in data
    return data


def no_stray_temp_files(directory):
    return [name for name in os.listdir(directory) if name.startswith(".")]


# ----------------------------------------------------------------- images ----

IMAGE_CASES = [
    # (extension, write_image kwargs, channels)
    ("png", {}, 3),
    ("png", {"bit_depth": 16}, 3),
    ("png", {}, 4),
    ("jpg", {}, 3),
    ("jpeg", {"quality": 95}, 3),
    ("gif", {}, 3),
    ("tiff", {}, 3),
    ("tiff", {"bit_depth": 16}, 3),
    ("webp", {"quality": 90}, 3),
    ("webp", {"quality": 90}, 4),
    ("webp", {"lossless_webp": True}, 3),
    ("webp", {"lossless_webp": True}, 4),
    ("exr", {}, 3),
    ("avif", {"quality": 90}, 3),
]


def _decode(path):
    array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if array is None:
        from PIL import Image

        with Image.open(path) as image:
            array = np.asarray(image.convert("RGBA"))
    return array


def _case_id(case):
    extension, kwargs, channels = case
    extra = "-".join(f"{k}{v}" for k, v in sorted(kwargs.items()))
    return f"{extension}-{channels}ch" + (f"-{extra}" if extra else "")


@pytest.mark.parametrize("case", IMAGE_CASES, ids=_case_id)
def test_every_image_format_is_labelled_through_write_image(tmp_path, case):
    extension, kwargs, channels = case
    directory = tmp_path / SECRET
    directory.mkdir()
    path = str(directory / f"{SECRET}_0001.{extension}")
    formats.write_image(a_gradient(channels=channels), path, extension=extension, **kwargs)

    data = assert_nothing_leaked(path)
    if extension == "exr":
        assert URI.encode() in data and SENTENCE.encode() in data
    else:
        assert disclosure.XMP_PACKET in data
    assert no_stray_temp_files(str(directory)) == []
    assert_labelled_per_exiftool(path, xmp=extension != "exr")


@pytest.mark.parametrize("case", IMAGE_CASES, ids=_case_id)
def test_the_label_does_not_change_a_single_pixel(tmp_path, case):
    """The same encoder output, with and without the label, decodes the same."""
    extension, kwargs, channels = case
    bare = str(tmp_path / f"bare.{extension}")
    formats._write_image_to(
        bare,
        a_gradient(channels=channels),
        extension,
        kwargs.get("quality", 100),
        kwargs.get("lossless_webp", False),
        kwargs.get("bit_depth", 8),
        False,
    )
    with open(bare, "rb") as f:
        original = f.read()
    marked = disclosure.mark_bytes(original, extension)
    assert marked != original, "the format is markable and was not marked"
    labelled = str(tmp_path / f"labelled.{extension}")
    with open(labelled, "wb") as f:
        f.write(marked)
    before, after = _decode(bare), _decode(labelled)
    assert before is not None and after is not None
    assert before.dtype == after.dtype and before.shape == after.shape
    assert np.array_equal(before, after)


def test_pillow_still_opens_and_reads_the_xmp(tmp_path):
    """A second independent reader, for the formats Pillow parses XMP from."""
    from PIL import Image

    for extension in ("png", "jpeg", "webp", "tiff"):
        path = str(tmp_path / f"a.{extension}")
        formats.write_image(a_gradient(), path, extension=extension)
        with Image.open(path) as image:
            image.load()
            info = image.info
        raw = info.get("xmp") or info.get("XML:com.adobe.xmp") or b""
        if isinstance(raw, str):
            raw = raw.encode()
        if extension == "tiff":
            raw = raw or bytes(image.tag_v2.get(700, b""))
        assert URI.encode() in raw, extension


def test_bmp_is_written_and_honestly_left_unlabelled(tmp_path):
    """BMP has no metadata field: the file is written, and nothing is faked."""
    path = str(tmp_path / "a.bmp")
    formats.write_image(a_gradient(), path, extension="bmp")
    with open(path, "rb") as f:
        data = f.read()
    assert data[:2] == b"BM"
    assert URI.encode() not in data
    assert cv2.imread(path) is not None


# ------------------------------------------------------ failure is harmless ----


def test_a_label_that_fails_never_fails_the_image(tmp_path, monkeypatch):
    def broken(_data):
        raise ValueError("simulated")

    monkeypatch.setitem(disclosure._MARKERS, "png", broken)
    monkeypatch.setattr(formats, "mark_file", disclosure.mark_file)
    path = str(tmp_path / "a.png")
    formats.write_image(a_gradient(), path, extension="png")
    assert cv2.imread(path) is not None
    with open(path, "rb") as f:
        # The XMP packet, not the URI: the C2PA manifest signed over the file
        # (`anymatix_c2pa.py`) carries the URI too, and is not what failed.
        assert disclosure.XMP_PACKET not in f.read()
    assert no_stray_temp_files(str(tmp_path)) == []


def test_mark_file_leaves_the_file_exactly_as_it_was_on_failure(tmp_path, monkeypatch):
    path = str(tmp_path / "a.png")
    formats._write_image_to(path, a_gradient(), "png", 100, False, 8, False)
    with open(path, "rb") as f:
        original = f.read()

    def broken(_data):
        raise ValueError("simulated")

    monkeypatch.setitem(disclosure._MARKERS, "png", broken)
    assert disclosure.mark_file(path, "png") is False
    with open(path, "rb") as f:
        assert f.read() == original
    assert no_stray_temp_files(str(tmp_path)) == []


def test_a_video_append_that_fails_is_truncated_back(tmp_path, monkeypatch):
    path = str(tmp_path / "a.mp4")
    original = disclosure._box(b"ftyp", b"isom\x00\x00\x02\x00isom") + disclosure._box(
        b"mdat", b"\x00" * 64
    )
    with open(path, "wb") as f:
        f.write(original)

    def half_a_box():
        raise OSError("disk full, simulated")

    monkeypatch.setattr(disclosure, "_xmp_uuid_box", half_a_box)
    assert disclosure.mark_file(path, "mp4") is False
    with open(path, "rb") as f:
        assert f.read() == original


@pytest.mark.parametrize(
    "extension,data",
    [
        ("png", b"not a png at all"),
        ("jpeg", b"\x00\x01\x02"),
        ("webp", b"RIFF\x04\x00\x00\x00WEBP"),
        ("webp", b"RIFF\x10\x00\x00\x00WEBPVP8 \x04\x00\x00\x00abcd"),
        ("gif", b"GIF"),
        ("tiff", b"II+\x00"),
        ("exr", b"\x76\x2f\x31\x01\x00\x02\x00\x00"),
        ("avif", b"\x00\x00\x00\x10ftypavif\x00\x00\x00\x00"),
        ("mp4", b"\x00\x00\x00\x08moov"),
        ("mp3", b"ID3\x04\x00\x40\x00\x00\x00\x00"),
        ("wav", b"RIFF\xff\xff\xff\xffWAVE"),
        ("bmp", b"BM"),
        ("json", b"{}"),
    ],
)
def test_what_is_not_recognised_comes_back_untouched(extension, data):
    assert disclosure.mark_bytes(data, extension) == data


def test_the_label_is_constant_and_carries_nothing_of_the_job():
    """By construction: the packet is a module constant with no parameter."""
    packet = disclosure.XMP_PACKET
    assert URI.encode() in packet and SENTENCE.encode() in packet
    assert b"CreatorTool=\"Anymatix\"" in packet
    assert b"\x00" not in packet  # the GIF embedding depends on it
    packet.decode("utf-8")
    assert disclosure.CONTAINER_TAGS == (
        ("comment", SENTENCE),
        ("DIGITAL_SOURCE_TYPE", URI),
    )


# --------------------------------------------------- the audio and video nodes ----


@pytest.fixture(scope="module")
def nodes():
    """
    The real save-node modules, imported as a package with ComfyUI's two
    modules stubbed: `folder_paths` (where the output goes) and `comfy.utils`
    (a progress bar). Nothing else of the node is replaced.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("av")
    package = "anymatix_nodes_under_test"
    saved = {name: sys.modules.get(name) for name in ("folder_paths", "comfy", "comfy.utils")}

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.output = None
    folder_paths.get_output_directory = lambda: folder_paths.output
    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")

    class ProgressBar(object):
        def __init__(self, total):
            self.total = total

        def update(self, _n):
            pass

    comfy_utils.ProgressBar = ProgressBar
    comfy.utils = comfy_utils
    sys.modules.update({"folder_paths": folder_paths, "comfy": comfy, "comfy.utils": comfy_utils})
    pkg = types.ModuleType(package)
    pkg.__path__ = [_ROOT]
    sys.modules[package] = pkg
    try:
        audio = importlib.import_module(f"{package}.anymatix_save_audio")
        video = importlib.import_module(f"{package}.anymatix_save_animated_mp4")
        yield types.SimpleNamespace(
            torch=torch, folder_paths=folder_paths, audio=audio, video=video
        )
    finally:
        for name in list(sys.modules):
            if name == package or name.startswith(package + "."):
                del sys.modules[name]
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _a_tone(torch, seconds=0.25, rate=44100):
    t = torch.arange(int(seconds * rate), dtype=torch.float32) / rate
    left = 0.3 * torch.sin(2 * np.pi * 440 * t)
    right = 0.3 * torch.sin(2 * np.pi * 660 * t)
    return {"waveform": torch.stack([left, right])[None], "sample_rate": rate}


@pytest.mark.parametrize(
    "rung,extension,xmp",
    [("mp3", "mp3", True), ("wav", "wav", True), ("aac", "m4a", True), ("flac", "flac", False)],
)
def test_every_audio_rung_is_labelled_through_the_node(nodes, tmp_path, rung, extension, xmp):
    import av

    nodes.folder_paths.output = str(tmp_path)
    tone = _a_tone(nodes.torch)
    saved = nodes.audio.AnymatixSaveAudio().save_audio(
        tone, output_path=SECRET, filename_prefix=SECRET, format=rung
    )
    assert saved["ui"]["audio"], "the save reported no file"
    path = str(tmp_path / SECRET / f"{SECRET}.{extension}")

    data = assert_nothing_leaked(path)
    if xmp:
        assert disclosure.XMP_PACKET in data
    with av.open(path) as container:
        tags = {key.upper(): value for key, value in container.metadata.items()}
        samples = sum(frame.samples for frame in container.decode(audio=0))
    if rung in ("flac", "mp3"):
        assert tags.get("DIGITAL_SOURCE_TYPE") == URI
    assert tags.get("COMMENT") == SENTENCE
    assert samples >= tone["waveform"].shape[-1]
    assert no_stray_temp_files(str(tmp_path / SECRET)) == []
    assert_labelled_per_exiftool(path, xmp=xmp)


def _a_clip(torch, frames=4, size=256, with_audio=False):
    y = torch.linspace(0, 1, size)[:, None]
    x = torch.linspace(0, 1, size)[None, :]
    images = torch.stack(
        [torch.stack([y * x, y * (1 - x), (1 - y) * x + i / 10], dim=-1) for i in range(frames)]
    ).clamp(0, 1)
    audio = _a_tone(torch, seconds=frames / 24) if with_audio else None
    components = types.SimpleNamespace(images=images, frame_rate=24, audio=audio)
    return types.SimpleNamespace(get_components=lambda: components)


@pytest.mark.parametrize(
    "rung,with_audio",
    [
        ("high_quality", False),
        ("high_quality", True),
        ("h265_crf24", False),
        ("prores4444", False),
        ("prores422hq", True),
        ("dnxhr_hqx", False),
        ("ffv1", False),
        ("ffv1", True),
    ],
)
def test_every_video_rung_is_labelled_through_the_node(nodes, tmp_path, rung, with_audio):
    import av

    if not nodes.video.FFMPEG_AVAILABLE:
        pytest.skip("no ffmpeg for the video node")
    nodes.folder_paths.output = str(tmp_path)
    frames = 4
    saved = nodes.video.AnymatixSaveAnimatedMP4().save_video(
        _a_clip(nodes.torch, frames=frames, with_audio=with_audio), SECRET, SECRET, rung
    )
    assert saved["ui"]["images"], "the save reported no file"
    container_ext = formats.video_container(rung)
    path = str(tmp_path / SECRET / f"{SECRET}.{container_ext}")

    data = assert_nothing_leaked(path)
    xmp = container_ext in ("mp4", "mov")
    if xmp:
        assert data.endswith(disclosure._xmp_uuid_box())
    with av.open(path) as container:
        tags = {key.upper(): value for key, value in container.metadata.items()}
        decoded = sum(1 for _ in container.decode(video=0))
    assert tags.get("COMMENT") == SENTENCE
    if container_ext == "mkv":
        assert tags.get("DIGITAL_SOURCE_TYPE") == URI
    assert decoded == frames
    assert no_stray_temp_files(str(tmp_path / SECRET)) == []
    assert_labelled_per_exiftool(path, xmp=xmp)
