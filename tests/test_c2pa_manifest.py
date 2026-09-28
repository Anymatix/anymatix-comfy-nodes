"""
Every result the save nodes write carries a self-signed C2PA manifest on top
of its IPTC label. See `anymatix_c2pa.py`.

The claims these tests defend:

  1. every format c2pa-rs can embed in is signed through the SAME code path a
     run takes (`write_image`, the audio node, the video node), and a C2PA
     reader validates it with exactly one failure code:
     `signingCredential.untrusted` -- the signer is on nobody's trust list,
     which is the point, and nothing else is wrong;
  2. the manifest says `c2pa.created` + `trainedAlgorithmicMedia`, generator
     `Anymatix`, and carries no thumbnail and nothing of the job;
  3. the IPTC label is still where exiftool reads it after signing, and the
     pixels decode exactly as they did before;
  4. a timestamp is added when the TSA answers, and an unreachable or failing
     TSA only costs the timestamp, within a short timeout;
  5. a signing failure of any kind leaves a readable, labelled file and a
     successful save;
  6. the signing key is generated on the machine once, kept owner-only, and
     reused.

The C2PA reader is c2pa-python's `Reader` (c2pa-rs), the same library the
Content Credentials tools are built on. `tests/conftest.py` keeps every test
off the network and out of the real per-user data directory.
"""

import datetime
import http.server
import json
import os
import socket
import stat
import sys
import threading
import time

import numpy as np
import pytest

from test_ai_disclosure import (
    IMAGE_CASES,
    SECRET,
    URI,
    _a_clip,
    _a_tone,
    _case_id,
    assert_labelled_per_exiftool,
    assert_nothing_leaked,
    a_gradient,
    disclosure,
    formats,
    no_stray_temp_files,
    nodes,  # noqa: F401 -- the fixture, shared
)

c2pa = pytest.importorskip("c2pa")
cv2 = pytest.importorskip("cv2")

#: The module instance `write_image` calls -- loaded standalone by
#: `anymatix_output_formats`, so it is patched here and not re-imported.
signing = sys.modules[formats.sign_file.__module__]

SIGNED_IMAGE_CASES = [case for case in IMAGE_CASES if case[0] in signing.MIME_TYPES]


def c2pa_report(path):
    with c2pa.Reader(path) as reader:
        return json.loads(reader.json())


def assert_signed_untrusted(path, timestamped=False):
    report = c2pa_report(path)
    assert report["validation_state"] == "Valid"
    results = report["validation_results"]["activeManifest"]
    assert [s["code"] for s in results.get("failure", [])] == ["signingCredential.untrusted"]

    manifest = report["manifests"][report["active_manifest"]]
    assert manifest["claim_generator_info"][0]["name"] == "Anymatix"
    actions = [
        action
        for assertion in manifest["assertions"]
        if assertion["label"].startswith("c2pa.actions")
        for action in assertion["data"]["actions"]
    ]
    assert actions == [
        {
            "action": "c2pa.created",
            "softwareAgent": {"name": "Anymatix"},
            "digitalSourceType": URI,
        }
    ]
    assert "thumbnail" not in manifest
    assert not any("thumbnail" in a["label"] for a in manifest["assertions"])
    assert "title" not in manifest

    info = manifest["signature_info"]
    assert info["issuer"] == "Anymatix"
    assert info["common_name"] == signing.SIGNER_COMMON_NAME
    assert ("time" in info) == timestamped
    assert SECRET not in json.dumps(report)
    return report


def assert_not_signed(path):
    with pytest.raises(c2pa.C2paError):
        c2pa_report(path)


@pytest.fixture
def fresh_tsa_state(monkeypatch):
    """No backoff carried over from another test."""
    monkeypatch.setattr(signing, "_tsa_unreachable_until", 0.0)


# ----------------------------------------------------------------- images ----


@pytest.mark.parametrize("case", SIGNED_IMAGE_CASES, ids=_case_id)
def test_every_image_format_is_signed_through_write_image(tmp_path, case):
    extension, kwargs, channels = case
    directory = tmp_path / SECRET
    directory.mkdir()
    path = str(directory / f"{SECRET}_0001.{extension}")
    formats.write_image(a_gradient(channels=channels), path, extension=extension, **kwargs)

    assert_signed_untrusted(path)
    assert_nothing_leaked(path)
    assert_labelled_per_exiftool(path)
    assert no_stray_temp_files(str(directory)) == []


def _decode(path, extension):
    """
    Pillow (libavif) for AVIF: OpenCV 4.13 sniffs an AVIF from its first 500
    bytes only and refuses any whose `meta` box starts later -- which a C2PA
    manifest, placed after `ftyp` as the spec asks, always makes it do.
    Measured 2026-09-28: Pillow 12.2 / libavif 1.4.1, macOS ImageIO and
    ffmpeg all decode the signed file.
    """
    if extension == "avif":
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


@pytest.mark.parametrize("case", SIGNED_IMAGE_CASES, ids=_case_id)
def test_signing_does_not_change_a_single_pixel(tmp_path, case):
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
        labelled = disclosure.mark_bytes(f.read(), extension)
    signed = signing.sign_bytes(labelled, extension)
    assert signed != labelled, "the format is signable and was not signed"
    path = str(tmp_path / f"signed.{extension}")
    with open(path, "wb") as f:
        f.write(signed)
    before, after = _decode(bare, extension), _decode(path, extension)
    assert before is not None and after is not None
    assert before.dtype == after.dtype and before.shape == after.shape
    assert np.array_equal(before, after)


@pytest.mark.parametrize("extension", ["exr", "bmp"])
def test_what_c2pa_cannot_carry_is_written_unsigned(tmp_path, extension):
    """exr keeps its IPTC label; bmp has no metadata at all. Nothing is faked."""
    path = str(tmp_path / f"a.{extension}")
    formats.write_image(a_gradient(), path, extension=extension)
    assert cv2.imread(path, cv2.IMREAD_UNCHANGED) is not None
    assert signing.sign_file(path, extension) is False
    assert_not_signed(path)
    if extension == "exr":
        assert_labelled_per_exiftool(path, xmp=False)


def test_the_manifest_adds_kilobytes_not_a_thumbnail(tmp_path):
    """
    A thumbnail would add hundreds of KB (measured on the experiment: +577 KB
    on a 768x512 PNG). Without one the manifest and the two certificates are
    about 4 KB.
    """
    bare = str(tmp_path / "bare.png")
    formats._write_image_to(bare, a_gradient(512, 768), "png", 100, False, 8, False)
    with open(bare, "rb") as f:
        data = f.read()
    assert len(signing.sign_bytes(data, "png")) - len(data) < 8 * 1024


# ------------------------------------------------------ the audio and video nodes ----


@pytest.mark.parametrize(
    "rung,extension,xmp",
    [("mp3", "mp3", True), ("wav", "wav", True), ("aac", "m4a", True), ("flac", "flac", False)],
)
def test_every_audio_rung_is_signed_through_the_node(nodes, tmp_path, rung, extension, xmp):
    import av

    nodes.folder_paths.output = str(tmp_path)
    tone = _a_tone(nodes.torch)
    saved = nodes.audio.AnymatixSaveAudio().save_audio(
        tone, output_path=SECRET, filename_prefix=SECRET, format=rung
    )
    assert saved["ui"]["audio"], "the save reported no file"
    path = str(tmp_path / SECRET / f"{SECRET}.{extension}")

    assert_signed_untrusted(path)
    assert_nothing_leaked(path)
    assert_labelled_per_exiftool(path, xmp=xmp)
    with av.open(path) as container:
        samples = sum(frame.samples for frame in container.decode(audio=0))
    assert samples >= tone["waveform"].shape[-1]
    assert no_stray_temp_files(str(tmp_path / SECRET)) == []


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
    ],
)
def test_every_video_rung_is_signed_through_the_node(nodes, tmp_path, rung, with_audio):
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

    assert_nothing_leaked(path)
    xmp = container_ext in ("mp4", "mov")
    if container_ext == "mkv":
        assert_not_signed(path)  # c2pa-rs has no Matroska handler
    else:
        assert_signed_untrusted(path)
    assert_labelled_per_exiftool(path, xmp=xmp)
    with av.open(path) as container:
        decoded = sum(1 for _ in container.decode(video=0))
    assert decoded == frames
    assert no_stray_temp_files(str(tmp_path / SECRET)) == []


# --------------------------------------------------------------- the timestamp ----


def test_a_reachable_tsa_timestamps_the_signature(tmp_path, monkeypatch, fresh_tsa_state):
    url = signing.TSA_URL
    try:
        socket.create_connection(("timestamp.digicert.com", 80), timeout=3).close()
    except OSError:
        pytest.skip("the public TSA is not reachable from here")
    monkeypatch.setenv("ANYMATIX_C2PA_TSA", url)
    path = str(tmp_path / "a.png")
    formats.write_image(a_gradient(), path, extension="png")
    report = assert_signed_untrusted(path, timestamped=True)
    assert "time" in report["manifests"][report["active_manifest"]]["signature_info"]


def test_a_refused_tsa_costs_only_the_timestamp(tmp_path, monkeypatch, capsys, fresh_tsa_state):
    monkeypatch.setenv("ANYMATIX_C2PA_TSA", "http://127.0.0.1:9/")
    path = str(tmp_path / "a.png")
    formats.write_image(a_gradient(), path, extension="png")
    assert_signed_untrusted(path, timestamped=False)
    assert "timestamp authority unreachable" in capsys.readouterr().out

    # Backed off: the next save does not even try.
    def must_not_probe(*_args, **_kwargs):
        raise AssertionError("the TSA was probed during the back-off")

    monkeypatch.setattr(signing.socket, "create_connection", must_not_probe)
    second = str(tmp_path / "b.png")
    formats.write_image(a_gradient(), second, extension="png")
    assert_signed_untrusted(second, timestamped=False)


def test_a_tsa_that_drops_packets_is_given_up_on_quickly(tmp_path, monkeypatch, fresh_tsa_state):
    """
    Left to c2pa-rs, a TSA that drops packets costs 30 s and then the whole
    signature (measured). The probe bounds it. 192.0.2.1 is TEST-NET-1
    (RFC 5737): never routed.
    """
    monkeypatch.setenv("ANYMATIX_C2PA_TSA", "http://192.0.2.1/")
    monkeypatch.setattr(signing, "TSA_PROBE_SECONDS", 0.5)
    path = str(tmp_path / "a.png")
    started = time.monotonic()
    formats.write_image(a_gradient(), path, extension="png")
    assert time.monotonic() - started < 3
    assert_signed_untrusted(path, timestamped=False)


def test_a_tsa_that_answers_with_an_error_costs_only_the_timestamp(
    tmp_path, monkeypatch, capsys, fresh_tsa_state
):
    """Reachable by TCP, useless by HTTP: c2pa-rs fails, and the retry signs."""

    class Refuses(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(500)
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Refuses)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("ANYMATIX_C2PA_TSA", f"http://127.0.0.1:{server.server_port}/")
        path = str(tmp_path / "a.png")
        formats.write_image(a_gradient(), path, extension="png")
    finally:
        server.shutdown()
        server.server_close()
    assert_signed_untrusted(path, timestamped=False)
    assert "timestamp failed, signing without one" in capsys.readouterr().out


def test_a_large_batch_is_never_timestamped(tmp_path, monkeypatch, fresh_tsa_state):
    """`fast` is set for a batch of more than four: no TSA round-trip per frame."""
    monkeypatch.setenv("ANYMATIX_C2PA_TSA", signing.TSA_URL)

    def must_not_probe(_url):
        raise AssertionError("a fast write asked for a timestamp")

    monkeypatch.setattr(signing, "_tsa_reachable", must_not_probe)
    path = str(tmp_path / "a.png")
    formats.write_image(a_gradient(), path, extension="png", fast=True)
    assert_signed_untrusted(path, timestamped=False)


# ------------------------------------------------------ failure is harmless ----


def _assert_labelled_readable_unsigned(path, directory):
    assert cv2.imread(path) is not None
    assert_not_signed(path)
    with open(path, "rb") as f:
        assert disclosure.XMP_PACKET in f.read()
    assert_labelled_per_exiftool(path)
    assert no_stray_temp_files(directory) == []


def test_a_signing_failure_leaves_a_labelled_file_and_a_successful_save(
    tmp_path, monkeypatch, capsys
):
    def broken(*_args):
        raise RuntimeError("simulated")

    monkeypatch.setattr(signing, "_sign", broken)
    path = str(tmp_path / "a.png")
    formats.write_image(a_gradient(), path, extension="png")
    _assert_labelled_readable_unsigned(path, str(tmp_path))
    out = capsys.readouterr().out
    assert out.count("C2PA manifest not written to a png") == 1


def test_a_missing_c2pa_library_leaves_a_labelled_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "c2pa", None)  # `import c2pa` raises
    path = str(tmp_path / "a.png")
    formats.write_image(a_gradient(), path, extension="png")
    monkeypatch.undo()
    _assert_labelled_readable_unsigned(path, str(tmp_path))
    assert "C2PA manifest not written to a png" in capsys.readouterr().out


def test_an_unwritable_credential_directory_leaves_a_labelled_file(tmp_path, monkeypatch):
    blocker = tmp_path / "a-file"
    blocker.write_bytes(b"")
    monkeypatch.setenv("ANYMATIX_C2PA_DIR", str(blocker / "cannot-be-a-directory"))
    path = str(tmp_path / "a.png")
    formats.write_image(a_gradient(), path, extension="png")
    _assert_labelled_readable_unsigned(path, str(tmp_path))


def test_sign_bytes_returns_the_bytes_untouched_on_failure(monkeypatch):
    def broken(*_args):
        raise RuntimeError("simulated")

    monkeypatch.setattr(signing, "_sign", broken)
    data = disclosure.mark_bytes(b"not really an mp3", "mp3")
    assert signing.sign_bytes(data, "mp3") == data
    assert signing.sign_bytes(b"{}", "json") == b"{}"


# ------------------------------------------------------ the signing credential ----


def _serial(path):
    report = c2pa_report(path)
    return report["manifests"][report["active_manifest"]]["signature_info"]["cert_serial_number"]


def test_the_key_is_generated_once_owner_only_and_reused(tmp_path, monkeypatch, capsys):
    home = tmp_path / "credentials"
    monkeypatch.setenv("ANYMATIX_C2PA_DIR", str(home))
    first, second = str(tmp_path / "a.png"), str(tmp_path / "b.jpg")
    formats.write_image(a_gradient(), first, extension="png")
    credential = home / signing.CREDENTIAL_FILE
    stored = credential.read_bytes()
    formats.write_image(a_gradient(), second, extension="jpg")

    assert credential.read_bytes() == stored
    assert _serial(first) == _serial(second)
    assert capsys.readouterr().out.count("generating this install's C2PA signing credential") == 1
    assert sorted(os.listdir(home)) == [signing.CREDENTIAL_FILE]
    # One private key -- the signer's. The root's is never written.
    assert stored.count(b"PRIVATE KEY-----") == 2  # BEGIN and END of one block
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(credential).st_mode) == 0o600
        assert stat.S_IMODE(os.stat(home).st_mode) == 0o700


def test_the_certificates_say_anymatix_and_self_signed():
    from cryptography import x509

    _key, chain = signing.parse_credential(signing.generate_credential())
    signer, root = x509.load_pem_x509_certificates(chain)
    for certificate, name in ((signer, signing.SIGNER_COMMON_NAME), (root, signing.ROOT_COMMON_NAME)):
        assert certificate.subject.rfc4514_string() == f"O=Anymatix,CN={name}"
        assert "self-signed" in name and "untrusted" in name
    assert signer.issuer == root.subject == root.issuer


@pytest.mark.parametrize("damage", ["malformed", "expired"])
def test_a_damaged_credential_is_replaced(tmp_path, monkeypatch, capsys, damage):
    home = tmp_path / "credentials"
    home.mkdir()
    credential = home / signing.CREDENTIAL_FILE
    if damage == "malformed":
        credential.write_bytes(b"-----BEGIN PRIVATE KEY-----\ntruncated")
    else:
        long_ago = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)
        credential.write_bytes(signing.generate_credential(now=long_ago, signer_days=10))
    monkeypatch.setenv("ANYMATIX_C2PA_DIR", str(home))
    path = str(tmp_path / "a.png")
    formats.write_image(a_gradient(), path, extension="png")
    assert_signed_untrusted(path)
    assert "replacing the C2PA signing credential" in capsys.readouterr().out
    signing.parse_credential(credential.read_bytes())


def test_the_default_credential_directory_is_outside_every_repository(monkeypatch):
    monkeypatch.delenv("ANYMATIX_C2PA_DIR")
    directory = os.path.abspath(signing.credential_dir())
    root = os.path.dirname(os.path.abspath(signing.__file__))
    assert not directory.startswith(root)
    assert directory.startswith(os.path.expanduser("~")) or "LOCALAPPDATA" in os.environ
