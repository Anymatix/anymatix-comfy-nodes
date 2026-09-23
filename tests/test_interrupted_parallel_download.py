#!/usr/bin/env python3
"""
AN INTERRUPTED PARALLEL DOWNLOAD MUST NEVER BE MISTAKEN FOR A FINISHED ONE.

`AsyncParallelDownloader` pre-allocates its part file to the final length
before a single byte arrives, then writes each segment at its offset. So a
transfer killed at 1% leaves a file of EXACTLY the declared size and almost no
content, and the resume path's question -- "is the part file as long as the
model?" -- answers yes.

Measured on fmt-5000, 2026-09-17: ComfyUI died mid-download, the next run
adopted a 12 GB z-image weight of exactly 12,309,866,400 bytes whose sha256 was
`50638dd8...` where Hugging Face states `24076130...`, renamed it into place and
recorded our own measurement as the truth about that url. The card rendered
uniform RGB noise and reported success.
bugs/a-parallel-download-pre-allocates-part-file-so

What these tests pin:

1. completion is a FACT THE DOWNLOADER RECORDS, not a length anybody can read;
2. where the server states a hash, a resumed file is checked against it before
   it is accepted, exactly as an adopted file already was;
3. a sidecar never records a hash the server did not state;
4. and none of it costs the resume: a genuinely short part file still carries
   on from its real prefix instead of re-fetching 12 GB.

    python3 -m pytest tests/test_interrupted_parallel_download.py -q
"""
import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fetch
from fetch import download_file, hash_string, part_path_for

# One server, one set of fakes: both harnesses already exist and a second copy
# of either is a copy that drifts.
from test_parallel_download import serve
from test_url_repointing_adoption import _FakeRequests, _FakeSession


def _provisional(url: str) -> str:
    name = url.rsplit("/", 1)[-1]
    stem, _, ext = name.rpartition(".")
    return f"{stem}_{hash_string(url)}.{ext}"


def _kill_a_parallel_download(directory: str, url: str, size: int) -> str:
    """What a SIGKILL mid-parallel-download leaves on disk: a part file of the
    full declared length holding almost nothing. Reproduced rather than
    described, because the whole defect is that this is indistinguishable from
    success by every test the code used to apply."""
    part = part_path_for(os.path.join(directory, _provisional(url)))
    with open(part, "wb") as f:
        f.truncate(size)
    assert os.path.getsize(part) == size
    return part


# --------------------------------------------------------------------------
# 1. The downloader records what it knows


def test_a_finished_parallel_download_records_its_completion():
    blob = os.urandom(1 << 20) * 6
    server, port, _ = serve(blob)
    directory = tempfile.mkdtemp()
    part = os.path.join(directory, "weights.bin.part")
    try:
        assert fetch.fetch_parallel(f"http://127.0.0.1:{port}/weights.bin", part) is True
        assert fetch.part_completion_is_recorded(part, len(blob)), \
            "the downloader finished and said nothing"
    finally:
        server.shutdown()


def test_the_record_dies_with_the_bytes_it_describes():
    directory = tempfile.mkdtemp()
    final = os.path.join(directory, "model.safetensors")
    part = part_path_for(final)
    with open(part, "wb") as f:
        f.write(b"x" * 100)
    fetch.mark_part_complete(part, 100)

    fetch.finalize_download(part, final, 100, "model.safetensors")
    assert not os.path.exists(fetch.completion_marker_for(part)), \
        "a record left behind would vouch for the NEXT part file written here"


def test_a_record_for_a_different_length_is_not_this_files_record():
    directory = tempfile.mkdtemp()
    part = os.path.join(directory, "model.safetensors.part")
    with open(part, "wb") as f:
        f.write(b"x" * 100)
    fetch.mark_part_complete(part, 100)
    assert fetch.part_completion_is_recorded(part, 100)
    assert not fetch.part_completion_is_recorded(part, 200)


# --------------------------------------------------------------------------
# 2. A full-size part file proves nothing on its own


def test_a_killed_parallel_download_is_downloaded_again_not_adopted():
    blob = os.urandom(1 << 20) * 6
    server, port, _ = serve(blob)
    directory = tempfile.mkdtemp()
    url = f"http://127.0.0.1:{port}/weights.bin"
    try:
        hole = _kill_a_parallel_download(directory, url, len(blob))
        got = download_file(url=url, dir=directory)
        assert open(got, "rb").read() == blob, "the hole was served as a model"
        assert not os.path.exists(hole)
    finally:
        server.shutdown()


def test_the_bytes_that_come_back_are_the_bytes_the_server_has():
    """The same gesture told from the other end: whatever the run decides to do
    with the part file, what it returns hashes to what the server serves."""
    blob = os.urandom(1 << 20) * 6
    server, port, _ = serve(blob)
    directory = tempfile.mkdtemp()
    url = f"http://127.0.0.1:{port}/weights.bin"
    try:
        _kill_a_parallel_download(directory, url, len(blob))
        got = download_file(url=url, dir=directory)
        assert hashlib.sha256(open(got, "rb").read()).hexdigest() == \
            hashlib.sha256(blob).hexdigest()
    finally:
        server.shutdown()


# --------------------------------------------------------------------------
# 3. Where the server states a hash, it is checked -- and believed over us


def test_a_full_size_part_file_is_checked_against_the_servers_hash(monkeypatch):
    payload = b"z-image turbo weights" * 500
    url = "https://huggingface.co/x/y/resolve/abc/model.safetensors"
    monkeypatch.setattr(fetch, "requests", _FakeRequests(payload))
    monkeypatch.setattr(fetch, "REQUESTS_AVAILABLE", True)

    with tempfile.TemporaryDirectory() as d:
        provisional = "model_%s.safetensors" % hash_string(url)
        part = os.path.join(d, provisional + ".part")
        # The full length, the wrong bytes, and NO completion record: the
        # pre-allocated file of a killed transfer. The server's hash refuses it.
        # (With a record it is believed -- see the next test -- since the
        # 2026-09-23 size skip.)
        with open(part, "wb") as f:
            f.write(b"\0" * len(payload))
        with open(os.path.join(d, "%s.json" % hash_string(url)), "w") as f:
            json.dump({
                "url": url,
                "file_name": provisional,
                "file_size": len(payload),
                "remote_sha256": hashlib.sha256(payload).hexdigest(),
            }, f)

        _FakeSession.bytes_served = 0
        got = download_file(url=url, dir=d)

        assert _FakeSession.bytes_served > 0, "the wrong bytes were accepted"
        assert open(got, "rb").read() == payload
        assert not os.path.exists(part)


def test_a_completion_record_is_believed_without_hashing(monkeypatch):
    """SKIP VERIFICATION IF SIZE MATCHES (Vincenzo, 2026-09-23).

    For a part file the size is pre-allocated and proves nothing; the
    downloader's completion record is what "the whole declared length arrived"
    means. It is now asked BEFORE the server's hash, so a process that died
    between the last byte and the rename costs a rename, not a multi-GB sha256.
    The price is named here on purpose: a part file whose record is right and
    whose bytes are wrong is accepted.
    """
    payload = b"z-image turbo weights" * 500
    url = "https://huggingface.co/x/y/resolve/abc/model.safetensors"
    monkeypatch.setattr(fetch, "requests", _FakeRequests(payload))
    monkeypatch.setattr(fetch, "REQUESTS_AVAILABLE", True)

    def no_hash(*a, **k):
        raise AssertionError("a size-proven file was hashed")
    monkeypatch.setattr(fetch, "compute_file_sha256", no_hash)

    with tempfile.TemporaryDirectory() as d:
        provisional = "model_%s.safetensors" % hash_string(url)
        part = os.path.join(d, provisional + ".part")
        with open(part, "wb") as f:
            f.write(payload)
        fetch.mark_part_complete(part, len(payload))
        with open(os.path.join(d, "%s.json" % hash_string(url)), "w") as f:
            json.dump({
                "url": url,
                "file_name": provisional,
                "file_size": len(payload),
                "remote_sha256": hashlib.sha256(payload).hexdigest(),
            }, f)

        _FakeSession.bytes_served = 0
        got = download_file(url=url, dir=d)

        assert _FakeSession.bytes_served == 0
        assert open(got, "rb").read() == payload
        with open(os.path.join(d, "%s.json" % hash_string(url))) as f:
            assert json.load(f)["verification"] == "size"


def test_a_download_of_the_stated_length_is_accepted_on_its_size():
    """The other half of the 2026-09-23 reversal, and what it gives up.

    The server states a hash the bytes do not have, but the length is right:
    the download is accepted on its size and NOT hashed. What still holds from
    the z-image lesson below: our own measurement is never recorded as the
    truth about this url (nothing was measured), and the file is not given a
    content-addressed name, because only a computed hash may produce one."""
    blob = os.urandom(1 << 20)
    server, port, _ = serve(blob)
    directory = tempfile.mkdtemp()
    url = f"http://127.0.0.1:{port}/weights.bin"
    lie = "a" * 64
    try:
        with open(os.path.join(directory, "%s.json" % hash_string(url)), "w") as f:
            json.dump({
                "url": url,
                "file_name": _provisional(url),
                "file_size": len(blob),
                "remote_sha256": lie,
            }, f)

        got = download_file(url=url, dir=directory)

        assert os.path.basename(got) == _provisional(url), \
            "a content-addressed name was produced without computing a hash"
        with open(os.path.join(directory, "%s.json" % hash_string(url))) as f:
            stored = json.load(f)
        assert stored["verification"] == "size"
        assert stored["verified_size"] == len(blob)
        assert stored.get("sha256") != hashlib.sha256(blob).hexdigest()
    finally:
        server.shutdown()


def test_a_sidecar_never_records_a_hash_the_server_did_not_state():
    """The measured hash may name the file. It may not overrule the server.

    This is what turned the z-image corruption from detectable into permanent:
    the file was renamed to `<base>_50638dd8...safetensors` and its sidecar
    recorded `50638dd8...` as the sha256 of a url Hugging Face says serves
    `24076130...`.
    """
    blob = os.urandom(1 << 20)
    server, port, _ = serve(blob)
    directory = tempfile.mkdtemp()
    url = f"http://127.0.0.1:{port}/weights.bin"
    lie = "a" * 64
    try:
        # A warm sidecar that already carries the server's claim, and the claim
        # is not what the server will serve. NO LENGTH: since 2026-09-23 a
        # stated length that matches is accepted without a hash (the test
        # above), so the hash — and this check — runs only where the server
        # stated none.
        with open(os.path.join(directory, "%s.json" % hash_string(url)), "w") as f:
            json.dump({
                "url": url,
                "file_name": _provisional(url),
                "file_size": None,
                "remote_sha256": lie,
            }, f)

        raised = ""
        try:
            download_file(url=url, dir=directory)
        except Exception as e:
            raised = str(e)

        assert "does not match the hash the server states" in raised, raised
        with open(os.path.join(directory, "%s.json" % hash_string(url))) as f:
            stored = json.load(f)
        assert stored.get("sha256") != hashlib.sha256(blob).hexdigest(), \
            "our own measurement was recorded as the truth about this url"
        assert not os.path.exists(os.path.join(directory, _provisional(url))), \
            "a file we cannot vouch for was left where a card can load it"
    finally:
        server.shutdown()


# --------------------------------------------------------------------------
# 4. And the resume still resumes


def test_a_short_part_file_still_resumes_from_its_real_prefix():
    """The segmented downloader exists to avoid re-fetching 27 GB. None of the
    proof above may cost that: a part file SHORTER than the declared size is a
    real prefix, and the single stream carries on from it."""
    blob = os.urandom(1 << 20)  # under 5 MB, so this is the single-stream path
    server, port, state = serve(blob)
    directory = tempfile.mkdtemp()
    url = f"http://127.0.0.1:{port}/weights.bin"
    prefix = 300 * 1024
    try:
        part = part_path_for(os.path.join(directory, _provisional(url)))
        with open(part, "wb") as f:
            f.write(blob[:prefix])

        got = download_file(url=url, dir=directory)

        assert open(got, "rb").read() == blob
        resumed = [start for start, _end in state["gets"] if start == prefix]
        assert resumed, f"it did not resume from {prefix}: served {state['gets']}"
    finally:
        server.shutdown()
