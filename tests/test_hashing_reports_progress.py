#!/usr/bin/env python3
"""
A HASH IS WORK, AND WORK THAT IS WATCHED REPORTS ITSELF.

Vincenzo, 2026-09-17, looking at FETCH CHECKPOINT at 0% with an empty bar and
asking "is this the first hashing maybe?" -- it was -- then: "it could probably
even report the hashing progress."

The bar had one vocabulary, downloaded-bytes-over-total, and one label, the
item's own name. So the 23 seconds it takes to prove a 12 GB weight already on
disk showed as a download stalled at nothing. Nothing was wrong except what the
screen said. bugs/fetch-model-sits-0-while-fetcher-hashing

These pin both halves: that a hash reports where it has got to, and that the
fetcher says which of its three phases each number belongs to.

    python3 -m pytest tests/test_hashing_reports_progress.py -q
"""
import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fetch
from fetch import compute_file_sha256, download_file, fetch_phase_message, hash_string


URL = ("https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/"
       "7d41107b653d3039be20972fb82398b01b3213eb/split_files/diffusion_models/"
       "qwen_image_edit_2511_bf16.safetensors")
OLD_URL = ("https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/"
           "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors")


class _FakeResponse:
    def __init__(self, headers, history=(), payload=None):
        self.headers = headers
        self.history = list(history)
        self.payload = payload

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.payload or b""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeSession:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kwargs):
        sha = hashlib.sha256(self.payload).hexdigest()
        return _FakeResponse(
            {
                "Content-Length": str(len(self.payload)),
                "Content-Disposition": 'inline; filename="qwen_image_edit_2511_bf16.safetensors";',
                "Accept-Ranges": "none",
            },
            history=[_FakeResponse({"X-Linked-Etag": '"%s"' % sha})],
            payload=self.payload,
        )


class _FakeRequests:
    def __init__(self, payload):
        self.payload = payload
        self.RequestException = Exception

    def Session(self):
        return _FakeSession(self.payload)


def _serving(monkeypatch, payload):
    monkeypatch.setattr(fetch, "requests", _FakeRequests(payload))
    monkeypatch.setattr(fetch, "REQUESTS_AVAILABLE", True)


def _write(path, payload: bytes) -> str:
    with open(path, "wb") as f:
        f.write(payload)
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# The hash itself


def test_a_hash_reports_where_it_has_got_to():
    payload = b"x" * (4 * 1024 * 1024)
    seen = []
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "weights.bin")
        _write(path, payload)
        got = compute_file_sha256(path, chunk_size=64 * 1024,
                                  progress=lambda done, total: seen.append((done, total)))

    assert got == hashlib.sha256(payload).hexdigest()
    # It says where it starts and where it ends, so a bar driven by it begins
    # at 0 and arrives at 100 rather than stopping wherever the last step fell.
    assert seen[0] == (0, len(payload))
    assert seen[-1] == (len(payload), len(payload))
    # Monotonic, and never past the end.
    assert seen == sorted(seen)
    assert all(0 <= done <= total for done, total in seen)


def test_a_big_file_does_not_report_once_per_gigabyte():
    # The number that matters: HASH_PROGRESS_STEPS is a fraction of the file,
    # not a byte count, so the granularity does not collapse as files grow.
    # 200 steps is ComfyUI's own 0.5% floor, below which it drops the update.
    assert fetch.HASH_PROGRESS_STEPS == 200

    payload = b"y" * (4 * 1024 * 1024)
    seen = []
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "weights.bin")
        _write(path, payload)
        compute_file_sha256(path, chunk_size=1024,
                            progress=lambda done, total: seen.append(done))

    # 200 steps plus the opening 0 and the closing total.
    assert 150 <= len(seen) <= 205, len(seen)


def test_a_file_smaller_than_one_step_still_starts_and_finishes():
    seen = []
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "small.json")
        _write(path, b"{}")
        compute_file_sha256(path, progress=lambda done, total: seen.append((done, total)))
    assert seen[0] == (0, 2)
    assert seen[-1] == (2, 2)


def test_an_empty_file_does_not_divide_by_its_own_size():
    seen = []
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "empty.bin")
        _write(path, b"")
        got = compute_file_sha256(path, progress=lambda done, total: seen.append((done, total)))
    assert got == hashlib.sha256(b"").hexdigest()
    assert seen[-1] == (0, 0)


def test_a_hash_with_no_listener_is_unchanged():
    payload = b"z" * 5000
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "weights.bin")
        _write(path, payload)
        assert compute_file_sha256(path) == hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# Which phase each number belongs to


def test_a_verification_is_announced_before_the_bar_it_drives(monkeypatch):
    """The label cannot be left over from the phase before it.

    `phase` is called BEFORE the progress it describes, so every number the
    caller sees is already under the right word. Asserted by recording both on
    one timeline and checking the order, not by checking that both happened.
    """
    payload = b"already here" * 500
    sha = hashlib.sha256(payload).hexdigest()
    timeline = []

    with tempfile.TemporaryDirectory() as d:
        name = "qwen_image_edit_2511_bf16_%s.safetensors" % sha
        _write(os.path.join(d, name), payload)
        with open(os.path.join(d, "%s.json" % hash_string(OLD_URL)), "w") as f:
            json.dump({"url": OLD_URL, "file_name": name,
                       "file_size": len(payload), "sha256": sha}, f)

        _serving(monkeypatch, payload)
        got = download_file(
            url=URL, dir=d,
            callback=lambda done, total: timeline.append(("progress", done, total)),
            phase=lambda name: timeline.append(("phase", name, None)),
        )
        assert got == os.path.join(d, name)

    phases = [name for kind, name, _ in timeline if kind == "phase"]
    assert phases == ["verifying", "adopting"], phases

    # Every progress tick falls after `verifying` and before `adopting`: the
    # hash is the only thing measured on this path, and it is measured under
    # the word for it.
    verifying_at = next(i for i, e in enumerate(timeline) if e[:2] == ("phase", "verifying"))
    adopting_at = next(i for i, e in enumerate(timeline) if e[:2] == ("phase", "adopting"))
    ticks = [i for i, e in enumerate(timeline) if e[0] == "progress"]
    assert ticks, "a 23-second wait reported nothing"
    assert all(verifying_at < i < adopting_at for i in ticks)


def test_a_download_says_fetching_and_then_verifies_what_it_wrote(monkeypatch):
    payload = b"fresh weights" * 600
    timeline = []

    with tempfile.TemporaryDirectory() as d:
        _serving(monkeypatch, payload)
        download_file(
            url=URL, dir=d,
            callback=lambda done, total: timeline.append(("progress", done, total)),
            phase=lambda name: timeline.append(("phase", name, None)),
        )

    phases = [name for kind, name, _ in timeline if kind == "phase"]
    # Nothing on disk to adopt, so: fetch the bytes, then hash them to name the
    # file by its content. Both are waits and both are named.
    assert phases == ["fetching", "verifying"], phases


def test_a_resumed_part_file_is_verified_under_its_own_word(monkeypatch):
    payload = b"interrupted then finished" * 300
    timeline = []

    with tempfile.TemporaryDirectory() as d:
        provisional = "model_%s.safetensors" % hash_string(URL)
        part = os.path.join(d, provisional + ".part")
        _write(part, payload)
        fetch.mark_part_complete(part, len(payload))
        with open(os.path.join(d, "%s.json" % hash_string(URL)), "w") as f:
            json.dump({"url": URL, "file_name": provisional,
                       "file_size": len(payload),
                       "remote_sha256": hashlib.sha256(payload).hexdigest()}, f)

        _serving(monkeypatch, payload)
        download_file(
            url=URL, dir=d,
            callback=lambda done, total: timeline.append(("progress", done, total)),
            phase=lambda name: timeline.append(("phase", name, None)),
        )

    phases = [name for kind, name, _ in timeline if kind == "phase"]
    # THE WORST CASE THE ITEM MEASURED, and it is honest about both halves.
    # Where the server stated a hash, the part file is proven against it and
    # then the result is hashed again to name it by its content: two full
    # passes over a 12 GB weight, ~2x23 s, and no fetch at all. Two bars, each
    # starting at zero, each under the word for what it is.
    assert phases == ["verifying", "verifying"], phases


def test_a_part_file_proven_by_its_completion_record_hashes_only_once(monkeypatch):
    # No server hash to check against, so the resumed part is proven by the
    # downloader's own record instead - which reads no bytes. Only the naming
    # hash is a wait, and only it is announced.
    payload = b"interrupted then finished" * 300
    phases = []

    with tempfile.TemporaryDirectory() as d:
        provisional = "model_%s.safetensors" % hash_string(URL)
        part = os.path.join(d, provisional + ".part")
        _write(part, payload)
        fetch.mark_part_complete(part, len(payload))
        with open(os.path.join(d, "%s.json" % hash_string(URL)), "w") as f:
            json.dump({"url": URL, "file_name": provisional,
                       "file_size": len(payload)}, f)

        _serving(monkeypatch, payload)
        download_file(url=URL, dir=d, phase=lambda name: phases.append(name))

    assert phases == ["verifying"], phases


def test_a_satisfied_url_announces_no_phase_at_all(monkeypatch):
    # Nothing is waited on, so there is nothing to label. A phase announced
    # here would put a step on screen for work that did not happen.
    payload = b"qwen edit weights" * 400
    with tempfile.TemporaryDirectory() as d:
        _serving(monkeypatch, payload)
        first = download_file(url=URL, dir=d)

        phases = []
        got = download_file(url=URL, dir=d, phase=lambda name: phases.append(name))
        assert got == first
        assert phases == []


def test_a_fetch_with_no_listener_still_works(monkeypatch):
    payload = b"fresh weights" * 600
    with tempfile.TemporaryDirectory() as d:
        _serving(monkeypatch, payload)
        got = download_file(url=URL, dir=d)
        with open(got, "rb") as f:
            assert f.read() == payload


# --------------------------------------------------------------------------
# The message that carries the phase to the app


def test_the_phase_message_carries_the_prompt_id():
    """THE FIELD THAT WAS EASY TO FORGET, AND ONCE WAS.

    The app's global websocket dispatcher routes every message by `prompt_id`
    and drops the ones that have none, so a phase sent without it is built,
    sent, and silently never arrives. The first cut of this work omitted it and
    the app's typecheck caught it, not a person watching a bar.
    """
    message = fetch_phase_message("7", "abc-123", "verifying", 400, 1000)
    assert message == {
        "node": "7",
        "prompt_id": "abc-123",
        "phase": "verifying",
        "value": 400,
        "max": 1000,
    }


def test_no_prompt_id_and_no_node_means_no_message():
    # Outside a running prompt there is nobody to tell, which is a normal state
    # for the fetcher's own tests rather than an error.
    assert fetch_phase_message(None, "abc-123", "verifying", 0, 1000) is None
    assert fetch_phase_message("7", None, "verifying", 0, 1000) is None


def test_only_the_three_words_the_fetcher_actually_uses_are_sent():
    assert fetch.FETCH_PHASES == ("fetching", "verifying", "adopting")
    for phase in fetch.FETCH_PHASES:
        assert fetch_phase_message("7", "p", phase, 0, 1000) is not None
    # A typo must not reach the app, where an unknown word falls back to the
    # item's name and the wait goes unexplained again.
    assert fetch_phase_message("7", "p", "verifiying", 0, 1000) is None
