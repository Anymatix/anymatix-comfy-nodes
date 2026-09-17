#!/usr/bin/env python3
"""
A URL THIS MACHINE HAS ALREADY FETCHED COSTS A `stat`, AND NOTHING ELSE.

Vincenzo, 2026-09-17:

    "if so hashing can be 'cached' by renaming correct files once and for all;
     also if a model for a url has been fetched, the fetcher (in python) must
     guarantee no url lookup."

It is the product's own claim -- local-first, works on your own machine -- so a
card whose models are all present must reach the sampler with zero network
activity and zero re-hashing. The strongest form of it: a fully-downloaded card
still runs on a machine with no internet.

HOW THESE ASSERT IT MATTERS. Counting log lines proves nothing about a request
issued somewhere the log does not mention, so the second fetch is run with
every network entry point replaced by one that RAISES, and with
`compute_file_sha256` replaced by one that RAISES. The test does not measure
that the cost was small; it makes the cost impossible.

Both exceptions derive from BaseException on purpose: `fetch_headers` swallows
`Exception` and returns empty headers, which would turn a failure into a
different code path instead of a red test.

    python3 -m pytest tests/test_a_satisfied_url_costs_nothing.py -q
"""
import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fetch
from fetch import download_file, hash_string


# --------------------------------------------------------------------------
# A server that answers like Hugging Face, and one that must never be asked


class _FakeResponse:
    def __init__(self, headers, history=(), payload=None):
        self.headers = headers
        self.history = list(history)
        self.payload = payload

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        _FakeSession.bytes_served += len(self.payload or b"")
        yield self.payload or b""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeSession:
    """Serves the payload and states its sha256 in `X-Linked-Etag`."""

    requests_made = 0
    bytes_served = 0

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kwargs):
        _FakeSession.requests_made += 1
        sha = hashlib.sha256(self.payload).hexdigest()
        redirect = _FakeResponse({"X-Linked-Etag": '"%s"' % sha})
        return _FakeResponse(
            {
                "Content-Length": str(len(self.payload)),
                "Content-Disposition": 'inline; filename="qwen_image_edit_2511_bf16.safetensors";',
                "Accept-Ranges": "none",
            },
            history=[redirect],
            payload=self.payload,
        )


class _FakeRequests:
    def __init__(self, payload):
        self.payload = payload
        self.RequestException = Exception

    def Session(self):
        return _FakeSession(self.payload)


class NetworkTouched(BaseException):
    """Raised where a satisfied fetch must never go."""


class HashComputed(BaseException):
    """Raised where a satisfied fetch must never go."""


class _ExplodingSession:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kwargs):
        raise NetworkTouched("GET issued for a url already on disk: %s" % url)

    def head(self, url, **kwargs):
        raise NetworkTouched("HEAD issued for a url already on disk: %s" % url)


class _ExplodingRequests:
    def __init__(self):
        self.RequestException = Exception

    def Session(self):
        return _ExplodingSession()

    def get(self, url, **kwargs):
        raise NetworkTouched("GET issued for a url already on disk: %s" % url)

    def head(self, url, **kwargs):
        raise NetworkTouched("HEAD issued for a url already on disk: %s" % url)


def _no_network(monkeypatch):
    """Unplug the machine, as far as `fetch` can tell."""
    monkeypatch.setattr(fetch, "requests", _ExplodingRequests())
    monkeypatch.setattr(fetch, "REQUESTS_AVAILABLE", True)
    for name in ("fetch_headers", "fetch", "fetch_parallel", "check_range_support"):
        monkeypatch.setattr(
            fetch, name,
            lambda *a, **k: (_ for _ in ()).throw(NetworkTouched("%s was called" % name)),
        )


def _no_hashing(monkeypatch):
    monkeypatch.setattr(
        fetch, "compute_file_sha256",
        lambda *a, **k: (_ for _ in ()).throw(HashComputed("a file was hashed again")),
    )


def _serving(monkeypatch, payload):
    monkeypatch.setattr(fetch, "requests", _FakeRequests(payload))
    monkeypatch.setattr(fetch, "REQUESTS_AVAILABLE", True)
    _FakeSession.requests_made = 0
    _FakeSession.bytes_served = 0


def _write(path, payload: bytes) -> str:
    with open(path, "wb") as f:
        f.write(payload)
    return hashlib.sha256(payload).hexdigest()


URL = ("https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/"
       "7d41107b653d3039be20972fb82398b01b3213eb/split_files/diffusion_models/"
       "qwen_image_edit_2511_bf16.safetensors")
OLD_URL = ("https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/"
           "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors")


# --------------------------------------------------------------------------
# Guarantee 1 and 2, on the plain path


def test_a_second_fetch_of_a_downloaded_url_touches_nothing(monkeypatch):
    payload = b"qwen edit weights" * 400

    with tempfile.TemporaryDirectory() as d:
        _serving(monkeypatch, payload)
        first = download_file(url=URL, dir=d)
        assert _FakeSession.bytes_served == len(payload)

        _no_network(monkeypatch)
        _no_hashing(monkeypatch)
        assert download_file(url=URL, dir=d) == first


def test_a_second_fetch_after_an_adoption_touches_nothing(monkeypatch):
    # The shape a8f13a50b left every machine in: the bytes are on disk under
    # their content name, the sidecar is named after the OLD url. The first
    # fetch of the new url hashes the file once and records it; the second must
    # not hash it again.
    payload = b"already here" * 500
    sha = hashlib.sha256(payload).hexdigest()

    with tempfile.TemporaryDirectory() as d:
        name = "qwen_image_edit_2511_bf16_%s.safetensors" % sha
        _write(os.path.join(d, name), payload)
        with open(os.path.join(d, "%s.json" % hash_string(OLD_URL)), "w") as f:
            json.dump({"url": OLD_URL, "file_name": name,
                       "file_size": len(payload), "sha256": sha}, f)

        _serving(monkeypatch, payload)
        first = download_file(url=URL, dir=d)
        assert first == os.path.join(d, name)
        assert _FakeSession.bytes_served == 0

        _no_network(monkeypatch)
        _no_hashing(monkeypatch)
        assert download_file(url=URL, dir=d) == first


def test_a_second_fetch_after_a_cross_directory_adoption_touches_nothing(monkeypatch):
    """THE ONE THAT WAS PAYING TWICE, AND WOULD HAVE GONE ON PAYING.

    On a pod the download goes to the NVMe cache (`dir`) while the models from
    every earlier run are on the volume (`adopt_dirs`). Adoption writes the
    sidecar beside the file it names -- on the volume -- and removes the one in
    the cache, because a sidecar in the cache dies with the container. The next
    call for the same url and the same `dir` then found no sidecar at all, and
    paid for a header request and a full re-hash of a 12 GB weight. Every run.
    """
    payload = b"volume weights" * 700
    sha = hashlib.sha256(payload).hexdigest()

    with tempfile.TemporaryDirectory() as cache, tempfile.TemporaryDirectory() as durable:
        name = "qwen_image_edit_2511_bf16_%s.safetensors" % sha
        _write(os.path.join(durable, name), payload)

        _serving(monkeypatch, payload)
        first = download_file(url=URL, dir=cache, adopt_dirs=[durable])
        assert first == os.path.join(durable, name)
        assert os.path.isfile(os.path.join(durable, "%s.json" % hash_string(URL)))

        _no_network(monkeypatch)
        _no_hashing(monkeypatch)
        assert download_file(url=URL, dir=cache, adopt_dirs=[durable]) == first


def test_a_json_already_fetched_is_not_fetched_again(monkeypatch):
    # A sidecar with no `file_size` -- servers that state no Content-Length --
    # is still satisfied when the file it names parses.
    payload = b'{"ok": true}'

    with tempfile.TemporaryDirectory() as cache, tempfile.TemporaryDirectory() as durable:
        with open(os.path.join(durable, "info.json"), "wb") as f:
            f.write(payload)
        with open(os.path.join(durable, "%s.json" % hash_string(URL)), "w") as f:
            json.dump({"url": URL, "file_name": "info.json", "file_size": None}, f)

        _no_network(monkeypatch)
        _no_hashing(monkeypatch)
        assert download_file(url=URL, dir=cache, adopt_dirs=[durable]) == \
            os.path.join(durable, "info.json")


# --------------------------------------------------------------------------
# What the guarantee does NOT excuse


def test_a_file_that_changed_size_is_proven_again(monkeypatch):
    payload = b"qwen edit weights" * 400

    with tempfile.TemporaryDirectory() as d:
        _serving(monkeypatch, payload)
        first = download_file(url=URL, dir=d)

        # Something truncated it. The declared size no longer matches, so the
        # sidecar's record proves nothing and the bytes are fetched again.
        with open(first, "r+b") as f:
            f.truncate(len(payload) // 2)

        _serving(monkeypatch, payload)
        again = download_file(url=URL, dir=d)
        assert _FakeSession.bytes_served == len(payload)
        with open(again, "rb") as f:
            assert f.read() == payload


def test_a_missing_file_is_downloaded_again(monkeypatch):
    payload = b"qwen edit weights" * 400

    with tempfile.TemporaryDirectory() as d:
        _serving(monkeypatch, payload)
        first = download_file(url=URL, dir=d)
        os.remove(first)

        _serving(monkeypatch, payload)
        again = download_file(url=URL, dir=d)
        assert _FakeSession.bytes_served == len(payload)
        assert again == first


def test_an_adopt_dir_sidecar_whose_file_is_gone_is_not_believed(monkeypatch):
    # The sidecar is on the volume; the model is not. A `stat` is the whole
    # check precisely so that this case answers honestly.
    payload = b"volume weights" * 700
    sha = hashlib.sha256(payload).hexdigest()

    with tempfile.TemporaryDirectory() as cache, tempfile.TemporaryDirectory() as durable:
        name = "qwen_image_edit_2511_bf16_%s.safetensors" % sha
        with open(os.path.join(durable, "%s.json" % hash_string(URL)), "w") as f:
            json.dump({"url": URL, "file_name": name,
                       "file_size": len(payload), "sha256": sha}, f)

        _serving(monkeypatch, payload)
        got = download_file(url=URL, dir=cache, adopt_dirs=[durable])
        assert _FakeSession.bytes_served == len(payload)
        assert os.path.dirname(got) == cache


# --------------------------------------------------------------------------
# What a CALLER can ask before it goes to the network
#
# `anymatix_checkpoint_fetcher` cannot be imported outside ComfyUI, so the
# question it asks lives here, where it can be pinned.


CIVITAI = "https://civitai.com/api/download/models/1234"
CIVITAI_AUTHED = CIVITAI + "?token=secret"


def test_a_url_already_on_disk_is_recognised_without_a_request():
    payload = b"a civitai lora" * 50
    sha = hashlib.sha256(payload).hexdigest()

    with tempfile.TemporaryDirectory() as d:
        name = "lora_%s.safetensors" % sha
        _write(os.path.join(d, name), payload)
        with open(os.path.join(d, "%s.json" % hash_string(CIVITAI)), "w") as f:
            json.dump({"url": CIVITAI, "file_name": name,
                       "file_size": len(payload), "sha256": sha}, f)

        assert fetch.satisfied_locally([d], [CIVITAI]) == os.path.join(d, name)


def test_the_sidecar_is_found_under_the_effective_url_too():
    # A credentialled url is what the sidecar is NAMED after; the base url is
    # what it stores. A caller that knows only one of the two must still find it.
    payload = b"a civitai lora" * 50
    sha = hashlib.sha256(payload).hexdigest()

    with tempfile.TemporaryDirectory() as d:
        name = "lora_%s.safetensors" % sha
        _write(os.path.join(d, name), payload)
        with open(os.path.join(d, "%s.json" % hash_string(CIVITAI_AUTHED)), "w") as f:
            json.dump({"url": CIVITAI, "file_name": name,
                       "file_size": len(payload), "sha256": sha}, f)

        assert fetch.satisfied_locally([d], [CIVITAI]) is None
        assert fetch.satisfied_locally([d], [CIVITAI, CIVITAI_AUTHED]) == os.path.join(d, name)


def test_a_url_whose_file_is_short_is_not_satisfied():
    payload = b"a civitai lora" * 50
    sha = hashlib.sha256(payload).hexdigest()

    with tempfile.TemporaryDirectory() as d:
        name = "lora_%s.safetensors" % sha
        _write(os.path.join(d, name), payload[:10])
        with open(os.path.join(d, "%s.json" % hash_string(CIVITAI)), "w") as f:
            json.dump({"url": CIVITAI, "file_name": name,
                       "file_size": len(payload), "sha256": sha}, f)

        assert fetch.satisfied_locally([d], [CIVITAI]) is None


def test_an_unknown_url_is_never_satisfied():
    with tempfile.TemporaryDirectory() as d:
        assert fetch.satisfied_locally([d, None], [CIVITAI, None]) is None


# --------------------------------------------------------------------------
# The line that made a correct system read as a broken one


def test_the_adoption_log_says_which_hash_is_which(capsys):
    """`[ANYMATIX ADOPT] verifying X against <64 hex> for Y` named two 64-hex
    values and said what neither of them was, so it read as a file verified
    against one hash and adopted under another. The url hash it printed --
    `4518bf83...` on fmt-5000 -- is `sha256` of the url STRING, proven by
    hashing the shipped url. A log line that makes a correct system look broken
    costs a reading, and this one already has.
    """
    payload = b"the weights" * 100
    sha = hashlib.sha256(payload).hexdigest()
    url_hash = hash_string(URL)

    with tempfile.TemporaryDirectory() as d:
        name = "qwen_image_edit_2511_bf16_%s.safetensors" % sha
        _write(os.path.join(d, name), payload)
        got = fetch.adopt_existing_file(
            [d], name, sha, len(payload), "%s.json" % url_hash,
            "qwen_image_edit_2511_bf16_%s.safetensors" % url_hash,
        )
        assert got == os.path.join(d, name)

    out = capsys.readouterr().out
    assert "content sha256 %s" % sha in out
    assert "url hash %s" % url_hash in out


def test_the_url_hash_in_that_line_is_the_sha256_of_the_url_string():
    # The value the report asked about, computed from the shipped url.
    assert hash_string(URL) == \
        "4518bf83ae6faac3af8c6f3b832e5c01baa9201ffffa43bdecfbb2eb549408f1"
