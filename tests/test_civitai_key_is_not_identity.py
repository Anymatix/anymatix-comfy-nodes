#!/usr/bin/env python3
"""
THE CIVITAI KEY IS NOT PART OF A WEIGHT'S IDENTITY.

Measured on fmt-5000 (2026-10-20): a keyless run of Flux 1.S failed with
`401 Unauthorized` although the 17 GB checkpoint was on disk, because the
sidecar was named `sha256(url + "?token=<key>")` and the keyless run looked
for `sha256(url)`. Adding, changing or removing a key made every Civitai
weight on the machine another url, downloaded again.

The key is for the REQUEST: it must reach the server, and nothing else.

    python3 -m pytest tests/test_civitai_key_is_not_identity.py -q
"""
import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fetch
from fetch import download_file, hash_string

BASE = "https://civitai.com/api/download/models/699279?type=Model&format=SafeTensor"


def _effective(key):
    return BASE + "&token=" + key


class _Response:
    def __init__(self, headers, payload=None):
        self.headers = headers
        self.history = []
        self.payload = payload

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        _Server.bytes_served += len(self.payload or b"")
        yield self.payload or b""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Server:
    """Civitai with a key: serves the weight, and records every url asked for."""

    bytes_served = 0
    asked = []

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kwargs):
        _Server.asked.append(url)
        return _Response(
            {
                "Content-Length": str(len(self.payload)),
                "Content-Disposition": 'attachment; filename="flux_schnell.safetensors"',
                "Accept-Ranges": "none",
            },
            payload=self.payload,
        )


class _Requests:
    def __init__(self, payload):
        self.payload = payload
        self.RequestException = Exception

    def Session(self):
        return _Server(self.payload)


class Unauthorized(BaseException):
    """What Civitai answers a keyless download. BaseException so that
    `fetch_headers` cannot swallow it into a different code path."""


def _serving(monkeypatch, payload):
    monkeypatch.setattr(fetch, "requests", _Requests(payload))
    monkeypatch.setattr(fetch, "REQUESTS_AVAILABLE", True)
    _Server.bytes_served = 0
    _Server.asked = []


def _refusing(monkeypatch):
    for name in ("fetch_headers", "fetch", "fetch_parallel", "check_range_support"):
        monkeypatch.setattr(
            fetch, name,
            lambda *a, **k: (_ for _ in ()).throw(Unauthorized("401 for a weight on disk")),
        )


def test_the_key_reaches_the_request_and_not_the_sidecar_name(monkeypatch):
    payload = b"flux schnell" * 300
    with tempfile.TemporaryDirectory() as d:
        _serving(monkeypatch, payload)
        got = download_file(url=BASE, dir=d, effective_url=_effective("KEY1"),
                            redact_append="token=KEY1")

        assert any("token=KEY1" in u for u in _Server.asked)
        assert os.path.isfile(os.path.join(d, "%s.json" % hash_string(BASE)))
        assert not os.path.exists(os.path.join(d, "%s.json" % hash_string(_effective("KEY1"))))
        with open(got, "rb") as f:
            assert f.read() == payload


def test_a_new_key_downloads_nothing(monkeypatch):
    payload = b"flux schnell" * 300
    with tempfile.TemporaryDirectory() as d:
        _serving(monkeypatch, payload)
        first = download_file(url=BASE, dir=d, effective_url=_effective("KEY1"))

        _serving(monkeypatch, payload)
        _refusing(monkeypatch)
        assert download_file(url=BASE, dir=d, effective_url=_effective("KEY2")) == first
        assert _Server.bytes_served == 0


def test_no_key_runs_a_weight_a_keyed_run_downloaded(monkeypatch):
    payload = b"flux schnell" * 300
    with tempfile.TemporaryDirectory() as d:
        _serving(monkeypatch, payload)
        first = download_file(url=BASE, dir=d, effective_url=_effective("KEY1"))

        _refusing(monkeypatch)
        assert download_file(url=BASE, dir=d) == first


def test_the_fmt_5000_shape_heals_without_a_request(monkeypatch):
    # Exactly what the machine held on 2026-10-20: the weight under its content
    # name, the sidecar a keyed run wrote (named after the credentialled url,
    # storing the base), and a stale keyless sidecar from an earlier failed
    # attempt, named after the base url, with `file_size: null` and no file.
    payload = b"flux schnell" * 300
    sha = hashlib.sha256(payload).hexdigest()
    with tempfile.TemporaryDirectory() as d:
        name = "flux_schnell_%s.safetensors" % sha
        with open(os.path.join(d, name), "wb") as f:
            f.write(payload)
        keyed = os.path.join(d, "%s.json" % hash_string(_effective("KEY1")))
        with open(keyed, "w") as f:
            json.dump({"url": BASE, "file_name": name, "file_size": len(payload),
                       "sha256": sha}, f)
        stale = os.path.join(d, "%s.json" % hash_string(BASE))
        with open(stale, "w") as f:
            json.dump({"url": BASE, "file_name": "flux_schnell_%s.safetensors" % hash_string(BASE),
                       "file_size": None}, f)

        _refusing(monkeypatch)
        assert download_file(url=BASE, dir=d) == os.path.join(d, name)

        # The record moved to the name the url is looked up by, so the next
        # run is a `stat`, and the stale one is gone.
        assert not os.path.exists(keyed)
        with open(stale) as f:
            assert json.load(f)["file_name"] == name
        assert download_file(url=BASE, dir=d, effective_url=_effective("KEY2")) == os.path.join(d, name)


def test_an_old_keyed_sidecar_on_the_volume_is_found_from_the_cache(monkeypatch):
    # On a pod `dir` is the NVMe cache and the bytes are on the volume.
    payload = b"flux schnell" * 300
    sha = hashlib.sha256(payload).hexdigest()
    with tempfile.TemporaryDirectory() as cache, tempfile.TemporaryDirectory() as volume:
        name = "flux_schnell_%s.safetensors" % sha
        with open(os.path.join(volume, name), "wb") as f:
            f.write(payload)
        with open(os.path.join(volume, "%s.json" % hash_string(_effective("KEY1"))), "w") as f:
            json.dump({"url": BASE, "file_name": name, "file_size": len(payload),
                       "sha256": sha}, f)

        _refusing(monkeypatch)
        assert download_file(url=BASE, dir=cache, adopt_dirs=[volume]) == os.path.join(volume, name)
        assert os.path.isfile(os.path.join(volume, "%s.json" % hash_string(BASE)))
        assert not os.listdir(cache)


def test_a_sidecar_for_another_url_is_never_taken(monkeypatch):
    payload = b"some other model" * 300
    sha = hashlib.sha256(payload).hexdigest()
    with tempfile.TemporaryDirectory() as d:
        name = "other_%s.safetensors" % sha
        with open(os.path.join(d, name), "wb") as f:
            f.write(payload)
        with open(os.path.join(d, "%s.json" % hash_string("https://civitai.com/api/download/models/1?token=K")), "w") as f:
            json.dump({"url": "https://civitai.com/api/download/models/1", "file_name": name,
                       "file_size": len(payload), "sha256": sha}, f)

        assert fetch.sidecar_recording([d], BASE) is None
