#!/usr/bin/env python3
"""
A SIDECAR WITH NO `file_size` KEY IS A SIDECAR WHOSE SERVER STATED NO LENGTH.

`satisfied_by_sidecar` and `remembered_size` already read it that way
(`data.get("file_size")`), but `download_file` indexed `data["file_size"]`, so
a sidecar written before the key existed raised `KeyError: 'file_size'`.

Measured 2026-10-08 on fmt-4000 (Windows, beta.17-internal): 13 Civitai
sidecars dated April 2026 carry only url, file_name, sha256 and data. Every
card that names one of them -- Flux 1.S, Qwen, Krea 2 and others -- failed
with "Failed to download checkpoint model: 'file_size'" while the model file
sat complete on disk.

    python3 -m pytest tests/test_legacy_sidecar_without_file_size.py -q
"""
import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fetch
from fetch import download_file, hash_string

URL = "https://civitai.com/api/download/models/699279?type=Model&format=SafeTensor&size=pruned&fp=fp32"


class _Session:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kwargs):
        raise AssertionError("a model already on disk must not be downloaded again")

    def head(self, url, **kwargs):
        raise AssertionError("a model already on disk must not be downloaded again")


class _Requests:
    RequestException = Exception

    def __init__(self, payload):
        self.payload = payload

    def Session(self):
        return _Session(self.payload)


def _legacy_store(d, payload):
    sha = hashlib.sha256(payload).hexdigest()
    name = "flux_schnell_%s.safetensors" % sha
    with open(os.path.join(d, name), "wb") as f:
        f.write(payload)
    with open(os.path.join(d, "%s.json" % hash_string(URL)), "w") as f:
        json.dump({"url": URL, "file_name": name, "sha256": sha,
                   "data": {"id": 699279, "files": [{"hashes": {"SHA256": sha.upper()}}]}}, f)
    return os.path.join(d, name)


def test_a_legacy_sidecar_without_file_size_resolves_its_model(monkeypatch):
    payload = b"flux schnell weights" * 300
    monkeypatch.setattr(fetch, "requests", _Requests(payload))
    monkeypatch.setattr(fetch, "REQUESTS_AVAILABLE", True)
    with tempfile.TemporaryDirectory() as d:
        path = _legacy_store(d, payload)
        assert download_file(url=URL, dir=d) == path
