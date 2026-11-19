#!/usr/bin/env python3
"""
A DOWNLOAD IS WRITTEN INSIDE ITS MODELS FOLDER, UNDER A NAME THAT CANNOT
LEAVE IT — AND A .safetensors LINK LANDS AS A .safetensors FILE.

App repo: naima bugs/model-downloads-trust-server-s-file-name (a server's name,
or a sidecar's, joined into the folder unreduced) and
bugs/collection-card-can-make-machine-download-pickle (a card from a GitHub
collection is refused unless its link names a .safetensors file; the server
must not then turn it into a `.ckpt` at download time).

    python3 -m pytest tests/test_download_stays_in_its_folder.py -q
"""
import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import fetch
from fetch import UnsafeModelFileName, download_file, hash_string, model_path_in


def safetensors_bytes(body=b"\0" * 64):
    header = json.dumps({"__metadata__": {"modelspec.title": "x"}}).encode()
    return struct.pack("<Q", len(header)) + header + body


class _Resp:
    def __init__(self, headers, payload=b""):
        self.headers = headers
        self.history = []
        self.status_code = 200
        self.payload = payload

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Server:
    def __init__(self, payload, disposition):
        self.payload = payload
        self.disposition = disposition

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kwargs):
        headers = {"Content-Length": str(len(self.payload)), "Accept-Ranges": "none"}
        if self.disposition:
            headers["Content-Disposition"] = self.disposition
        return _Resp(headers, payload=self.payload)


class _Requests:
    def __init__(self, server):
        self.server = server
        self.RequestException = Exception

    def Session(self):
        return self.server

    def get(self, url, **kwargs):
        return self.server.get(url, **kwargs)


def _serve(monkeypatch, payload, disposition):
    server = _Server(payload, disposition)
    monkeypatch.setattr(fetch, "requests", _Requests(server))
    monkeypatch.setattr(fetch, "REQUESTS_AVAILABLE", True)
    monkeypatch.setattr(fetch, "fetch_parallel", lambda *a, **k: False)
    return server


def _inside(d, path):
    root = os.path.realpath(d)
    return os.path.realpath(path).startswith(root + os.sep)


# --- the guard itself --------------------------------------------------------

@pytest.mark.parametrize("name", [
    "../evil.safetensors",
    "..",
    ".",
    "",
    ".hidden.safetensors",
    "sub/model.safetensors",
    "sub\\model.safetensors",
    "/etc/passwd",
    "C:\\Windows\\x.safetensors",
    "model\x00.safetensors",
    "model\n.safetensors",
    None,
])
def test_a_name_that_could_leave_the_folder_is_refused(name):
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(UnsafeModelFileName):
            model_path_in(d, name)


def test_a_bare_name_stays_inside():
    with tempfile.TemporaryDirectory() as d:
        path = model_path_in(d, "flux_abc123.safetensors")
        assert path == os.path.join(d, "flux_abc123.safetensors")
        assert _inside(d, path)


def test_a_symlinked_name_that_points_outside_is_refused():
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as outside:
        os.symlink(os.path.join(outside, "x.safetensors"), os.path.join(d, "x.safetensors"))
        with pytest.raises(UnsafeModelFileName):
            model_path_in(d, "x.safetensors")


# --- through a download ------------------------------------------------------

def test_a_hostile_content_disposition_writes_inside_the_folder(monkeypatch):
    with tempfile.TemporaryDirectory() as root:
        d = os.path.join(root, "models", "loras")
        _serve(monkeypatch, safetensors_bytes(), 'attachment; filename="../../escaped.safetensors"')
        got = download_file(url="https://example.com/files/abc", dir=d)
        assert _inside(d, got)
        assert not os.path.exists(os.path.join(root, "escaped.safetensors"))
        assert [n for n in os.listdir(root) if n != "models"] == []


def test_a_sidecar_naming_a_path_outside_is_refused(monkeypatch):
    # A sidecar written before the header name was reduced, or planted: its
    # file_name is joined into the folder on every later run.
    url = "https://example.com/files/abc.safetensors"
    with tempfile.TemporaryDirectory() as root:
        d = os.path.join(root, "loras")
        os.makedirs(d)
        with open(os.path.join(d, "%s.json" % hash_string(url)), "w") as f:
            json.dump({"url": url, "file_name": "../outside.safetensors", "name": "outside.safetensors",
                       "file_size": 10}, f)
        _serve(monkeypatch, safetensors_bytes(), None)
        with pytest.raises(UnsafeModelFileName):
            download_file(url=url, dir=d)
        assert not os.path.exists(os.path.join(root, "outside.safetensors"))


def test_a_url_path_of_dot_dot_does_not_name_the_file(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        _serve(monkeypatch, safetensors_bytes(), None)
        got = download_file(url="https://example.com/a/..", dir=d)
        assert _inside(d, got)
        assert not os.path.basename(got).startswith(".")


# --- a .safetensors link lands as a .safetensors file ------------------------

def test_a_safetensors_link_the_server_renames_to_ckpt_is_refused(monkeypatch):
    url = "https://example.com/u/model.safetensors"
    with tempfile.TemporaryDirectory() as d:
        _serve(monkeypatch, b"PK\x03\x04 pickle", 'attachment; filename="model.ckpt"')
        with pytest.raises(UnsafeModelFileName) as err:
            download_file(url=url, dir=d)
        assert "model.ckpt" in str(err.value)
        assert [n for n in os.listdir(d) if not n.endswith(".json")] == []


def test_a_safetensors_link_named_safetensors_by_the_server_downloads(monkeypatch):
    url = "https://example.com/u/model.safetensors"
    with tempfile.TemporaryDirectory() as d:
        _serve(monkeypatch, safetensors_bytes(), 'attachment; filename="Model Final.safetensors"')
        got = download_file(url=url, dir=d)
        assert got.endswith(".safetensors") and _inside(d, got)


def test_a_link_with_no_extension_keeps_the_server_name(monkeypatch):
    # Civitai links name no file: the rule above does not apply to them.
    with tempfile.TemporaryDirectory() as d:
        _serve(monkeypatch, safetensors_bytes(), 'attachment; filename="lora.safetensors"')
        got = download_file(url="https://civitai.com/api/download/models/1", dir=d)
        assert os.path.basename(got).startswith("lora_")


def test_a_sidecar_pointing_outside_is_not_a_model_here():
    with tempfile.TemporaryDirectory() as root:
        d = os.path.join(root, "loras")
        os.makedirs(d)
        with open(os.path.join(root, "outside.safetensors"), "wb") as f:
            f.write(b"x" * 10)
        assert fetch.satisfied_by_sidecar(d, {"file_name": "../outside.safetensors", "file_size": 10}) is None
