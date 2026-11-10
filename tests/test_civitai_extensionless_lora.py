#!/usr/bin/env python3
"""
A CIVITAI LORA MUST REACH THE DISK WITH ITS EXTENSION.

Measured on sleipnir (2026-11-08, again 2026-11-10): a LoRA picked from
Discover, `https://civitai.com/api/download/models/3417654`, was saved as
`3417654_<sha256>` with no extension. ComfyUI chooses its loader by extension,
so the valid safetensors file went to `torch.load` and failed with
"Weights only load failed ... Unsupported operand 165". The name came from the
url's last segment because no name was read from the header request.

App repo: TRACKERS/BUGS/a-civitai-lora-picked-from-discover-saved.

    python3 -m pytest tests/test_civitai_extensionless_lora.py -q
"""
import hashlib
import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import fetch
from fetch import (
    download_file,
    ensure_model_extension,
    fetch_headers,
    filename_from_content_disposition,
    has_file_extension,
    hash_string,
    sniff_model_extension,
)

BASE = "https://civitai.com/api/download/models/3417654"
KEY = "SECRETKEY"
EFFECTIVE = BASE + "?token=" + KEY


def safetensors_bytes(body=b"\0" * 64):
    header = json.dumps({"__metadata__": {"modelspec.title": "moss"}}).encode()
    return struct.pack("<Q", len(header)) + header + body


# --- every form of Content-Disposition --------------------------------------

@pytest.mark.parametrize("value,expected", [
    ('attachment; filename="Amateur_Nudes_MOSS-Image.safetensors"', "Amateur_Nudes_MOSS-Image.safetensors"),
    ('attachment; filename=Amateur_Nudes_MOSS-Image.safetensors', "Amateur_Nudes_MOSS-Image.safetensors"),
    ("attachment; filename*=UTF-8''na%C3%AFve%20lora.safetensors", "na\u00efve lora.safetensors"),
    ("attachment; filename*=utf-8'en'moss.safetensors", "moss.safetensors"),
    ('attachment; filename="fallback.bin"; filename*=UTF-8\'\'real.safetensors', "real.safetensors"),
    ("attachment; filename*=UTF-8''real.safetensors; filename=\"fallback.bin\"", "real.safetensors"),
    ('attachment; filename="moss.safetensors"; size=129017232', "moss.safetensors"),
    ('attachment;filename=moss.safetensors;size=1', "moss.safetensors"),
    ('inline; FileName="moss.safetensors"', "moss.safetensors"),
    ('attachment; filename="we\\"ird.safetensors"', 'we"ird.safetensors'),
    ('attachment; filename="../../etc/moss.safetensors"', "moss.safetensors"),
    ('attachment; filename="C:\\\\Windows\\\\moss.safetensors"', "moss.safetensors"),
    ('attachment', None),
    ('', None),
    (None, None),
])
def test_every_content_disposition_form(value, expected):
    assert filename_from_content_disposition(value) == expected


# --- the header request carries the key --------------------------------------

class _Resp:
    def __init__(self, headers, status=200, payload=b""):
        self.headers = headers
        self.history = []
        self.status_code = status
        self.payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            err = Exception("%d Client Error: Unauthorized for url: %s" % (self.status_code, EFFECTIVE))
            err.response = self
            raise err

    def iter_content(self, chunk_size):
        yield self.payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Civitai:
    """Civitai for a key-only file: 401 without the key, the file with it."""

    asked = []

    def __init__(self, payload, disposition):
        self.payload = payload
        self.disposition = disposition

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kwargs):
        _Civitai.asked.append(url)
        if "token=" + KEY not in url:
            return _Resp({}, status=401)
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
    server = _Civitai(payload, disposition)
    monkeypatch.setattr(fetch, "requests", _Requests(server))
    monkeypatch.setattr(fetch, "REQUESTS_AVAILABLE", True)
    # The single-connection path, so the test does not depend on aiohttp.
    monkeypatch.setattr(fetch, "fetch_parallel", lambda *a, **k: False)
    _Civitai.asked = []
    return server


def test_the_header_request_carries_the_key(monkeypatch, capsys):
    payload = safetensors_bytes()
    server = _serve(monkeypatch, payload, 'attachment; filename="moss.safetensors"')
    got = fetch_headers(EFFECTIVE, server)
    assert got["file_name"] == "moss.safetensors"
    assert got["headers_failed"] is False
    assert _Civitai.asked == [EFFECTIVE]

    keyless = fetch_headers(BASE, server)
    assert keyless["file_name"] is None and keyless["headers_failed"] is True
    out = capsys.readouterr().out
    assert "401" in out and KEY not in out


def test_a_key_only_lora_gets_its_name_from_the_header(monkeypatch):
    payload = safetensors_bytes()
    with tempfile.TemporaryDirectory() as d:
        _serve(monkeypatch, payload, "attachment; filename*=UTF-8''moss_lora.safetensors")
        got = download_file(url=BASE, dir=d, effective_url=EFFECTIVE, redact_append="token=" + KEY)
        assert os.path.basename(got).startswith("moss_lora_")
        assert got.endswith(".safetensors")
        # Every request, the header one included, carried the key.
        assert _Civitai.asked and all("token=" + KEY in u for u in _Civitai.asked)


def test_no_header_name_falls_back_to_civitai_metadata(monkeypatch):
    payload = safetensors_bytes()
    info = {"files": [{"name": "training.zip", "type": "Training Data"},
                      {"name": "Amateur_Nudes_MOSS-Image.safetensors", "type": "Model", "primary": True}]}
    with tempfile.TemporaryDirectory() as d:
        _serve(monkeypatch, payload, None)
        got = download_file(url=BASE, dir=d, effective_url=EFFECTIVE, expand_info=lambda u: info)
        assert os.path.basename(got).startswith("Amateur_Nudes_MOSS-Image_")
        assert got.endswith(".safetensors")


def test_a_failed_header_request_leaves_no_sidecar(monkeypatch):
    # A keyless attempt: the header request and the download are both refused.
    with tempfile.TemporaryDirectory() as d:
        _serve(monkeypatch, safetensors_bytes(), 'attachment; filename="moss.safetensors"')
        with pytest.raises(Exception):
            download_file(url=BASE, dir=d)
        assert not os.path.exists(os.path.join(d, "%s.json" % hash_string(BASE)))


def test_the_sidecar_a_keyless_attempt_left_is_asked_again(monkeypatch):
    # Exactly what earlier code wrote after a 401: the url's last segment as the
    # name, no length, nothing on disk. The keyed run must not inherit it.
    payload = safetensors_bytes()
    with tempfile.TemporaryDirectory() as d:
        url_hash = hash_string(BASE)
        with open(os.path.join(d, "%s.json" % url_hash), "w") as f:
            json.dump({"url": BASE, "file_name": "3417654_%s" % url_hash, "name": "3417654",
                       "file_size": None, "remote_sha256": None}, f)
        _serve(monkeypatch, payload, 'attachment; filename="moss.safetensors"')
        got = download_file(url=BASE, dir=d, effective_url=EFFECTIVE)
        assert os.path.basename(got) == "moss_%s.safetensors" % url_hash


# --- the sniff backstop -------------------------------------------------------

def test_sniff_recognises_safetensors_only():
    with tempfile.TemporaryDirectory() as d:
        st = os.path.join(d, "a")
        with open(st, "wb") as f:
            f.write(safetensors_bytes())
        pickle = os.path.join(d, "b")
        with open(pickle, "wb") as f:
            f.write(b"\x80\x02}q\x00(X\x01\x00\x00\x00aq\x01K\x01u.")
        lying = os.path.join(d, "c")
        with open(lying, "wb") as f:
            f.write(struct.pack("<Q", 10 ** 9) + b"{")
        assert sniff_model_extension(st) == ".safetensors"
        assert sniff_model_extension(pickle) is None
        assert sniff_model_extension(lying) is None


def test_has_file_extension():
    assert has_file_extension("x.safetensors")
    assert not has_file_extension("3417654_" + "a" * 64)
    assert not has_file_extension("flux1.dev_" + "a" * 64)


def test_a_download_that_still_has_no_name_is_sniffed(monkeypatch, capsys):
    # No header name, no metadata: the url's last segment is all there is.
    payload = safetensors_bytes()
    with tempfile.TemporaryDirectory() as d:
        _serve(monkeypatch, payload, None)
        got = download_file(url=BASE, dir=d, effective_url=EFFECTIVE)
        assert os.path.basename(got) == "3417654_%s.safetensors" % hash_string(BASE)
        with open(got, "rb") as f:
            assert f.read() == payload
        with open(os.path.join(d, "%s.json" % hash_string(BASE))) as f:
            assert json.load(f)["file_name"] == os.path.basename(got)
        assert "renamed to" in capsys.readouterr().out


def test_a_non_safetensors_file_without_extension_is_left_alone(capsys):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "3417654_" + "b" * 64)
        with open(p, "wb") as f:
            f.write(b"PK\x03\x04 not a safetensors file")
        assert ensure_model_extension(p) == p
        assert "left as it is" in capsys.readouterr().out


# --- an extensionless file already on disk is repaired, not re-downloaded -----

def _refuse_network(monkeypatch):
    for name in ("fetch_headers", "fetch", "fetch_parallel", "check_range_support"):
        monkeypatch.setattr(
            fetch, name,
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("network touched")),
        )


def test_the_sleipnir_file_is_repaired_without_a_request(monkeypatch):
    # What sleipnir held: `3417654_<content sha256>`, its sidecar naming it,
    # hash-verified because the failed header request stated no length.
    payload = safetensors_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    with tempfile.TemporaryDirectory() as d:
        broken = "3417654_%s" % sha
        with open(os.path.join(d, broken), "wb") as f:
            f.write(payload)
        sidecar = os.path.join(d, "%s.json" % hash_string(BASE))
        with open(sidecar, "w") as f:
            json.dump({"url": BASE, "file_name": broken, "file_size": None, "sha256": sha,
                       "verification": "sha256", "verified_size": len(payload)}, f)
        # A second url adopted the same file: its sidecar must follow the rename.
        other = os.path.join(d, "%s.json" % hash_string(BASE + "?type=Model"))
        with open(other, "w") as f:
            json.dump({"url": BASE + "?type=Model", "file_name": broken, "sha256": sha,
                       "file_size": len(payload)}, f)

        _refuse_network(monkeypatch)
        got = download_file(url=BASE, dir=d, effective_url=EFFECTIVE)

        assert got == os.path.join(d, broken + ".safetensors")
        assert not os.path.exists(os.path.join(d, broken))
        with open(got, "rb") as f:
            assert f.read() == payload
        for path in (sidecar, other):
            with open(path) as f:
                assert json.load(f)["file_name"] == broken + ".safetensors"
        # And the next run is a `stat`, still with no request.
        assert download_file(url=BASE, dir=d, effective_url=EFFECTIVE) == got


def test_a_repaired_duplicate_reuses_the_named_twin():
    payload = safetensors_bytes()
    with tempfile.TemporaryDirectory() as d:
        bare = os.path.join(d, "3417654_" + "c" * 64)
        for p in (bare, bare + ".safetensors"):
            with open(p, "wb") as f:
                f.write(payload)
        assert ensure_model_extension(bare) == bare + ".safetensors"
        assert not os.path.exists(bare)


def test_a_different_twin_is_never_overwritten():
    with tempfile.TemporaryDirectory() as d:
        bare = os.path.join(d, "3417654_" + "c" * 64)
        with open(bare, "wb") as f:
            f.write(safetensors_bytes(b"\1" * 64))
        with open(bare + ".safetensors", "wb") as f:
            f.write(safetensors_bytes(b"\2" * 64))
        assert ensure_model_extension(bare) == bare
        assert os.path.exists(bare)
