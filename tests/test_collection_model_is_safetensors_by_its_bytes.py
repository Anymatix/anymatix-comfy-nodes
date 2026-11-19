#!/usr/bin/env python3
"""
A CARD FROM A GITHUB COLLECTION LOADS SAFETENSORS FILES ONLY, AND THE BYTES DECIDE.

App repo: naima todos/collection-cards-check-each-model-s-content (and
bugs/collection-card-can-make-machine-download-pickle). The app sends
`safetensors_only: true` in the fetcher's `url` for a collection card; the
fetcher reads the first bytes before downloading (an 8-byte little-endian
header length, then a JSON object) and checks the whole header on disk before
anything loads it. A pickle, a zip, a web page, a GGUF or a truncated file is
refused, naming the file, and a download it stopped leaves nothing behind.
Without the flag nothing changes.

    python3 -m pytest tests/test_collection_model_is_safetensors_by_its_bytes.py -q
"""
import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import fetch
from fetch import (
    NotSafetensorsModel,
    check_safetensors_file,
    download_file,
    hash_string,
    safetensors_prefix_problem,
)

from test_download_stays_in_its_folder import _serve  # noqa: E402  (the same fake server)


def safetensors_bytes(body=b"\0" * 64):
    header = json.dumps({"w": {"dtype": "F16", "shape": [32], "data_offsets": [0, 64]}}).encode()
    return struct.pack("<Q", len(header)) + header + body


PICKLE = b"\x80\x02}q\x00(X\x01\x00\x00\x00aq\x01K\x01u." + b"\0" * 64
ZIP = b"PK\x03\x04\x14\x00\x00\x00\x08\x00" + b"\0" * 64
HTML = b"<!DOCTYPE html><html><body>Not found</body></html>" + b" " * 32
GGUF = b"GGUF\x03\x00\x00\x00" + b"\0" * 64
TRUNCATED = safetensors_bytes()[:20]

BAD = {"pickle": PICKLE, "zip": ZIP, "html": HTML, "gguf": GGUF, "truncated": TRUNCATED}


# --- the byte rule ----------------------------------------------------------

def test_a_real_safetensors_header_passes():
    data = safetensors_bytes()
    assert safetensors_prefix_problem(data, ended=True, total_size=len(data)) is None


def test_the_start_of_a_long_header_passes_while_the_stream_goes_on():
    data = safetensors_bytes()
    assert safetensors_prefix_problem(data[:12], ended=False) is None


@pytest.mark.parametrize("name", sorted(BAD))
def test_every_other_kind_of_file_is_refused(name):
    data = BAD[name]
    assert safetensors_prefix_problem(data, ended=True, total_size=len(data)) is not None


def test_what_it_is_instead_is_said():
    assert "pickle" in safetensors_prefix_problem(PICKLE, ended=True)
    assert "zip" in safetensors_prefix_problem(ZIP, ended=True)
    assert "web page" in safetensors_prefix_problem(HTML, ended=True)
    assert "GGUF" in safetensors_prefix_problem(GGUF, ended=True)
    assert "cut short" in safetensors_prefix_problem(TRUNCATED, ended=True)


def test_a_header_length_past_the_bound_is_refused():
    data = struct.pack("<Q", fetch.SAFETENSORS_HEADER_MAX_BYTES + 1) + b"{" + b"\0" * 32
    assert safetensors_prefix_problem(data, ended=False) is not None


def test_a_header_that_is_not_json_is_refused():
    data = struct.pack("<Q", 6) + b"{nope!" + b"\0" * 8
    assert "not JSON" in safetensors_prefix_problem(data, ended=True)


def test_a_header_longer_than_the_file_is_refused():
    data = safetensors_bytes()
    assert safetensors_prefix_problem(data[:30], ended=False, total_size=30) is not None


@pytest.mark.parametrize("name", sorted(BAD))
def test_a_file_on_disk_is_checked_the_same_way(name):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        with open(path, "wb") as f:
            f.write(BAD[name])
        with pytest.raises(NotSafetensorsModel) as raised:
            check_safetensors_file(path)
        assert "model.safetensors" in str(raised.value)
        assert "Cards from GitHub collections run safetensors models only." in str(raised.value)
        assert os.path.exists(path), "a file already here is refused, never deleted"


def test_a_real_file_on_disk_passes():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        with open(path, "wb") as f:
            f.write(safetensors_bytes())
        check_safetensors_file(path)


# --- through a download ------------------------------------------------------

CIVITAI = "https://civitai.com/api/download/models/123456"


def test_a_civitai_link_serving_safetensors_downloads(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        _serve(monkeypatch, safetensors_bytes(), 'attachment; filename="style.safetensors"')
        got = download_file(url=CIVITAI, dir=d, safetensors_only=True)
        assert got.endswith(".safetensors") and os.path.isfile(got)


@pytest.mark.parametrize("name", ["pickle", "zip", "html", "gguf"])
def test_a_collection_download_that_is_not_safetensors_leaves_nothing(monkeypatch, name):
    with tempfile.TemporaryDirectory() as d:
        _serve(monkeypatch, BAD[name], 'attachment; filename="renamed.safetensors"')
        with pytest.raises(NotSafetensorsModel) as raised:
            download_file(url=CIVITAI, dir=d, safetensors_only=True)
        assert "renamed.safetensors" in str(raised.value)
        assert "nothing was kept" in str(raised.value)
        assert os.listdir(d) == [], f"left behind: {os.listdir(d)}"


def test_a_ckpt_link_from_a_collection_is_stopped_at_its_first_bytes(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        server = _serve(monkeypatch, PICKLE, None)
        gets = []
        real_get = server.get
        server.get = lambda url, **kw: (gets.append(kw.get("headers")), real_get(url, **kw))[1]
        with pytest.raises(NotSafetensorsModel) as raised:
            download_file(url="https://example.com/m/model.ckpt", dir=d, safetensors_only=True)
        assert "model.ckpt" in str(raised.value)
        assert any(h and "Range" in h for h in gets), "the first bytes were not asked for"
        assert os.listdir(d) == []


def test_a_server_that_swaps_the_bytes_after_the_probe_is_caught_on_disk(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        server = _serve(monkeypatch, safetensors_bytes(), None)
        real_get = server.get

        def get(url, **kw):
            if kw.get("headers", {}).get("Range"):
                return real_get(url, **kw)
            server.payload = PICKLE.ljust(len(safetensors_bytes()), b"\0")
            return real_get(url, **kw)

        server.get = get
        with pytest.raises(NotSafetensorsModel):
            download_file(url="https://example.com/m/x.safetensors", dir=d, safetensors_only=True)
        assert [n for n in os.listdir(d) if not n.endswith(".json")] == []
        assert not os.path.exists(os.path.join(d, f"{hash_string('https://example.com/m/x.safetensors')}.json"))


def test_without_the_flag_nothing_changes(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        _serve(monkeypatch, PICKLE, None)
        got = download_file(url="https://example.com/m/model.ckpt", dir=d)
        assert os.path.isfile(got)


def test_a_pickle_already_here_is_refused_for_a_collection_card_and_kept(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        url = "https://example.com/m/model.ckpt"
        _serve(monkeypatch, PICKLE, None)
        got = download_file(url=url, dir=d)
        with pytest.raises(NotSafetensorsModel) as raised:
            download_file(url=url, dir=d, safetensors_only=True)
        assert "not loaded" in str(raised.value)
        assert os.path.isfile(got), "the library's own file is not deleted"


# --- the fetcher node ---------------------------------------------------------

def _fetcher(monkeypatch):
    from test_a_stop_is_not_a_failed_download import InterruptProcessingException, _load
    return _load(monkeypatch, InterruptProcessingException)


def test_the_node_passes_the_flag_and_the_refusal_arrives_unwrapped(monkeypatch):
    module = _fetcher(monkeypatch)
    seen = []

    def refusing(**kwargs):
        seen.append(kwargs.get("safetensors_only"))
        raise NotSafetensorsModel("“x.ckpt” is not a safetensors file")

    monkeypatch.setattr(module, "download_file", refusing)
    with pytest.raises(NotSafetensorsModel) as raised:
        module.AnymatixFetcher().download_model(
            {"type": "lora", "url": CIVITAI, "safetensors_only": True})
    assert seen == [True]
    assert str(raised.value).startswith("“x.ckpt”")


def test_the_node_without_the_flag_asks_for_no_check(monkeypatch):
    module = _fetcher(monkeypatch)
    seen = []

    def ok(**kwargs):
        seen.append(kwargs.get("safetensors_only"))
        return "/tmp/x.safetensors"

    monkeypatch.setattr(module, "download_file", ok)
    module.AnymatixFetcher().download_model({"type": "lora", "url": CIVITAI})
    assert seen == [False]
