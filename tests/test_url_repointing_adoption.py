#!/usr/bin/env python3
"""
A MODEL THE MACHINE ALREADY HAS SURVIVES A CHANGE OF ITS DOWNLOAD URL.

A cached file is named by the sha256 of its BYTES; its sidecar is named by the
sha256 of its URL. `a8f13a50b` (2026-09-16) repointed all 75 shipped Hugging
Face urls from `/resolve/main/` to `/resolve/<commit>/`, which changed every
sidecar name and no file name. Nothing looked for the file under its content
name, so every machine downloaded its whole model set again -- measured on
fmt-5000 on 2026-09-17, where z-image turbo re-fetched from zero.

These tests pin the rule: before spending bandwidth, ask the server what the
bytes should hash to, look for a file on disk that hashes to it, and adopt it.

    python3 -m pytest tests/test_url_repointing_adoption.py -q
"""
import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fetch
from fetch import (
    adopt_existing_file,
    canonical_model_name,
    content_sha256_from_headers,
    download_file,
    hash_string,
)


# Measured against Hugging Face on 2026-09-17 for
# Comfy-Org/Krea-2 .../loras/krea2_darkbrush.safetensors. Both headers are 64
# hex characters; only X-Linked-Etag is the sha256 of the bytes -- the file was
# downloaded and hashed, and it came out equal to the first and not the second.
HF_LINKED_ETAG = "f47c4316dd93af66e0518c93b582f459571d4925b519133770c73a52cd5db7c6"
HF_CDN_ETAG = "fdead7c2f83f6c9412f05c18f48b06122a470e890b5d79d558c2c0c112860d1a"


def _write(path, payload: bytes):
    with open(path, "wb") as f:
        f.write(payload)
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# What the server says the bytes are


def test_the_linked_etag_is_the_content_hash():
    assert content_sha256_from_headers(
        {"X-Linked-Etag": '"%s"' % HF_LINKED_ETAG}
    ) == HF_LINKED_ETAG


def test_the_cdn_etag_is_never_mistaken_for_the_content_hash():
    # The CDN's own ETag is the Xet content-addressing hash: 64 hex, a
    # different function of the same file. Naming a model after it would make
    # every later lookup miss.
    assert content_sha256_from_headers({"ETag": '"%s"' % HF_CDN_ETAG}) is None


def test_a_git_blob_sha1_is_refused():
    # Non-LFS files answer with a 40-hex git blob sha1 in both headers.
    assert content_sha256_from_headers(
        {"X-Linked-Etag": '"352aecd07acc46ad96d42ae6a7ef62f8f17172d5"'}
    ) is None


def test_an_s3_multipart_etag_is_refused():
    assert content_sha256_from_headers(
        {"X-Linked-Etag": '"d41d8cd98f00b204e9800998ecf8427e-7"'}
    ) is None


def test_a_weak_validator_is_unwrapped():
    assert content_sha256_from_headers(
        {"X-Linked-Etag": 'W/"%s"' % HF_LINKED_ETAG.upper()}
    ) == HF_LINKED_ETAG


def test_no_headers_means_no_claim():
    assert content_sha256_from_headers({}) is None
    assert content_sha256_from_headers(None) is None


# --------------------------------------------------------------------------
# The name a finished download wears


def test_the_url_hash_suffix_is_replaced_by_the_content_hash():
    url_hash = hash_string("https://example.com/a")
    provisional = "model_%s.safetensors" % url_hash
    assert canonical_model_name(provisional, "a" * 64) == "model_%s.safetensors" % ("a" * 64)


def test_a_bare_name_still_gets_a_content_hash():
    assert canonical_model_name("model.safetensors", "b" * 64) == "model_%s.safetensors" % ("b" * 64)


def test_a_name_without_an_extension_survives():
    assert canonical_model_name("model", "c" * 64) == "model_%s" % ("c" * 64)


# --------------------------------------------------------------------------
# Adoption itself


def test_the_file_under_its_content_name_is_adopted():
    with tempfile.TemporaryDirectory() as d:
        sha = _write(os.path.join(d, "model_%s.safetensors" % ("0" * 64)), b"")
        payload = b"the weights" * 100
        sha = _write(os.path.join(d, "model_%s.safetensors" % hashlib.sha256(payload).hexdigest()), payload)
        got = adopt_existing_file(
            [d], canonical_model_name("model_deadbeef.safetensors", sha), sha,
            len(payload), "irrelevant.json", "model",
        )
        assert got == os.path.join(d, "model_%s.safetensors" % sha)


def test_a_file_whose_name_lies_about_its_hash_is_refused():
    # `library-asset-hash-integrity`: a data file whose name is not the sha256
    # of its content is exactly what "corrupted" means here. Adoption must not
    # serve it, and must not serve it in silence either -- it downloads.
    with tempfile.TemporaryDirectory() as d:
        payload = b"not what the name claims"
        liar = "model_%s.safetensors" % ("a" * 64)
        _write(os.path.join(d, liar), payload)
        got = adopt_existing_file(
            [d], liar, "a" * 64, len(payload), "irrelevant.json", "model",
        )
        assert got is None


def test_a_file_of_the_wrong_size_is_refused_without_hashing_it():
    with tempfile.TemporaryDirectory() as d:
        payload = b"short"
        sha = hashlib.sha256(payload).hexdigest()
        name = "model_%s.safetensors" % sha
        _write(os.path.join(d, name), payload)
        assert adopt_existing_file([d], name, sha, 999999, "x.json", "model") is None


def test_a_sidecar_pointing_at_the_bytes_is_enough():
    # Same bytes reached under a different original filename: the content name
    # does not match, but a sidecar already recorded the hash.
    with tempfile.TemporaryDirectory() as d:
        payload = b"shared weights" * 50
        sha = hashlib.sha256(payload).hexdigest()
        _write(os.path.join(d, "under-another-name.safetensors"), payload)
        with open(os.path.join(d, "%s.json" % ("e" * 64)), "w") as f:
            json.dump({"file_name": "under-another-name.safetensors", "sha256": sha}, f)
        got = adopt_existing_file(
            [d], "model_%s.safetensors" % sha, sha, len(payload), "mine.json", "model",
        )
        assert got == os.path.join(d, "under-another-name.safetensors")


def test_the_durable_dir_is_searched_when_the_download_would_go_to_the_cache():
    with tempfile.TemporaryDirectory() as cache, tempfile.TemporaryDirectory() as durable:
        payload = b"volume weights" * 40
        sha = hashlib.sha256(payload).hexdigest()
        name = "model_%s.safetensors" % sha
        _write(os.path.join(durable, name), payload)
        got = adopt_existing_file([cache, durable], name, sha, len(payload), "x.json", "model")
        assert got == os.path.join(durable, name)


def test_a_file_still_named_by_an_old_url_hash_is_adopted():
    # THE STATE EVERY REAL MACHINE IS IN. A parallel download returned before
    # the content rename, so large models sit on disk as
    # `<base>_<URL hash><ext>` with a sidecar that has no `sha256` at all.
    # Neither the content name nor any recorded hash can find them; the
    # original name and the size can.
    with tempfile.TemporaryDirectory() as d:
        payload = b"weights fetched in parallel" * 400
        sha = hashlib.sha256(payload).hexdigest()
        old_url_hash = hash_string("https://huggingface.co/x/y/resolve/main/model.safetensors")
        stale = "model_%s.safetensors" % old_url_hash
        _write(os.path.join(d, stale), payload)
        with open(os.path.join(d, "%s.json" % old_url_hash), "w") as f:
            json.dump({"url": "...", "file_name": stale, "file_size": len(payload)}, f)

        got = adopt_existing_file(
            [d], "model_%s.safetensors" % sha, sha, len(payload), "new.json", "model",
        )
        assert got == os.path.join(d, stale)


def test_same_name_and_size_but_different_bytes_is_refused():
    # Name and size only propose. The hash disposes.
    with tempfile.TemporaryDirectory() as d:
        mine = b"A" * 4096
        theirs = b"B" * 4096
        _write(os.path.join(d, "model_%s.safetensors" % ("9" * 64)), theirs)
        got = adopt_existing_file(
            [d], "model_%s.safetensors" % hashlib.sha256(mine).hexdigest(),
            hashlib.sha256(mine).hexdigest(), len(mine), "x.json", "model",
        )
        assert got is None


def test_a_different_model_of_the_same_size_is_not_offered():
    with tempfile.TemporaryDirectory() as d:
        payload = b"Z" * 2048
        sha = hashlib.sha256(payload).hexdigest()
        _write(os.path.join(d, "something_else_%s.safetensors" % ("1" * 64)), b"Y" * 2048)
        assert adopt_existing_file(
            [d], "model_%s.safetensors" % sha, sha, 2048, "x.json", "model",
        ) is None


def test_a_part_file_is_never_offered_for_adoption():
    with tempfile.TemporaryDirectory() as d:
        payload = b"half a model" * 100
        sha = hashlib.sha256(payload).hexdigest()
        _write(os.path.join(d, "model_abc.safetensors.part"), payload)
        assert adopt_existing_file(
            [d], "model_%s.safetensors" % sha, sha, len(payload), "x.json", "model",
        ) is None


def test_an_empty_store_adopts_nothing():
    with tempfile.TemporaryDirectory() as d:
        assert adopt_existing_file([d], "model_%s.bin" % ("f" * 64), "f" * 64, 10, "x.json", "m") is None


# --------------------------------------------------------------------------
# End to end: the exact shape of a8f13a50b


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
    """Answers headers the way Hugging Face does, and counts what it serves."""

    calls = []
    bytes_served = 0

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kwargs):
        _FakeSession.calls.append(url)
        sha = hashlib.sha256(self.payload).hexdigest()
        redirect = _FakeResponse({
            "X-Linked-Etag": '"%s"' % sha,
            "X-Linked-Size": str(len(self.payload)),
        })
        return _FakeResponse(
            {
                "Content-Length": str(len(self.payload)),
                "Content-Disposition": 'inline; filename="krea2_darkbrush.safetensors";',
                "ETag": '"%s"' % HF_CDN_ETAG,
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


def test_repointing_the_revision_downloads_nothing(monkeypatch):
    payload = b"z-image turbo weights" * 500
    sha = hashlib.sha256(payload).hexdigest()

    old_url = "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/loras/krea2_darkbrush.safetensors"
    new_url = "https://huggingface.co/Comfy-Org/Krea-2/resolve/e5ea8b4dd7f38f348b138eb0fe29f92c0e367e96/loras/krea2_darkbrush.safetensors"
    assert hash_string(old_url) != hash_string(new_url)

    monkeypatch.setattr(fetch, "requests", _FakeRequests(payload))
    monkeypatch.setattr(fetch, "REQUESTS_AVAILABLE", True)

    with tempfile.TemporaryDirectory() as d:
        # The state a8f13a50b left every machine in: the file sits under its
        # content name, with a sidecar named after the OLD url.
        name = "krea2_darkbrush_%s.safetensors" % sha
        _write(os.path.join(d, name), payload)
        with open(os.path.join(d, "%s.json" % hash_string(old_url)), "w") as f:
            json.dump({"url": old_url, "file_name": name, "file_size": len(payload), "sha256": sha}, f)

        _FakeSession.calls = []
        _FakeSession.bytes_served = 0
        got = download_file(url=new_url, dir=d)

        assert got == os.path.join(d, name)
        # One request, for headers. Not one byte of the model.
        assert len(_FakeSession.calls) == 1
        assert _FakeSession.bytes_served == 0

        # The new url now has its own sidecar, pointing at the same file.
        with open(os.path.join(d, "%s.json" % hash_string(new_url))) as f:
            sidecar = json.load(f)
        assert sidecar["file_name"] == name
        assert sidecar["sha256"] == sha

        # And the old one is untouched: two urls, one file.
        with open(os.path.join(d, "%s.json" % hash_string(old_url))) as f:
            assert json.load(f)["file_name"] == name


def test_adoption_writes_the_sidecar_beside_the_file_it_names(monkeypatch):
    # On a pod the download goes to the NVMe cache but the file is on the
    # volume. A sidecar left in the cache would die with the container and
    # point at nothing meanwhile.
    payload = b"volume weights" * 300
    sha = hashlib.sha256(payload).hexdigest()
    url = "https://huggingface.co/x/y/resolve/abc/model.safetensors"

    monkeypatch.setattr(fetch, "requests", _FakeRequests(payload))
    monkeypatch.setattr(fetch, "REQUESTS_AVAILABLE", True)

    with tempfile.TemporaryDirectory() as cache, tempfile.TemporaryDirectory() as durable:
        name = "krea2_darkbrush_%s.safetensors" % sha
        _write(os.path.join(durable, name), payload)

        got = download_file(url=url, dir=cache, adopt_dirs=[durable])

        assert got == os.path.join(durable, name)
        assert os.path.isfile(os.path.join(durable, "%s.json" % hash_string(url)))
        assert not os.path.exists(os.path.join(cache, "%s.json" % hash_string(url)))


def test_a_download_is_named_by_its_content(monkeypatch):
    payload = b"fresh weights" * 600
    sha = hashlib.sha256(payload).hexdigest()
    url = "https://huggingface.co/x/y/resolve/abc/krea2_darkbrush.safetensors"

    monkeypatch.setattr(fetch, "requests", _FakeRequests(payload))
    monkeypatch.setattr(fetch, "REQUESTS_AVAILABLE", True)

    with tempfile.TemporaryDirectory() as d:
        got = download_file(url=url, dir=d)
        assert os.path.basename(got) == "krea2_darkbrush_%s.safetensors" % sha
        with open(os.path.join(d, "%s.json" % hash_string(url))) as f:
            assert json.load(f)["sha256"] == sha


def test_a_completed_part_file_is_named_by_its_content_too(monkeypatch):
    # The second path that used to return before the rename: a `.part` already
    # complete because the process died between the last byte and the rename.
    payload = b"interrupted then finished" * 300
    sha = hashlib.sha256(payload).hexdigest()
    url = "https://huggingface.co/x/y/resolve/abc/model.safetensors"

    monkeypatch.setattr(fetch, "requests", _FakeRequests(payload))
    monkeypatch.setattr(fetch, "REQUESTS_AVAILABLE", True)

    with tempfile.TemporaryDirectory() as d:
        provisional = "model_%s.safetensors" % hash_string(url)
        _write(os.path.join(d, provisional + ".part"), payload)
        with open(os.path.join(d, "%s.json" % hash_string(url)), "w") as f:
            json.dump({"url": url, "file_name": provisional, "file_size": len(payload)}, f)

        _FakeSession.bytes_served = 0
        got = download_file(url=url, dir=d)

        assert _FakeSession.bytes_served == 0
        assert os.path.basename(got) == "model_%s.safetensors" % sha
        assert not os.path.exists(os.path.join(d, provisional + ".part"))


def test_a_url_serving_different_bytes_is_not_adopted(monkeypatch):
    # THE CACHE KEY IS NOT NORMALISED TO DROP THE REVISION, and this is why:
    # two revisions of one repo path may hold different bytes. Identity is the
    # content hash, so a second revision of the same path downloads.
    payload_v2 = b"revision two is different" * 200
    url_v1 = "https://huggingface.co/x/y/resolve/aaaa/model.safetensors"
    url_v2 = "https://huggingface.co/x/y/resolve/bbbb/model.safetensors"

    monkeypatch.setattr(fetch, "requests", _FakeRequests(payload_v2))
    monkeypatch.setattr(fetch, "REQUESTS_AVAILABLE", True)

    with tempfile.TemporaryDirectory() as d:
        old_payload = b"revision one" * 200
        old_sha = hashlib.sha256(old_payload).hexdigest()
        old_name = "model_%s.safetensors" % old_sha
        _write(os.path.join(d, old_name), old_payload)
        with open(os.path.join(d, "%s.json" % hash_string(url_v1)), "w") as f:
            json.dump({"url": url_v1, "file_name": old_name,
                       "file_size": len(old_payload), "sha256": old_sha}, f)

        _FakeSession.bytes_served = 0
        got = download_file(url=url_v2, dir=d)

        # Revision two is downloaded, not served from revision one's file.
        assert _FakeSession.bytes_served == len(payload_v2)
        assert got != os.path.join(d, old_name)
        with open(got, "rb") as f:
            assert f.read() == payload_v2
        # And revision one is still there, still reachable by its own url.
        assert os.path.isfile(os.path.join(d, old_name))
