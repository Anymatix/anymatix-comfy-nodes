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
    #
    # IT NOW HAS TO SAY SO. This test used to hand `download_file` nothing but a
    # full-size part file, and that was the bug it was unwittingly pinning: a
    # parallel download pre-allocates its part file, so the size it checked is
    # true from the first byte onwards. The part file is accepted here because
    # the downloader recorded its completion, which is a fact about the
    # transfer rather than about the length of a file.
    # bugs/a-parallel-download-pre-allocates-part-file-so
    payload = b"interrupted then finished" * 300
    sha = hashlib.sha256(payload).hexdigest()
    url = "https://huggingface.co/x/y/resolve/abc/model.safetensors"

    monkeypatch.setattr(fetch, "requests", _FakeRequests(payload))
    monkeypatch.setattr(fetch, "REQUESTS_AVAILABLE", True)

    with tempfile.TemporaryDirectory() as d:
        provisional = "model_%s.safetensors" % hash_string(url)
        part = os.path.join(d, provisional + ".part")
        _write(part, payload)
        fetch.mark_part_complete(part, len(payload))
        with open(os.path.join(d, "%s.json" % hash_string(url)), "w") as f:
            json.dump({"url": url, "file_name": provisional, "file_size": len(payload)}, f)

        _FakeSession.bytes_served = 0
        got = download_file(url=url, dir=d)

        assert _FakeSession.bytes_served == 0
        assert os.path.basename(got) == "model_%s.safetensors" % sha
        assert not os.path.exists(part)
        assert not os.path.exists(fetch.completion_marker_for(part))


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


# --------------------------------------------------------------------------
# Deletion, now that one file has several referrers
#
# Storing a model under the sha256 of its bytes and its sidecar under the
# sha256 of its url means several sidecars legitimately name one file. That is
# what adoption produces, and it used to be rare for large models only because
# the parallel path left them url-named. Deleting one url must not take the
# bytes another url is still using -- and must not leave them behind either.


def _store(d, url, file_name, payload, sha=None):
    """One url's sidecar, as download_file leaves it."""
    path = os.path.join(d, file_name)
    if not os.path.exists(path):
        _write(path, payload)
    sidecar = {"url": url, "file_name": file_name, "file_size": len(payload)}
    if sha:
        sidecar["sha256"] = sha
    with open(os.path.join(d, "%s.json" % hash_string(url)), "w") as f:
        json.dump(sidecar, f)
    return path


def test_deleting_one_url_leaves_the_file_the_other_adopted():
    # THE CASE THAT MATTERS. Two urls adopted onto one file; delete one and the
    # other must still load.
    from fetch import delete_files

    with tempfile.TemporaryDirectory() as d:
        payload = b"shared weights" * 400
        sha = hashlib.sha256(payload).hexdigest()
        name = "model_%s.safetensors" % sha
        url_a = "https://huggingface.co/x/y/resolve/aaaa/model.safetensors"
        url_b = "https://huggingface.co/x/y/resolve/bbbb/model.safetensors"
        _store(d, url_a, name, payload, sha)
        _store(d, url_b, name, payload, sha)

        delete_files(url_a, d)

        assert os.path.isfile(os.path.join(d, name)), "the bytes url_b still needs were deleted"
        assert not os.path.exists(os.path.join(d, "%s.json" % hash_string(url_a)))
        sidecar_b = os.path.join(d, "%s.json" % hash_string(url_b))
        assert os.path.isfile(sidecar_b)
        with open(sidecar_b) as f:
            assert json.load(f)["file_name"] == name


def test_deleting_the_last_url_takes_the_file_with_it():
    # The other half: no leak. Sweep 1 used to eat the sidecar before the
    # referrer-aware sweep could see it, so a content-named file survived every
    # deletion and sat on disk forever.
    from fetch import delete_files

    with tempfile.TemporaryDirectory() as d:
        payload = b"sole weights" * 400
        sha = hashlib.sha256(payload).hexdigest()
        name = "model_%s.safetensors" % sha
        url = "https://huggingface.co/x/y/resolve/aaaa/model.safetensors"
        _store(d, url, name, payload, sha)

        delete_files(url, d)

        assert not os.path.exists(os.path.join(d, name))
        assert not os.path.exists(os.path.join(d, "%s.json" % hash_string(url)))


def test_deleting_the_second_url_then_takes_the_file():
    from fetch import delete_files

    with tempfile.TemporaryDirectory() as d:
        payload = b"shared then sole" * 200
        sha = hashlib.sha256(payload).hexdigest()
        name = "model_%s.safetensors" % sha
        url_a = "https://huggingface.co/x/y/resolve/aaaa/model.safetensors"
        url_b = "https://huggingface.co/x/y/resolve/bbbb/model.safetensors"
        _store(d, url_a, name, payload, sha)
        _store(d, url_b, name, payload, sha)

        delete_files(url_a, d)
        assert os.path.isfile(os.path.join(d, name))
        delete_files(url_b, d)
        assert not os.path.exists(os.path.join(d, name))
        assert [x for x in os.listdir(d) if x.endswith(".safetensors")] == []


def test_a_legacy_url_named_file_another_url_adopted_is_not_deleted():
    # The sweep that matched `url_hash in filename` had no referrer check at
    # all. A file still wearing an OLD url hash is exactly what adoption reuses,
    # so deleting the url it was named after took a file in active use.
    from fetch import delete_files

    with tempfile.TemporaryDirectory() as d:
        payload = b"legacy weights" * 400
        sha = hashlib.sha256(payload).hexdigest()
        url_a = "https://huggingface.co/x/y/resolve/main/model.safetensors"
        url_b = "https://huggingface.co/x/y/resolve/bbbb/model.safetensors"
        legacy = "model_%s.safetensors" % hash_string(url_a)
        _store(d, url_a, legacy, payload)          # old sidecar, no sha256
        _store(d, url_b, legacy, payload, sha)     # adopted it

        delete_files(url_a, d)

        assert os.path.isfile(os.path.join(d, legacy)), "adopted legacy file was deleted"
        assert os.path.isfile(os.path.join(d, "%s.json" % hash_string(url_b)))


def test_an_orphan_named_by_this_url_is_still_swept():
    # Keeping what the name sweep was FOR: a file wearing this url's hash that
    # no sidecar names is this url's litter and goes.
    from fetch import delete_files

    with tempfile.TemporaryDirectory() as d:
        payload = b"litter" * 100
        url = "https://huggingface.co/x/y/resolve/main/model.safetensors"
        orphan = "model_%s.safetensors" % hash_string(url)
        _write(os.path.join(d, orphan), payload)

        delete_files(url, d)
        assert not os.path.exists(os.path.join(d, orphan))


def test_a_partial_download_goes_with_its_url():
    from fetch import delete_files

    with tempfile.TemporaryDirectory() as d:
        payload = b"complete" * 100
        sha = hashlib.sha256(payload).hexdigest()
        name = "model_%s.safetensors" % sha
        url = "https://huggingface.co/x/y/resolve/aaaa/model.safetensors"
        _store(d, url, name, payload, sha)
        _write(os.path.join(d, name + ".part"), b"half")

        delete_files(url, d)
        assert not os.path.exists(os.path.join(d, name + ".part"))


def test_another_models_file_is_never_touched():
    from fetch import delete_files

    with tempfile.TemporaryDirectory() as d:
        mine = b"mine" * 400
        theirs = b"theirs" * 400
        url_a = "https://huggingface.co/x/y/resolve/aaaa/model.safetensors"
        url_b = "https://huggingface.co/x/z/resolve/aaaa/other.safetensors"
        _store(d, url_a, "model_%s.safetensors" % hashlib.sha256(mine).hexdigest(), mine)
        other = _store(d, url_b, "other_%s.safetensors" % hashlib.sha256(theirs).hexdigest(), theirs)

        delete_files(url_a, d)
        assert os.path.isfile(other)
        assert os.path.isfile(os.path.join(d, "%s.json" % hash_string(url_b)))


def test_a_credentialled_url_deletes_the_sidecar_that_stored_its_base():
    # download_file stores the BASE url and names the sidecar after the
    # EFFECTIVE one, so neither identifies the other on its own.
    from fetch import delete_files

    with tempfile.TemporaryDirectory() as d:
        payload = b"civitai weights" * 200
        sha = hashlib.sha256(payload).hexdigest()
        name = "model_%s.safetensors" % sha
        base = "https://civitai.com/api/download/models/12345"
        effective = base + "?token=SECRET"
        _write(os.path.join(d, name), payload)
        with open(os.path.join(d, "%s.json" % hash_string(effective)), "w") as f:
            json.dump({"url": base, "file_name": name, "file_size": len(payload), "sha256": sha}, f)

        delete_files(effective, d)
        assert not os.path.exists(os.path.join(d, name))
        assert not os.path.exists(os.path.join(d, "%s.json" % hash_string(effective)))


def test_a_sidecar_pointing_at_a_gone_file_re_fetches_rather_than_lying(monkeypatch):
    # THE REVERSE HAZARD, and it is benign by design. A sidecar whose file has
    # been removed does not report the model present: download_file's size
    # check fails, adoption looks for the bytes elsewhere, and failing that it
    # downloads. It can never serve a different model, because adoption proves
    # the content hash before adopting.
    payload = b"re-fetched" * 400
    sha = hashlib.sha256(payload).hexdigest()
    url = "https://huggingface.co/x/y/resolve/aaaa/model.safetensors"

    monkeypatch.setattr(fetch, "requests", _FakeRequests(payload))
    monkeypatch.setattr(fetch, "REQUESTS_AVAILABLE", True)

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "%s.json" % hash_string(url)), "w") as f:
            json.dump({"url": url, "file_name": "model_%s.safetensors" % sha,
                       "file_size": len(payload), "sha256": sha}, f)

        _FakeSession.bytes_served = 0
        got = download_file(url=url, dir=d)

        assert _FakeSession.bytes_served == len(payload)
        assert os.path.basename(got) == "model_%s.safetensors" % sha
        with open(got, "rb") as f:
            assert f.read() == payload


def test_a_sidecar_whose_file_moved_adopts_instead_of_downloading(monkeypatch):
    # Same situation, but the bytes are still on the machine under another
    # name: it adopts them rather than paying for them twice.
    payload = b"moved not gone" * 300
    sha = hashlib.sha256(payload).hexdigest()
    url = "https://huggingface.co/x/y/resolve/aaaa/model.safetensors"

    monkeypatch.setattr(fetch, "requests", _FakeRequests(payload))
    monkeypatch.setattr(fetch, "REQUESTS_AVAILABLE", True)

    with tempfile.TemporaryDirectory() as d:
        _write(os.path.join(d, "model_%s.safetensors" % sha), payload)
        with open(os.path.join(d, "%s.json" % hash_string(url)), "w") as f:
            json.dump({"url": url, "file_name": "model_under_an_old_name.safetensors",
                       "file_size": len(payload), "sha256": sha}, f)

        _FakeSession.bytes_served = 0
        got = download_file(url=url, dir=d)

        assert _FakeSession.bytes_served == 0
        assert os.path.basename(got) == "model_%s.safetensors" % sha


# --------------------------------------------------------------------------
# The referrer rule itself
#
# serve_delete in __init__.py is the deletion path the APP actually calls
# (ComfyMachineInfo.ts posts to /anymatix/delete_resource); delete_files has no
# caller in the pack. __init__.py cannot be imported outside ComfyUI, so the
# rule it obeys lives in fetch.py and is exercised here directly -- the same
# function object the route calls, not a copy of its logic.


def test_a_surviving_sidecar_holds_the_file():
    from fetch import model_file_is_spoken_for

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "aaaa.json"), "w") as f:
            json.dump({"url": "u1", "file_name": "shared.safetensors"}, f)
        with open(os.path.join(d, "bbbb.json"), "w") as f:
            json.dump({"url": "u2", "file_name": "shared.safetensors"}, f)

        assert model_file_is_spoken_for(d, "shared.safetensors", lambda n, _d: n == "aaaa.json")


def test_the_last_sidecar_releases_the_file():
    from fetch import model_file_is_spoken_for

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "aaaa.json"), "w") as f:
            json.dump({"url": "u1", "file_name": "sole.safetensors"}, f)

        assert not model_file_is_spoken_for(d, "sole.safetensors", lambda n, _d: n == "aaaa.json")


def test_two_sidecars_dying_together_do_not_hold_each_other_up():
    # THE STANDOFF. Two sidecars storing the same base url, differing only in
    # the auth tail their names were hashed from, both matched by one delete.
    # Each used to count the other as a referrer, so the bytes were kept and
    # both sidecars removed -- a file with nothing left pointing at it.
    from fetch import model_file_is_spoken_for, sidecar_url_matches

    base = "https://civitai.com/api/download/models/12345"
    with tempfile.TemporaryDirectory() as d:
        for tail in ("OLDKEY", "NEWKEY"):
            with open(os.path.join(d, "%s.json" % hash_string(base + "?token=" + tail)), "w") as f:
                json.dump({"url": base, "file_name": "shared.safetensors"}, f)

        url = base + "?token=NEWKEY"
        doomed = lambda _n, data: sidecar_url_matches(data.get("url"), url)
        assert not model_file_is_spoken_for(d, "shared.safetensors", doomed)


def test_a_different_model_does_not_hold_the_file():
    from fetch import model_file_is_spoken_for

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "aaaa.json"), "w") as f:
            json.dump({"url": "u1", "file_name": "mine.safetensors"}, f)
        with open(os.path.join(d, "bbbb.json"), "w") as f:
            json.dump({"url": "u2", "file_name": "theirs.safetensors"}, f)

        assert not model_file_is_spoken_for(d, "mine.safetensors", lambda n, _d: n == "aaaa.json")


def test_an_unreadable_directory_refuses_to_release_the_file():
    # It cannot prove the file is free, so it does not claim it is. Refusing to
    # delete leaves a reclaimable file; deleting on a failed read does not.
    from fetch import model_file_is_spoken_for

    assert model_file_is_spoken_for("/nonexistent-dir-for-this-test", "x.safetensors", lambda *_: False)


def test_a_malformed_sidecar_is_not_a_referrer():
    from fetch import model_file_is_spoken_for

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "broken.json"), "w") as f:
            f.write("{not json")
        with open(os.path.join(d, "aaaa.json"), "w") as f:
            json.dump({"url": "u1", "file_name": "sole.safetensors"}, f)

        assert not model_file_is_spoken_for(d, "sole.safetensors", lambda n, _d: n == "aaaa.json")
