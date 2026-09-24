#!/usr/bin/env python3
"""
`IS_CHANGED` NO LONGER SERVES A TRUNCATED WEIGHT AS WHOLE.

bugs/is-changed-serves-truncated-weight-as-whole: the check asked only
`os.path.exists(file_path)`, so a mirror interrupted after the sidecar was
written, a disk that filled mid-copy, or an adoption pointing at a file
something later truncated all answered "unchanged" and were handed to the
sampler as though whole -- the download that would have resumed it never ran.

`_sidecar_verified_size_matches` is the fix: a `stat` against the sidecar's
`verified_size`, the field set only once completeness was actually
established (either by the length the server declared and `download_file`
accepted on its own since 2026-09-23, or, when the server stated none, by the
hash step that measured the file it wrote). This test reproduces the bug's
own gesture -- truncate a downloaded weight to half its length, leaving its
sidecar in place -- against the helper directly, and confirms a sidecar
written before this field existed is not treated as a mismatch (no
regression against every model already on disk), and that the 2026-09-23
size-skip's own claim (`verified_size == file_size`, no hash) still reads as
whole.

The module under test imports `comfy`, so the pure helper is lifted from the
shipped source by `ast`, as every other unit test in this file does for
`anymatix_checkpoint_fetcher.py`.

    python3 -m pytest tests/test_is_changed_truncated_weight.py -q
"""
import ast
import os
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(os.path.dirname(_HERE), "anymatix_checkpoint_fetcher.py")


def _load_under_test():
    with open(_SOURCE, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_sidecar_verified_size_matches"
    ]
    assert len(wanted) == 1, f"expected _sidecar_verified_size_matches in {_SOURCE}"
    namespace = {"os": os}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), _SOURCE, "exec"), namespace)
    return namespace["_sidecar_verified_size_matches"]


matches = _load_under_test()


def _weight(tmp_path, nbytes):
    path = os.path.join(tmp_path, "model.safetensors")
    with open(path, "wb") as f:
        f.write(b"\0" * nbytes)
    return path


def test_the_bug_s_own_gesture_a_weight_truncated_to_half_length():
    with tempfile.TemporaryDirectory() as tmp:
        path = _weight(tmp, 1000)
        sidecar = {"file_name": "model.safetensors", "file_size": 1000, "verified_size": 1000}
        assert matches(path, sidecar) is True  # whole, as downloaded

        # Interrupted mirror / filled disk / truncated adoption: the sidecar
        # still claims 1000, the bytes on disk are half that.
        with open(path, "wb") as f:
            f.write(b"\0" * 500)
        assert matches(path, sidecar) is False


def test_a_sidecar_with_no_verified_size_makes_no_claim_to_check():
    # Written before this field existed, or a deduplication pointer at a file
    # another url finished -- neither regresses against models already on disk.
    with tempfile.TemporaryDirectory() as tmp:
        path = _weight(tmp, 42)
        assert matches(path, {"file_name": "model.safetensors"}) is True


def test_the_2026_09_23_size_skip_still_reads_as_whole():
    # download_file's own claim for a server-declared length it accepted
    # without hashing: verified_size == file_size, no sha256 at all.
    with tempfile.TemporaryDirectory() as tmp:
        path = _weight(tmp, 2048)
        sidecar = {
            "file_name": "model.safetensors",
            "file_size": 2048,
            "verification": "size",
            "verified_size": 2048,
        }
        assert matches(path, sidecar) is True


def test_a_missing_file_is_not_this_helper_s_job():
    # IS_CHANGED already forces a re-download when the file is gone; this
    # helper only answers the size question for a file that exists.
    assert matches("/nonexistent/path/model.safetensors", {"verified_size": 10}) is False
