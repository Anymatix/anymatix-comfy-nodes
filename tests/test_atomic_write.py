"""
A result URL must never point at a half-written file.

The claim under test is narrow: whatever `anymatix_atomic_write.atomic_output`
wraps, either the final file appears complete and byte-identical to what was
written, or it does not appear at all — never partial, and never a stray temp
file left behind for a listing/glob to pick up.

The module under test imports neither `comfy` nor `folder_paths`, so it is
loaded directly, the same way `test_output_formats.py` loads its module.
"""

import glob
import importlib.util
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "anymatix_atomic_write", os.path.join(_HERE, "..", "anymatix_atomic_write.py")
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

temp_path_for = _MODULE.temp_path_for
publish = _MODULE.publish
cleanup_temp = _MODULE.cleanup_temp
atomic_output = _MODULE.atomic_output


def _listing(directory):
    return sorted(os.listdir(directory))


# -------------------------------------------------------- temp_path_for ----


def test_temp_name_is_hidden_and_excluded_from_a_prefix_glob(tmp_path):
    final = str(tmp_path / "ComfyUI_0001.png")
    temp = temp_path_for(final)

    assert os.path.dirname(temp) == os.path.dirname(final)
    assert os.path.basename(temp).startswith(".")
    # The counter-scan pattern `Anymatix_Image_Save` uses is
    # `{prefix}{delimiter}(\d+)`, anchored with `re.match` at the start of the
    # basename. A leading dot can never match that, or a plain `*.png` glob.
    assert not glob.fnmatch.fnmatch(os.path.basename(temp), "ComfyUI_*.png")


def test_two_temp_names_for_the_same_final_path_never_collide():
    final = "/some/dir/result.json"
    assert temp_path_for(final) != temp_path_for(final)


# ------------------------------------------------------------- publish -----


def test_publish_moves_bytes_unchanged_into_the_final_path(tmp_path):
    final = str(tmp_path / "data.json")
    temp = temp_path_for(final)
    payload = b'{"count": 3}'
    with open(temp, "wb") as f:
        f.write(payload)

    publish(temp, final)

    assert not os.path.exists(temp)
    assert os.path.exists(final)
    with open(final, "rb") as f:
        assert f.read() == payload


def test_final_path_never_exists_until_publish_runs(tmp_path):
    final = str(tmp_path / "video.mp4")
    temp = temp_path_for(final)
    with open(temp, "wb") as f:
        f.write(b"partial-bytes-mid-encode")

    # A reader polling `final` by URL before publish() must see nothing.
    assert not os.path.exists(final)
    assert _listing(str(tmp_path)) == [os.path.basename(temp)]

    publish(temp, final)
    assert _listing(str(tmp_path)) == ["video.mp4"]


# -------------------------------------------------------- cleanup_temp -----


def test_cleanup_removes_a_leftover_temp_file(tmp_path):
    temp = str(tmp_path / ".x.tmp-1-deadbeef")
    with open(temp, "wb") as f:
        f.write(b"never finished")

    cleanup_temp(temp)

    assert not os.path.exists(temp)


def test_cleanup_of_a_missing_temp_file_does_not_raise(tmp_path):
    cleanup_temp(str(tmp_path / ".nothing-here.tmp"))


# ------------------------------------------------------------------------- #
# atomic_output — the context manager every saver in this pack wraps its
# final write in.
# ------------------------------------------------------------------------- #


def test_success_leaves_final_file_identical_to_what_was_written(tmp_path):
    final = str(tmp_path / "prefix_0001.png")
    payload = os.urandom(4096)

    with atomic_output(final, "wb") as (f, temp):
        f.write(payload)

    assert _listing(str(tmp_path)) == ["prefix_0001.png"]
    with open(final, "rb") as f:
        assert f.read() == payload


def test_success_writes_text_mode_identically(tmp_path):
    final = str(tmp_path / "data.json")
    text = '{"count": 7}'

    with atomic_output(final, "w") as (f, temp):
        f.write(text)

    with open(final, "r") as f:
        assert f.read() == text


def test_interrupted_write_leaves_no_final_file_and_no_temp_file(tmp_path):
    final = str(tmp_path / "prefix_0001.png")

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with atomic_output(final, "wb") as (f, temp):
            f.write(b"only some of the bytes")
            raise Boom("encoder died mid-frame")

    # Neither the final path nor the temp file survive a failed write — a
    # directory listing after a crash mid-write must not pick up either one.
    assert _listing(str(tmp_path)) == []


def test_interrupted_write_does_not_clobber_a_pre_existing_final_file(tmp_path):
    """
    A re-run that fails must not destroy the previous, still-good result —
    `os.replace()` only ever runs on a clean exit, so a failed second attempt
    leaves the first file exactly as it was.
    """
    final = str(tmp_path / "prefix_0001.png")
    original = b"first successful write"
    with open(final, "wb") as f:
        f.write(original)

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with atomic_output(final, "wb") as (f, temp):
            f.write(b"a second, doomed attempt")
            raise Boom("second write failed")

    assert _listing(str(tmp_path)) == ["prefix_0001.png"]
    with open(final, "rb") as f:
        assert f.read() == original


def test_success_never_exposes_the_temp_name_alongside_the_final_one(tmp_path):
    """
    A poll that lists the directory between the write and the replace must
    only ever see the hidden temp name, never a final name with 0 bytes.
    """
    final = str(tmp_path / "prefix_0001.png")

    with atomic_output(final, "wb") as (f, temp):
        f.write(b"some bytes")
        # Mid-write: temp file exists, final file does not.
        assert os.path.basename(temp) in _listing(str(tmp_path))
        assert not os.path.exists(final)

    # Post-write: only the final file remains.
    assert _listing(str(tmp_path)) == ["prefix_0001.png"]
