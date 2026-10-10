#!/usr/bin/env python3
"""
A PARALLEL DOWNLOAD KILLED MID-WAY RESUMES FROM WHAT LANDED, NOT FROM ZERO.

bugs/a-dropped-ssh-link-remote-machine-kills (app repo). On pc-ciancia,
2026-10-06, the remote's dead-man switch ended ComfyUI during an 11.9 GB
parallel download; the next boot hashed the holed, pre-allocated part file,
found it wrong, and fetched `0.00/11.9G` again at ~12 MB/s.

The kill is reproduced, not described: the process is gone, so no Python in it
ran. What it leaves is a full-length part file whose segments got partway, and
the journal `SegmentJournal` wrote as they went.

    python3 -m pytest tests/test_killed_parallel_download_resumes.py -q
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import fetch
from test_parallel_download import serve

pytest.importorskip("aiohttp")
pytest.importorskip("aiofiles")


def _killed_at_half(part: str, blob: bytes, segment_size: int) -> int:
    """The part file and journal a SIGKILL leaves when every segment is half done."""
    size = len(blob)
    done = {}
    with open(part, "wb") as f:
        f.truncate(size)
        sid = 0
        for start in range(0, size, segment_size):
            end = min(start + segment_size, size)
            half = (end - start) // 2
            f.seek(start)
            f.write(blob[start:start + half])
            done[str(sid)] = half
            sid += 1
    with open(fetch.segment_journal_for(part), "w") as j:
        json.dump({"size": size, "segment_size": segment_size, "done": done}, j)
    return sum(done.values())


def test_a_killed_parallel_download_fetches_only_what_is_missing():
    blob = os.urandom(1 << 20) * 8
    server, port, served = serve(blob)
    part = os.path.join(tempfile.mkdtemp(), "weights.bin.part")
    try:
        landed = _killed_at_half(part, blob, 2 << 20)
        assert fetch.read_segment_journal(part, len(blob)) is not None
        assert fetch.fetch_parallel(f"http://127.0.0.1:{port}/weights.bin", part) is True
        with open(part, "rb") as f:
            assert f.read() == blob, "the resumed file is not the served file"
        assert fetch.part_completion_is_recorded(part, len(blob))
        assert not os.path.exists(fetch.segment_journal_for(part)), "a finished download keeps no journal"
        # The one-byte range probe (`probe_range`) is not a segment.
        segments = [(s, e) for s, e in served["gets"] if (s, e) != (0, 0)]
        fetched = sum(e - s + 1 for s, e in segments)
        assert fetched == len(blob) - landed, f"fetched {fetched} bytes, {len(blob) - landed} were missing"
        assert all(s > 0 for s, e in segments), "a segment started again from byte zero"
    finally:
        server.shutdown()


def test_a_fresh_parallel_download_leaves_a_journal_only_while_it_runs():
    blob = os.urandom(1 << 20) * 6
    server, port, _ = serve(blob)
    part = os.path.join(tempfile.mkdtemp(), "weights.bin.part")
    try:
        assert fetch.fetch_parallel(f"http://127.0.0.1:{port}/weights.bin", part) is True
        assert not os.path.exists(fetch.segment_journal_for(part))
    finally:
        server.shutdown()


def test_a_journal_for_another_size_or_a_short_file_is_not_trusted():
    part = os.path.join(tempfile.mkdtemp(), "weights.bin.part")
    with open(part, "wb") as f:
        f.truncate(1000)
    with open(fetch.segment_journal_for(part), "w") as j:
        json.dump({"size": 2000, "segment_size": 500, "done": {"0": 10}}, j)
    assert fetch.read_segment_journal(part, 2000) is None
    assert fetch.read_segment_journal(part, 1000) is None


