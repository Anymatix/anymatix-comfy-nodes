#!/usr/bin/env python3
"""
A STOPPED PARALLEL DOWNLOAD RESUMES FROM THE LAST BYTE EACH SEGMENT WROTE.

bugs/a-paused-download-re-fetches-up-256-mib (app repo). Each segment wrote its
progress to `<part>.segments` only every `SEGMENT_JOURNAL_STRIDE` (32 MiB), and
a Stop cancels the segment tasks mid-stream with no last write, so a resume
fetched again up to 32 MiB of every segment: measured 426M -> 302M of 7.26G
and 285M -> 134M of 9.35G after a pause in the app.

The Stop here is the real one: a prefetch's own interrupt check
(`own_interrupt_check`) raising `PrefetchCancelled`, which `fetch_parallel`'s
node thread turns into `task.cancel()` on the transfer, exactly as a Pause does.
Segments are 8 MiB, under one stride, so before the fix the journal said
NOTHING had landed and the resume fetched every byte again.

    python3 -m pytest tests/test_stopped_parallel_download_journals_every_byte.py -q
"""
import hashlib
import http.server
import json
import os
import socketserver
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import fetch

pytest.importorskip("aiohttp")
pytest.importorskip("aiofiles")

MIB = 1024 * 1024
SIZE = 32 * MIB
CONNECTIONS = 4  # -> four 8 MiB segments (`segment_step`)


def _slow_server(blob: bytes, chunk: int = 64 * 1024, pause: float = 0.02):
    """A Range-capable server at ~3 MB/s per connection, so a Stop lands
    mid-segment. `gets` records every range it was asked for."""
    size = len(blob)
    state = {"gets": []}

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            header = self.headers.get("Range")
            if header:
                a, _, b = header.replace("bytes=", "").partition("-")
                start, end = int(a), (int(b) if b else size - 1)
            else:
                start, end = 0, size - 1
            state["gets"].append((start, end))
            view = memoryview(blob)[start:end + 1]
            self.send_response(206 if header else 200)
            if header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(len(view)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            try:
                for offset in range(0, len(view), chunk):
                    self.wfile.write(view[offset:offset + chunk])
                    if len(view) > 1:
                        time.sleep(pause)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client stopped, which is the point

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], state


def _prefix_on_disk(part: str, blob: bytes, start: int, end: int) -> int:
    """How many bytes from `start` really are the served file's bytes."""
    with open(part, "rb") as f:
        f.seek(start)
        got = f.read(end - start + 1)
    want = blob[start:end + 1]
    for i, (x, y) in enumerate(zip(got, want)):
        if x != y:
            return i
    return len(got)


def test_a_stop_journals_each_segment_s_last_byte_and_the_resume_fetches_only_the_rest():
    blob = os.urandom(SIZE)
    server, port, served = _slow_server(blob)
    url = f"http://127.0.0.1:{port}/weights.bin"
    part = os.path.join(tempfile.mkdtemp(), "weights.bin.part")
    stop = threading.Event()

    def progress(done, total):
        if done >= total // 4:
            stop.set()

    def prefetch_check():
        if stop.is_set():
            raise fetch.PrefetchCancelled("paused")

    try:
        with fetch.own_interrupt_check(prefetch_check):
            with pytest.raises(fetch.PrefetchCancelled):
                fetch.fetch_parallel(url, part, progress, max_connections=CONNECTIONS)

        journal = fetch.read_segment_journal(part, SIZE)
        assert journal is not None, "the Stop left no journal a resume can trust"
        segments = fetch.segment_layout(SIZE, journal["segment_size"])
        assert len(segments) == CONNECTIONS
        done = journal["done"]

        on_disk_total = 0
        for sid, start, end in segments:
            on_disk = _prefix_on_disk(part, blob, start, end)
            claimed = done.get(sid, 0)
            on_disk_total += on_disk
            # Never a byte the disk does not hold...
            assert claimed <= on_disk, \
                f"segment {sid}: journal claims {claimed} bytes, {on_disk} are on disk"
            # ...and never a stride's worth fewer than it does.
            assert on_disk - claimed < MIB, \
                f"segment {sid}: {on_disk - claimed} bytes on disk the journal forgot"
        assert 0 < on_disk_total < SIZE, f"the Stop did not land mid-download ({on_disk_total} of {SIZE})"
        assert all(done.get(sid, 0) > 0 for sid, _, _ in segments), \
            f"a segment that had started journalled nothing: {done}"

        before = len(served["gets"])
        assert fetch.fetch_parallel(url, part, max_connections=CONNECTIONS) is True

        resumed = [(s, e) for s, e in served["gets"][before:] if (s, e) != (0, 0)]
        starts = {start + done.get(sid, 0) for sid, start, end in segments if done.get(sid, 0) <= end - start}
        assert {s for s, _ in resumed} == starts, \
            f"the resume asked for {sorted(resumed)}, the journal said {sorted(starts)}"
        refetched = sum(e - s + 1 for s, e in resumed) - (SIZE - on_disk_total)
        assert 0 <= refetched < CONNECTIONS * MIB, f"{refetched} bytes fetched twice"

        with open(part, "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == hashlib.sha256(blob).hexdigest()
        assert fetch.part_completion_is_recorded(part, SIZE)
        assert not os.path.exists(fetch.segment_journal_for(part))
        print(f"\n[proof] on disk at the Stop: {on_disk_total} B; journalled: {sum(done.values())} B; "
              f"fetched twice on resume: {refetched} B over {CONNECTIONS} segments")
    finally:
        server.shutdown()


def test_the_journal_never_claims_bytes_a_flush_did_not_land(monkeypatch):
    """A flush that fails on the way out journals nothing for that segment:
    the journal under-claims, it never over-claims."""
    blob = os.urandom(SIZE)
    server, port, _ = _slow_server(blob)
    url = f"http://127.0.0.1:{port}/weights.bin"
    part = os.path.join(tempfile.mkdtemp(), "weights.bin.part")
    stop = threading.Event()
    real_landed = fetch.SegmentJournal.landed
    calls_after_stop = []

    def landed(self, sid, offset):
        if stop.is_set():
            calls_after_stop.append((sid, offset))
        return real_landed(self, sid, offset)

    monkeypatch.setattr(fetch.SegmentJournal, "landed", landed)

    import aiofiles.threadpool.binary as binary

    real_flush = binary.AsyncBufferedIOBase.flush

    async def failing_flush(self):
        if stop.is_set():
            raise OSError(5, "Input/output error")
        return await real_flush(self)

    monkeypatch.setattr(binary.AsyncBufferedIOBase, "flush", failing_flush)

    def progress(done, total):
        if done >= total // 4:
            stop.set()

    def prefetch_check():
        if stop.is_set():
            raise fetch.PrefetchCancelled("paused")

    try:
        with fetch.own_interrupt_check(prefetch_check):
            with pytest.raises(fetch.PrefetchCancelled):
                fetch.fetch_parallel(url, part, progress, max_connections=CONNECTIONS)
        assert calls_after_stop == [], f"journalled after a failed flush: {calls_after_stop}"
    finally:
        server.shutdown()
