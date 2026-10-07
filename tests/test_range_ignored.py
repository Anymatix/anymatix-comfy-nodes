#!/usr/bin/env python3
"""
A SERVER THAT IGNORES `Range` MUST NOT GROW THE PART FILE.

`bugs/a-parallel-segment-download-accepts-http-200` (anymatix repo): on
fmt-5000, 2026-10-07, a Civitai LTX 2.3 VAE part file grew to 2,722,984,832 of
its 1,452,258,578 bytes, because every segment that got a `200` wrote the whole
body from its own offset. Here a local server answers every range request with
`200` and the full file, as that CDN did.

Run: `python -m pytest tests/test_range_ignored.py -q` from the repository root.
"""
import asyncio
import http.server
import os
import socketserver
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import fetch

aiohttp = pytest.importorskip("aiohttp")
pytest.importorskip("aiofiles")


def serve_ignoring_range(blob, content_range=None):
    """Every GET gets `200` and the whole blob, Range or not — or, with
    `content_range`, a `206` whose Content-Range lies about where it starts."""
    size = len(blob)

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            if content_range:
                self.send_response(206)
                self.send_header("Content-Range", content_range)
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            self.wfile.write(blob)

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def _segment(url, part, start, end):
    async def go():
        async with aiohttp.ClientSession() as session:
            return await fetch.fetch_async_segment(session, url, start, end, 1, part_path=part)
    return asyncio.run(go())


@pytest.mark.parametrize("content_range", [None, "bytes 0-1048575/4194304"])
def test_a_segment_refuses_an_answer_that_is_not_its_range(content_range):
    blob = os.urandom(4 << 20)
    server, port = serve_ignoring_range(blob, content_range)
    part = os.path.join(tempfile.mkdtemp(), "weights.bin.part")
    try:
        with open(part, "wb") as f:
            f.truncate(len(blob))
        with pytest.raises(fetch.RangeNotHonoured):
            _segment(f"http://127.0.0.1:{port}/w.bin", part, 2 << 20, (3 << 20) - 1)
        assert os.path.getsize(part) == len(blob)
    finally:
        server.shutdown()


def test_the_range_check_itself():
    fetch.check_range_response(206, "bytes 100-199/1000", 100, 199)
    fetch.check_range_response(206, None, 100, 199)
    for status, cr in [(200, None), (200, "bytes 100-199/1000"), (206, "bytes 0-99/1000"), (206, "items 100-199/1000"), (416, None)]:
        with pytest.raises(fetch.RangeNotHonoured):
            fetch.check_range_response(status, cr, 100, 199)
