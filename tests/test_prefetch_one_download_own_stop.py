"""A prefetch downloads each file once, and its Stop is its own.

App tracker `features/a-first-run-model-download-does-not-hold`: a card's
missing models are downloaded outside the prompt queue (`anymatix_prefetch.py`)
so a card whose models are present can run meanwhile. Three things must hold,
and each is a way the separation could quietly break:

  * two cards needing one file start ONE download (the second joins);
  * a prefetch's Stop ends the prefetch and never reads ComfyUI's interrupt
    flag, which is one per process and cleared by reading it: a prefetch that
    asked it would steal the Stop of the card that is generating;
  * a segment worker, a pool thread, asks the check of the thread that owns
    the download, not ComfyUI's.
"""

import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fetch
import anymatix_prefetch


def _ended():
    seen = []
    done = threading.Event()

    def on_end(p):
        seen.append(p.state)
        done.set()

    return seen, done, on_end


def test_two_requests_for_one_file_download_it_once():
    release = threading.Event()
    calls = []

    def slow_fetch(url):
        calls.append(url)
        release.wait(5)

    seen, done, on_end = _ended()
    url = {"url": "https://example.test/a.safetensors", "type": "checkpoint"}
    key = anymatix_prefetch.prefetch_key(url)
    first, joined_first = anymatix_prefetch.start(url, slow_fetch, on_end, key)
    second, joined_second = anymatix_prefetch.start(dict(url), slow_fetch, on_end, key)
    assert joined_first is False
    assert joined_second is True
    assert second is first
    release.set()
    assert done.wait(5)
    assert calls == [url]
    assert seen == ["done"]
    assert anymatix_prefetch.status(first.id)["state"] == "done"


def test_a_prefetch_stop_never_reads_comfyui_flag(monkeypatch):
    comfy_reads = []
    monkeypatch.setattr(fetch, "_comfy_check_interrupted", lambda: comfy_reads.append(threading.get_ident()))
    started = threading.Event()

    def looping_fetch(url):
        started.set()
        while True:
            fetch.check_interrupted()
            threading.Event().wait(0.01)

    seen, done, on_end = _ended()
    url = {"url": "https://example.test/b.safetensors", "type": "lora"}
    p, _ = anymatix_prefetch.start(url, looping_fetch, on_end, anymatix_prefetch.prefetch_key(url))
    assert started.wait(5)
    assert anymatix_prefetch.cancel(p.id) is True
    assert done.wait(5)
    assert seen == ["cancelled"]
    assert comfy_reads == []
    # Outside a prefetch the node's check is ComfyUI's, unchanged.
    fetch.check_interrupted()
    assert comfy_reads == [threading.get_ident()]


def test_a_cancelled_prefetch_raises_what_is_interrupt_recognises():
    assert fetch.is_interrupt(fetch.PrefetchCancelled("x"))


def test_segment_workers_ask_the_owning_thread_check(monkeypatch):
    monkeypatch.setattr(fetch, "_comfy_check_interrupted", lambda: None)
    mine = lambda: None
    with fetch.own_interrupt_check(mine):
        downloader = fetch.SegmentDownloader("https://example.test/c", "/tmp/never-written", 4 * 1024 * 1024)
    assert downloader.check_interrupted is mine
    outside = fetch.SegmentDownloader("https://example.test/c", "/tmp/never-written", 4 * 1024 * 1024)
    assert outside.check_interrupted is fetch._comfy_check_interrupted


def test_a_node_waits_for_a_running_prefetch_of_its_file():
    release = threading.Event()
    seen, done, on_end = _ended()
    url = {"url": "https://example.test/d.safetensors", "type": "vae"}
    key = anymatix_prefetch.prefetch_key(url)
    anymatix_prefetch.start(url, lambda u: release.wait(5), on_end, key)
    threading.Timer(0.2, release.set).start()
    assert anymatix_prefetch.wait_for_key(key, lambda: None, poll_seconds=0.05) is True
    assert done.wait(5)
    assert anymatix_prefetch.wait_for_key(key, lambda: None) is False
