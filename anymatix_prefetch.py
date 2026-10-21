"""A MODEL DOWNLOAD THAT DOES NOT HOLD THE QUEUE.

Vincenzo, 2026-10-20: "Yes, build the OS notification and make a first-run
download not hold the queue, for 1.0." (app tracker
`features/a-first-run-model-download-does-not-hold`).

The fetcher is a node (`AnymatixFetcher`), so a card whose models are missing
used to download them INSIDE its prompt, and ComfyUI runs one prompt at a
time: a 20 GB first download held every later run, including runs whose models
were already here. The download is not GPU work, so it does not need the
prompt slot. The app now asks for a card's missing models HERE first, on a
thread of its own, and queues the card's prompt only when they have landed;
the fetcher node in that prompt then finds the file and returns at once.

What this module owns, and nothing else:

  * ONE DOWNLOAD PER FILE. Two cards that need the same file, or two windows,
    join the one prefetch already running for it (`start` answers `joined`).
    A fetcher node that runs while a prefetch of its file is going waits for
    it instead of writing into the same part file (`wait_for_key`).
  * ITS OWN STOP. `cancel` ends one prefetch and touches no prompt; a Stop
    on the card that is generating ends that generation and no download.
    `fetch.own_interrupt_check` is how. The part file and its journal stay, so
    the next request resumes from the byte reached: that is the pause.
  * WHAT HAPPENED. `status` answers for a prefetch that finished, so a client
    that missed the last websocket message still learns how it ended.

The download itself is `AnymatixFetcher.download_model`, unchanged: this
module only runs it somewhere else and routes its progress by the prefetch's
own id.
"""

import threading
import time
import uuid
from typing import Callable, Dict, Optional

try:
    from .fetch import PrefetchCancelled, own_interrupt_check
except ImportError:
    from fetch import PrefetchCancelled, own_interrupt_check


class Prefetch:
    def __init__(self, prefetch_id: str, key: str, url: dict):
        self.id = prefetch_id
        self.key = key
        self.url = url
        self.state = "running"  # running | done | failed | cancelled
        self.error: Optional[str] = None
        self.cancel_requested = threading.Event()
        self.finished = threading.Event()
        self.ended_at: Optional[float] = None

    def check(self) -> None:
        if self.cancel_requested.is_set():
            raise PrefetchCancelled(f"prefetch {self.id} stopped")

    def describe(self) -> dict:
        out = {"id": self.id, "state": self.state}
        if self.error is not None:
            out["error"] = self.error
        return out


_lock = threading.Lock()
_running_by_key: Dict[str, Prefetch] = {}
_by_id: Dict[str, Prefetch] = {}
_by_thread: Dict[int, Prefetch] = {}

# How long a finished prefetch can still be asked about. Long enough for a
# client that lost its socket to poll once; short enough that a day of runs
# does not grow the table without end.
FINISHED_KEPT_SECONDS = 600


def prefetch_key(url: dict, strip: Callable[[str], str] = lambda u: u) -> str:
    """Which file this is: its type and its url, token removed, case folded."""
    raw = str((url or {}).get("url") or "")
    return f"{(url or {}).get('type')}|{strip(raw).strip().lower()}"


def current() -> Optional[Prefetch]:
    """The prefetch the calling thread is downloading for, if any."""
    return _by_thread.get(threading.get_ident())


def _forget_old(now: float) -> None:
    for pid in [p.id for p in _by_id.values() if p.ended_at is not None and now - p.ended_at > FINISHED_KEPT_SECONDS]:
        _by_id.pop(pid, None)


def start(
    url: dict,
    fetch: Callable[[dict], object],
    on_end: Callable[[Prefetch], None],
    key: str,
    prefetch_id: Optional[str] = None,
):
    """Start downloading `url`, or join the download of it already running.

    Returns `(prefetch, joined)`. `fetch` is called with `url` on a thread of
    its own; `on_end` is called once on that thread when it ends, however it
    ends.
    """
    with _lock:
        _forget_old(time.time())
        existing = _running_by_key.get(key)
        if existing is not None:
            return existing, True
        p = Prefetch(prefetch_id or f"prefetch-{uuid.uuid4()}", key, url)
        _running_by_key[key] = p
        _by_id[p.id] = p

    def run() -> None:
        ident = threading.get_ident()
        _by_thread[ident] = p
        try:
            with own_interrupt_check(p.check):
                fetch(url)
            p.state = "done"
        except PrefetchCancelled:
            p.state = "cancelled"
        except BaseException as e:  # the thread must always say how it ended
            p.state = "failed"
            p.error = str(e) or type(e).__name__
        finally:
            _by_thread.pop(ident, None)
            with _lock:
                if _running_by_key.get(key) is p:
                    _running_by_key.pop(key, None)
                p.ended_at = time.time()
            p.finished.set()
            on_end(p)

    threading.Thread(target=run, name=f"anymatix-prefetch-{p.id[-8:]}", daemon=True).start()
    return p, False


def cancel(prefetch_id: str) -> bool:
    """Ask one prefetch to stop. False when there is no such running prefetch."""
    p = _by_id.get(prefetch_id)
    if p is None or p.state != "running":
        return False
    p.cancel_requested.set()
    return True


def status(prefetch_id: str) -> Optional[dict]:
    p = _by_id.get(prefetch_id)
    return p.describe() if p is not None else None


def wait_for_key(key: str, check: Callable[[], None], poll_seconds: float = 0.5) -> bool:
    """Block while a prefetch of this file is running, asking `check` between
    waits so the caller's own Stop still reaches it. True if there was one."""
    p = _running_by_key.get(key)
    if p is None:
        return False
    while not p.finished.wait(poll_seconds):
        check()
    return True
