from pathlib import Path
try:
    from .expunge import delete_file_and_cleanup_dir
except ImportError:
    # Fallback for testing or standalone execution
    def delete_file_and_cleanup_dir(file_path, base_dir):
        if os.path.exists(file_path):
            os.remove(file_path)
            # Try to remove parent directory if empty
            try:
                parent = file_path.parent if hasattr(file_path, 'parent') else Path(file_path).parent
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
            except:
                pass

# Import ComfyUI's interrupt checking if available
try:
    import comfy.model_management
    def check_interrupted():
        comfy.model_management.throw_exception_if_processing_interrupted()
except ImportError:
    # Fallback for standalone execution
    def check_interrupted():
        pass

import hashlib
import json
import os
import re
import threading
import time
import math
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterator, Optional, List, Tuple

# Optional high-performance dependencies - graceful fallback if not available
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

try:
    from requests import Session
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback tqdm implementation
    class tqdm:
        def __init__(self, total=None, initial=0):
            self.total = total
            self.n = initial
        def update(self, n=1):
            self.n += n
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode


# A STALLED DOWNLOAD IS DETECTED, RETRIED FROM WHERE IT STOPPED, AND ONLY THEN
# REPORTED — naming the file. `bugs/a-local-run-stalled-model-download-given`
# (app repo): a parallel download with no read timeout sat at 591M of 1.45G for
# 28 minutes on sleipnir, with no error, until the app gave the run up.
#
# DOWNLOAD_STALL_SECONDS  how long a connection may deliver no byte at all
#                         before it is treated as dead. A slow link still
#                         delivers bytes; only silence trips it.
# DOWNLOAD_ATTEMPTS       how many times one segment, or the single stream, is
#                         tried in all, each resuming at the byte it reached.
# DOWNLOAD_RETRY_PAUSES   seconds between attempts, the last repeated.
DOWNLOAD_STALL_SECONDS = 60
DOWNLOAD_CONNECT_SECONDS = 30
DOWNLOAD_ATTEMPTS = 5
DOWNLOAD_RETRY_PAUSES = (2, 5, 10, 20)

# RANGE_WARMUP_PAUSES     seconds before asking AGAIN for a range that was
#                         answered with the whole file (`200`), one per retry.
#
# A `200` to a range request is not always a server that does not do ranges.
# Measured 2026-10-10 against Civitai's live API, no key, public files
# (`bugs/a-resumed-civitai-download-re-fetches-whole-file`, app repo): the
# download redirects to a signed Backblaze B2 URL behind Cloudflare, and on a
# file nobody has fetched lately B2 answers the first one or two range requests
# with `200` and every later one with `206` — 40 cold files of 40, one request a
# second. Treating the first answer as final failed every segment of the
# sleipnir resume and sent a 1.45 GB VAE back to byte 0.
RANGE_WARMUP_PAUSES = (1, 2, 3, 5, 8)


class DownloadStalled(Exception):
    """A transfer that kept failing after every attempt it was allowed.

    `reason` is the short form for the person (*no data for 60s, 5 attempts*);
    the message adds where it stopped, for the log."""

    def __init__(self, message: str, reason: Optional[str] = None):
        super().__init__(message)
        self.reason = reason or message


def stall_in(e: Optional[BaseException]) -> Optional["DownloadStalled"]:
    """The `DownloadStalled` at the root of a wrapped download error, if any."""
    seen = 0
    while e is not None and seen < 10:
        if isinstance(e, DownloadStalled):
            return e
        e = e.__cause__ or e.__context__
        seen += 1
    return None


def stalled_download_message(file_name, stall: "DownloadStalled") -> str:
    """What a run that ran out of attempts says, naming the file."""
    return (f"Download of {file_name} stopped: {stall.reason}. "
            f"What arrived is kept; run again to resume.")


class DownloadTransportError(Exception):
    """One single-stream attempt lost its connection or went silent."""


def retry_pause_seconds(attempt: int) -> float:
    if not DOWNLOAD_RETRY_PAUSES:
        return 0
    return DOWNLOAD_RETRY_PAUSES[min(attempt, len(DOWNLOAD_RETRY_PAUSES) - 1)]


def is_interrupt(e: BaseException) -> bool:
    """ComfyUI's Stop, which must travel up untouched: never retried, never
    wrapped into a download error the caller would fall back from."""
    return "InterruptProcessingException" in type(e).__name__


def describe_transport_error(e: BaseException) -> str:
    """What failed, in the words the error message carries. A read timeout is
    the stall itself, so it says so instead of aiohttp's or urllib3's phrasing."""
    if isinstance(e, TimeoutError) or "Timeout" in type(e).__name__:
        return f"no data for {DOWNLOAD_STALL_SECONDS}s"
    text = str(e).strip()
    return f"{type(e).__name__}: {text}" if text else type(e).__name__


RETRYABLE_ASYNC_ERRORS: tuple = (asyncio.TimeoutError, TimeoutError, ConnectionError)
if AIOHTTP_AVAILABLE:
    RETRYABLE_ASYNC_ERRORS = RETRYABLE_ASYNC_ERRORS + (aiohttp.ClientError,)


def hash_string(input_string):
    encoded_string = input_string.encode()
    hash_object = hashlib.sha256(encoded_string)
    return hash_object.hexdigest()


# ── NVMe cache vs. the durable volume ────────────────────────────────────────
#
# The NVMe cache (ANYMATIX_NVME_MODEL_CACHE) is a read-only-by-convention
# overlay: `extra_model_paths.yaml` puts it first in the search order so loads
# come from local disk, but a pod STOP wipes it — only the durable directory
# (the volume, on a pod) survives. Anything that WRITES a model file — a
# download, or an upload of a user's own weights — must therefore pick the
# first search path that is *not* the cache, never `paths[0]` blindly.
#
# bugs/an-uploaded-user-model-may-land-nvme: `/anymatix/uploadAsset` used
# `folder_paths.get_folder_paths(key)[0]` directly and landed a user's
# imported model in the cache, which a pod stop then wipes with no URL to
# re-fetch it from. `get_anymatix_models_dir` in `anymatix_checkpoint_fetcher.py`
# already got this right for downloads; these two functions are that same
# logic, pulled out so both callers share one definition and it can be unit
# tested without booting ComfyUI.
def is_under_nvme_cache(path: str, cache_root: Optional[str] = None) -> bool:
    """True when `path` is the NVMe cache directory or inside it.

    `cache_root` defaults to the `ANYMATIX_NVME_MODEL_CACHE` environment
    variable; pass it explicitly to test without touching the environment.
    """
    if cache_root is None:
        cache_root = os.environ.get("ANYMATIX_NVME_MODEL_CACHE") or ""
    cache_root = cache_root.strip()
    if not cache_root:
        return False
    cache_root = os.path.normpath(cache_root)
    p = os.path.normpath(path)
    return p == cache_root or p.startswith(cache_root + os.sep)


def pick_durable_dir(dirs, cache_root: Optional[str] = None) -> Optional[str]:
    """The first of `dirs` that is not under the NVMe cache — the one
    directory a write survives a pod stop in.

    Falls back to `dirs[0]` when every candidate is under the cache (or the
    cache is not configured at all, in which case nothing is skipped) so a
    caller never gets `None` back for a non-empty list where nothing else
    applies. Returns `None` only for an empty or falsy `dirs`.
    """
    dirs = list(dirs or [])
    if not dirs:
        return None
    for d in dirs:
        if not is_under_nvme_cache(d, cache_root):
            return d
    return dirs[0]


def is_valid_json_file(file_path: str) -> bool:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except Exception:
        return False


# How many times a hash reports its position, at most, over the whole file.
#
# 200 is 0.5% a step, and it is ComfyUI's own floor rather than a taste: its
# ProgressBar drops any update that moved less than 0.5% or arrived within
# 100 ms of the last one, so a finer step buys nothing that reaches the screen
# and a coarser one leaves the bar standing still. At the measured 530 MB/s it
# puts an update roughly every 0.115 s on a 12 GB weight — 200 of them across
# the 23 s — and a 469 MB file still gets 200 steps rather than one per
# gigabyte, which is the failure this number exists to avoid.
HASH_PROGRESS_STEPS = 200


def compute_file_sha256(file_path: str, chunk_size: int = 1024 * 1024,
                        progress: Optional[Callable[[int, Optional[int]], None]] = None) -> str:
    """Compute SHA256 hash of a file efficiently, saying how far it has got.

    `progress(bytes_hashed, total_bytes)` is the SAME SHAPE a download already
    reports, on purpose: a hash reads the file sequentially and knows its size,
    so the bar it drives needs no new mechanism and no new widget. It was
    always available and simply never asked for — which is why Vincenzo watched
    FETCH CHECKPOINT sit at 0% with an empty bar for 23 s while the machine was
    doing real, correct work. bugs/fetch-model-sits-0-while-fetcher-hashing

    The caller is told the start and the end as well as the middle, so a bar
    that is driven by this always begins at 0 and always arrives at 100 rather
    than stopping wherever the last step happened to fall.
    """
    try:
        total = os.path.getsize(file_path)
    except OSError:
        total = 0
    step = max(chunk_size, total // HASH_PROGRESS_STEPS) if total else chunk_size
    sha256_hash = hashlib.sha256()
    done = 0
    next_report = step
    if progress:
        progress(0, total)
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(byte_block)
            done += len(byte_block)
            if progress and done >= next_report:
                progress(done, total)
                next_report = done + step
    if progress:
        # The file is the length it turned out to be, not the length it was
        # when we stat-ed it: a bar must not finish at 99% because somebody
        # appended a byte, nor at 140% because they truncated it.
        progress(done, max(total, done))
    return sha256_hash.hexdigest()


CONTENT_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# How many candidate files an adoption may hash before giving up and downloading.
# Adoption picks candidates by content hash, so one is the normal number; the
# bound exists so a store full of same-named files cannot turn a fetch into a
# linear scan of the disk.
MAX_ADOPTION_CANDIDATES = 4


def canonical_model_name(file_name: str, sha256: str) -> str:
    """The name a finished download wears: `<original base>_<content sha256><ext>`.

    `file_name` is the provisional `<base>_<url hash><ext>` built by
    `download_file`, or a bare `<base><ext>`. The url hash suffix is dropped,
    because the FILE is named by its bytes and only the sidecar is named by the
    url. Two urls serving the same bytes therefore land on one filename, which
    is what makes adoption possible at all.

    Factored out of the post-download deduplication block so the pre-download
    adoption check and the post-download rename cannot disagree about the name.
    """
    parts = file_name.rsplit("_", 1)
    if len(parts) > 1:
        basename = parts[0]
        suffix = parts[1]
        ext_parts = suffix.split(".", 1)
        ext = ("." + ext_parts[1]) if len(ext_parts) > 1 else ""
    else:
        basename_parts = file_name.rsplit(".", 1)
        basename = basename_parts[0]
        ext = ("." + basename_parts[1]) if len(basename_parts) > 1 else ""
    return f"{basename}_{sha256}{ext}"


def content_sha256_from_headers(headers) -> Optional[str]:
    """The file's own sha256, if the server stated it.

    ONLY `X-Linked-Etag` is read, and the distinction is load-bearing. Measured
    against Hugging Face on 2026-09-17 for
    `Comfy-Org/Krea-2 .../loras/krea2_darkbrush.safetensors`:

        X-Linked-Etag: "f47c4316...cd5db7c6"   (on the 302 from huggingface.co)
        ETag:          "fdead7c2...2860d1a"    (on the CDN response it redirects to)

    Both are 64 hex characters. Only the first is the sha256 of the bytes —
    downloading the file and hashing it returned `f47c4316...cd5db7c6` exactly.
    The second is the Xet content-addressing hash, a different function of the
    same file. Accepting "any 64-hex ETag" would therefore name the file after
    a hash it does not have, and every later lookup would miss.

    Non-LFS files answer with a 40-hex git blob sha1 in both headers, and S3
    multipart objects with `<md5>-<n>`; the 64-hex shape rejects both, so the
    caller simply learns nothing and downloads, which is the correct outcome.
    """
    if not headers:
        return None
    raw = headers.get("X-Linked-Etag") or headers.get("x-linked-etag")
    if not raw:
        return None
    value = raw.strip()
    if value.startswith("W/"):
        value = value[2:]
    value = value.strip('"').strip().lower()
    return value if CONTENT_SHA256_RE.match(value) else None


def adoption_candidates(dirs, canonical_name: str, sha256: str, self_sidecar: str,
                        expected_size=None) -> List[str]:
    """Files that could already hold these bytes — strongest signal first.

    Nothing is hashed here. This only proposes; `adopt_existing_file` disposes.
    The scoping matters: without it, "do we already have this?" would mean
    hashing every model in the store on every fetch.

      1. the canonical name itself — one `stat`;
      2. any file whose name carries this sha256 under a different base name;
      3. any sidecar in the directory that already recorded this sha256;
      4. ANY FILE WITH THE SAME ORIGINAL NAME AND THE SAME SIZE, whatever
         suffix it wears.

    Rule 4 is not a loosening, it is the only rule that fires on a machine that
    has been running Anymatix. Until 2026-09-17 a parallel download — which is
    every large model — returned before the content rename, so the store is
    full of files named `<base>_<URL hash><ext>` whose sidecars carry no
    `sha256` at all. Rules 1 to 3 all look for a content hash that was never
    written. Rule 4 finds the file by the only two things that survived, and
    the caller then proves it by hashing it.

    A size that does not match is dropped here rather than in the caller, so
    the caller's hashing budget is spent on plausible files only.
    """
    seen = set()
    found: List[str] = []

    def offer(path):
        if not path or path in seen or not os.path.isfile(path):
            return
        try:
            if expected_size is not None and os.path.getsize(path) != expected_size:
                return
        except OSError:
            # It was there a moment ago and is not now. Proposing it would only
            # move the failure into the caller's hashing loop.
            return
        seen.add(path)
        found.append(path)

    stem_parts = canonical_name.rsplit(".", 1)
    ext = ("." + stem_parts[1]) if len(stem_parts) > 1 else ""
    base = stem_parts[0].rsplit("_", 1)[0]

    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        offer(os.path.join(d, canonical_name))
        try:
            entries = os.listdir(d)
        except OSError:
            continue
        for item in entries:
            if item.endswith(".json"):
                continue
            if item.rsplit(".", 1)[0].rsplit("_", 1)[-1].lower() == sha256:
                offer(os.path.join(d, item))
        for item in entries:
            if not item.endswith(".json") or item == self_sidecar:
                continue
            # A sidecar we cannot read is a sidecar with nothing to say
            # about this file. It is not this function's job to repair it.
            other = read_sidecar(os.path.join(d, item))
            if isinstance(other, dict) and str(other.get("sha256", "")).lower() == sha256:
                name = other.get("file_name")
                if name:
                    offer(os.path.join(d, name))
        # Rule 4 identifies by name and size only, so it is worth nothing
        # without a size to check against.
        if expected_size is None:
            continue
        for item in entries:
            if item.endswith(".json") or item.endswith(".part") or item.endswith(".complete") \
                    or item.endswith(".segments") or item.endswith(".segments.tmp"):
                continue
            if ext and not item.endswith(ext):
                continue
            item_stem = item[:-len(ext)] if ext else item
            if item_stem == base or item_stem.rsplit("_", 1)[0] == base:
                offer(os.path.join(d, item))
    return found


def adopt_existing_file(dirs, canonical_name: str, sha256: str, expected_size,
                        self_sidecar: str, label: str,
                        progress: Optional[Callable[[int, Optional[int]], None]] = None) -> Optional[str]:
    """A file the machine already has, PROVEN to be the bytes we were about to
    download — or None, which means download.

    The proof is a full re-hash, not the filename. A filename asserting a hash
    is written by this code and is normally true, but a truncated or swapped
    file wearing a canonical name would otherwise be served silently as a
    model, and a wrong model is a worse outcome than a redundant download.

    The re-hash is affordable, measured on 2026-09-17 on this machine:
    sha256 of a 469 MB safetensors took 0.89 s (~530 MB/s), against 22 s to
    download the same file — so verification costs about 4% of what it saves,
    and roughly 11 s for a 6 GB checkpoint.

    Anything that does not verify is skipped with a loud line and the caller
    downloads normally. Adoption never falls back to a file it could not prove.
    """
    # EVERY 64-HEX VALUE IN THESE LINES SAYS WHICH HASH IT IS.
    #
    # They used to read `verifying <file>_ae42d927... against ae42d927... for
    # <file>_4518bf83....safetensors`, which names two 64-hex values and says
    # what neither of them is. It reads as a file verified against one hash and
    # then adopted under a different one, and it was read that way on
    # 2026-09-17. It is not: `ae42d927...` is the sha256 of the BYTES and
    # `4518bf83...` is the sha256 of the URL STRING, which is what the sidecar
    # is named after and what `label` still carries when no content hash has
    # been recorded for this url yet. A log line that makes a correct system
    # look broken costs a reading, and this one already has.
    url_hash = self_sidecar[:-len(".json")] if self_sidecar.endswith(".json") else self_sidecar
    candidates = adoption_candidates(dirs, canonical_name, sha256, self_sidecar, expected_size)
    for path in candidates[:MAX_ADOPTION_CANDIDATES]:
        name = os.path.basename(path)
        try:
            print(f"[ANYMATIX ADOPT] hashing {name} to check it holds content sha256 "
                  f"{sha256}, which is what url hash {url_hash} asks for (recorded as {label})")
            actual = compute_file_sha256(path, progress=progress).lower()
            if actual != sha256:
                print(f"[ANYMATIX ADOPT] {name} holds content sha256 {actual}, not the "
                      f"{sha256} url hash {url_hash} asks for - refusing to adopt it")
                continue
            print(f"[ANYMATIX ADOPT] adopting {name} (content sha256 {sha256}) for "
                  f"url hash {url_hash}: no download needed")
            return path
        except Exception as e:
            print(f"[ANYMATIX ADOPT] could not hash {name}: {e}")
    return None


# HOW A MODEL ON DISK WAS LAST PROVEN, recorded in its sidecar as
# `verification`. Vincenzo, 2026-09-23, watching a download sit on
# "VERIFYING MODEL 9%": "it's a showstopper; for now do skip verification if
# size matches (and memorize the sizes)". So a finished download whose byte
# length is the one the server stated is accepted on that length alone, and
# the sidecar says so rather than pretending a hash was taken:
#
#   "size"    the length matched; no sha256 was computed over these bytes
#   "sha256"  the bytes were hashed (no size to compare with, or the file was
#             adopted, which always hashes)
#
# The Models manager is to show this and offer a hash on demand (release 1.1,
# TRACKERS/FEATURES/skip-model-verification-when-size-matches-models).
VERIFIED_BY_SIZE = "size"
VERIFIED_BY_SHA256 = "sha256"


def remembered_size(data: dict):
    """The byte length a sidecar lets us check a file against, or None.

    `file_size` is what the SERVER said (Content-Length) and always wins.
    `verified_size` is what THIS code measured on a file it had just proven,
    written for the servers that state no length — the "memorize the sizes"
    half of the 2026-09-23 request — so the next check is a `stat` too.
    """
    size = data.get("file_size")
    if size is None:
        size = data.get("verified_size")
    return size


def satisfied_by_sidecar(dirpath: str, data: dict) -> Optional[str]:
    """The model this sidecar describes, when the machine already has it — or
    None, which means something still has to be decided.

    A `stat`, and nothing else. It is not blind trust: `file_size` is what the
    SERVER said, so a file that was truncated, swapped for a shorter one, or
    half-written answers no and is proven again the long way. What it refuses
    to do is re-prove what was already proven — the hash of a 12 GB weight is
    ~23 s, and paying it on every run of a card whose models are all present
    contradicts the only claim the product makes about itself.

    A sidecar with no `file_size` is one whose server stated no Content-Length.
    Those are the small JSON metadata files, and parsing one is the same order
    of cost as stat-ing it, so that is the check they get.
    """
    name = data.get("file_name")
    if not name:
        return None
    path = os.path.join(dirpath, name)
    if not os.path.isfile(path):
        return None
    size = remembered_size(data)
    if size is None:
        if path.lower().endswith(".json") and is_valid_json_file(path):
            return path
        return None
    try:
        return path if os.path.getsize(path) == size else None
    except OSError:
        return None


# The three things `download_file` can make a person wait on, and the only
# words `phase` is ever called with. The app maps two of them to a label; the
# third is the item's own name, which already says "Fetch ...".
FETCH_PHASES = ("fetching", "verifying", "adopting")


def fetch_phase_message(node_id, prompt_id, phase: str, value: int, maximum: int) -> Optional[dict]:
    """The `anymatix.fetch_phase` payload, or None when there is nobody to tell.

    Here, beside the vocabulary it uses, so the words and their wire form
    cannot drift apart — and as a pure function so the one field that is easy
    to forget can be pinned by a test.

    `prompt_id` IS THAT FIELD. The app's global websocket dispatcher routes
    every message by it and drops the ones that have none, so a phase sent
    without a prompt id would be built, sent, and silently never arrive. It was
    caught by the app's typecheck rather than by anyone watching a bar, which
    is the only reason it is not in the shipped build.
    """
    if node_id is None or prompt_id is None:
        return None
    if phase not in FETCH_PHASES:
        return None
    return {
        "node": str(node_id),
        "prompt_id": str(prompt_id),
        "phase": phase,
        "value": int(value),
        "max": int(maximum),
    }


def satisfied_locally(dirs, urls) -> Optional[str]:
    """The model one of these urls already resolves to, found without asking
    anybody anything — or None.

    `download_file` answers this question for itself; this is for CALLERS who
    do something before calling it. `anymatix_checkpoint_fetcher` issues a
    request to the Civitai API (`expand_info`) ahead of every fetch, to
    deduplicate by the hash Civitai states — so a Civitai model the machine
    already holds paid for a url lookup on every single run, and on a machine
    with no internet paid for its timeout instead. The deduplication that block
    performs is the one `adopt_existing_file` now performs from the same hash,
    so skipping it when the url is already satisfied loses nothing and is the
    difference between a card that runs offline and one that hangs first.

    Several urls because the sidecar is named after the EFFECTIVE url (base
    plus any auth tail) and the base url is what gets persisted; several dirs
    because on a pod the bytes may be on the volume or in the NVMe cache.
    """
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for u in urls:
            if not u:
                continue
            data = read_sidecar(os.path.join(d, f"{hash_string(u)}.json"))
            if not data:
                continue
            hit = satisfied_by_sidecar(d, data)
            if hit:
                return hit
    return None


CREDENTIAL_QUERY_KEYS = {"token", "api_key", "apikey", "access_token"}


def redact_url(u: str, appended: Optional[str] = None) -> str:
    """Return a safe-to-log URL string.

    Always strips credential query parameters (token, api_key, ...), and in
    addition removes the specific parameters contained in 'appended' when the
    caller knows what it appended.

    It used to return 'u' unchanged when 'appended' was None, which made the
    name a promise it did not keep: the download URL carries the user's
    civitai key as a 'token=' tail, and every message interpolating a raw URL
    put that key into comfyui.runtime.log in cleartext. Measured 2026-08-26 in
    the Anymatix security audit (F3b). The request itself is always built from
    the untouched URL; only what gets LOGGED passes through here.
    """
    try:
        # Parse both URL and appended query tail
        p = urlparse(u)
        current = parse_qsl(p.query, keep_blank_values=True)
        remove_pairs = set(parse_qsl(appended, keep_blank_values=True)) if appended else set()
        # Drop the caller's appended pairs, and mask every credential parameter
        # whoever put it there.
        kept = []
        for key, value in current:
            if (key, value) in remove_pairs:
                continue
            if key.lower() in CREDENTIAL_QUERY_KEYS:
                kept.append((key, "<redacted>"))
                continue
            kept.append((key, value))
        new_query = urlencode(kept)
        return urlunparse(p._replace(query=new_query))
    except Exception:
        return u


def fetch_headers(url, session):
    """Fetch headers with error handling for missing requests"""
    if not REQUESTS_AVAILABLE:
        return {"file_name": None, "file_size": None, "remote_sha256": None}

    file_name = None
    file_size = None
    remote_sha256 = None
    try:
        # TODO: FIXME: should this be session.head??
        with session.get(url, allow_redirects=True, stream=True,
                         timeout=(DOWNLOAD_CONNECT_SECONDS, DOWNLOAD_STALL_SECONDS)) as response:
            response.raise_for_status()
            if "Content-Disposition" in response.headers:
                filename_match = re.search(
                    r'filename="(.+)"', response.headers["Content-Disposition"])
                if filename_match:
                    file_name = filename_match.group(1)
            if "Content-Length" in response.headers:
                file_size = int(response.headers.get('Content-Length', 0))
            # The content hash is announced on the FIRST response, not the last:
            # Hugging Face puts `X-Linked-Etag` on the 302 that sends us to the
            # CDN, and the CDN's own response does not carry it. `response`
            # here is the end of the chain, so the redirects have to be walked.
            for hop in list(response.history) + [response]:
                remote_sha256 = content_sha256_from_headers(hop.headers)
                if remote_sha256:
                    break
    except Exception:
        pass
    return {"file_name": file_name, "file_size": file_size, "remote_sha256": remote_sha256}


def fetch(url: str, session, callback: Callable[[bytes], None], local_file_size: int = 0, chunk_size=8192) -> None:
    """One connection, start to finish — and the only path that can resume."""
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library not available")
        
    req_headers = {}

    if local_file_size > 0:
        req_headers = {'Range': f'bytes={local_file_size}-'}

    try:
        # `timeout` is (connect, read): the read bound is the gap between two
        # bytes, never the length of the download — a silent link raises
        # instead of waiting for ever (`DOWNLOAD_STALL_SECONDS`).
        with session.get(url, headers=req_headers, allow_redirects=True, stream=True,
                         timeout=(DOWNLOAD_CONNECT_SECONDS, DOWNLOAD_STALL_SECONDS)) as response_2:
            response_2.raise_for_status()
            if local_file_size > 0 and response_2.status_code == 200:
                # The whole file, to a request for its tail: appended, it
                # would corrupt the part file. The caller asks again, and
                # starts over only when the server keeps doing it.
                raise RangeIgnored(f"HTTP 200 to Range bytes={local_file_size}-")
            for item in response_2.iter_content(chunk_size):
                # Check for ComfyUI interrupt signal before processing each chunk
                check_interrupted()
                callback(item)
    except RangeIgnored:
        raise
    except requests.HTTPError as e:
        raise Exception(f"HTTP request failed during single-stream download: {e}") from e
    except requests.RequestException as e:
        # The link, not the answer: worth another attempt from the byte reached.
        raise DownloadTransportError(describe_transport_error(e)) from e
    except Exception as e:
        # Re-raise InterruptProcessingException as-is for proper handling
        if "InterruptProcessingException" in type(e).__name__ or "InterruptProcessingException" in str(type(e)):
            raise
        raise Exception(f"Unexpected error during single-stream download: {e}") from e


class SegmentDownloader:
    """Many connections at once, each fetching its own byte range."""
    
    def __init__(self, url: str, file_path: str, total_size: int, 
                 progress_callback: Optional[Callable[[int, int], None]] = None,
                 max_connections: int = 8, segment_size: int = 1024*1024*8):  # 8MB segments
        self.url = url
        self.file_path = file_path
        self.total_size = total_size
        self.progress_callback = progress_callback
        self.max_connections = min(max_connections, max(1, total_size // (1024*1024)))  # Adaptive connections
        self.segment_size = segment_size
        self.downloaded_bytes = 0
        self.lock = threading.Lock()
        self.segments = []
        self.active_segments = {}
        self.failed_segments = []
        
    def _calculate_segments(self) -> List[Tuple[int, int, int]]:
        """Calculate optimal segment ranges with adaptive sizing"""
        segments = []
        remaining = self.total_size
        segment_id = 0
        start = 0
        
        # Dynamic segment sizing based on file size
        if self.total_size > 100 * 1024 * 1024:  # >100MB
            base_segment_size = 16 * 1024 * 1024  # 16MB segments
        elif self.total_size > 10 * 1024 * 1024:  # >10MB  
            base_segment_size = 4 * 1024 * 1024   # 4MB segments
        else:
            base_segment_size = 1024 * 1024       # 1MB segments
            
        while remaining > 0:
            # Adaptive segment size - smaller segments at the end for better load balancing
            if remaining < base_segment_size * 2:
                segment_size = remaining
            else:
                segment_size = min(base_segment_size, remaining)
                
            end = start + segment_size - 1
            segments.append((segment_id, start, end))
            start = end + 1
            remaining -= segment_size
            segment_id += 1
            
        return segments
        
    def _download_segment_sync(self, segment_id: int, start: int, end: int) -> bool:
        """Download a single segment with exponential backoff retry"""
        max_retries = 3
        backoff_base = 1.0
        
        for attempt in range(max_retries):
            try:
                headers = {'Range': f'bytes={start}-{end}'}
                with requests.get(self.url, headers=headers, stream=True, timeout=30) as response:
                    check_range_response(response.status_code, response.headers.get('Content-Range'), start, end)
                        
                    segment_data = b''
                    for chunk in response.iter_content(chunk_size=8192):
                        # Check for ComfyUI interrupt signal
                        check_interrupted()
                        if chunk:
                            segment_data += chunk
                            chunk_size = len(chunk)
                            with self.lock:
                                self.downloaded_bytes += chunk_size
                                # Update TQDM progress bar if available
                                if hasattr(self, '_progress_bar') and self._progress_bar:
                                    self._progress_bar.update(chunk_size)
                                if self.progress_callback:
                                    self.progress_callback(self.downloaded_bytes, self.total_size)
                                
                                # Console progress every 100MB for anymatix terminal
                                if self.downloaded_bytes % (100 * 1024 * 1024) < chunk_size:
                                    mb_downloaded = self.downloaded_bytes / (1024 * 1024)
                                    mb_total = self.total_size / (1024 * 1024)
                                    percent = (self.downloaded_bytes / self.total_size) * 100
                                    active_segments = len([s for s in self.active_segments.keys()])
                                    print(f"[ANYMATIX PARALLEL] {mb_downloaded:.0f}MB / {mb_total:.0f}MB ({percent:.1f}%) - {active_segments} segments active")
                    
                    if len(segment_data) != end - start + 1:
                        raise RangeNotHonoured(
                            f"segment {segment_id} received {len(segment_data)} of its {end - start + 1} bytes"
                        )

                    # Write segment to temp file
                    temp_path = f"{self.file_path}.segment_{segment_id}"
                    with open(temp_path, 'wb') as f:
                        f.write(segment_data)
                    
                    with self.lock:
                        self.active_segments[segment_id] = temp_path
                        
                    return True
                    
            except Exception as e:
                # Re-raise InterruptProcessingException for proper handling
                if "InterruptProcessingException" in type(e).__name__:
                    raise
                if attempt < max_retries - 1:
                    wait_time = backoff_base * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    with self.lock:
                        error_msg = f"Segment {segment_id} download failed after {max_retries} attempts"
                        self.failed_segments.append((segment_id, start, end, error_msg))
                    return False
        return False
        
    def download_parallel(self) -> bool:
        """Execute parallel download with intelligent load balancing"""
        progress_bar = None
        failed_segment_errors = []
        
        try:
            # Initialize TQDM progress bar
            if TQDM_AVAILABLE:
                try:
                    progress_bar = tqdm(
                        total=self.total_size,
                        desc="Threaded Download",
                        unit='B',
                        unit_scale=True,
                        leave=True
                    )
                    # Store as instance variable for access in _download_segment_sync
                    self._progress_bar = progress_bar
                except:
                    progress_bar = None
                    self._progress_bar = None
            else:
                self._progress_bar = None
        
            segments = self._calculate_segments()
            
            # Use ThreadPoolExecutor for optimal thread management
            with ThreadPoolExecutor(max_workers=self.max_connections, 
                                  thread_name_prefix="download_segment") as executor:
                # Submit all segment download tasks
                futures = {
                    executor.submit(self._download_segment_sync, seg_id, start, end): (seg_id, start, end)
                    for seg_id, start, end in segments
                }
                
                # Wait for completion with progress tracking
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    seg_id, start, end = futures[future]
                    try:
                        success = future.result()
                        if not success:
                            # Collect the error message from failed segments
                            with self.lock:
                                for failed_seg in self.failed_segments:
                                    if failed_seg[0] == seg_id and len(failed_seg) > 3:
                                        failed_segment_errors.append(f"Segment {seg_id}: {failed_seg[3]}")
                                    elif failed_seg[0] == seg_id:
                                        failed_segment_errors.append(f"Segment {seg_id} failed")
                    except Exception as e:
                        failed_segment_errors.append(f"Segment {seg_id} threw exception: {e}")
                        with self.lock:
                            self.failed_segments.append((seg_id, start, end, str(e)))
            
            # ONE RETRY PASS, AND IT USED TO BE UNREACHABLE.
            #
            # The raise below sat ABOVE this block, so a failed segment ended
            # the download before the retry it was written for could run. Each
            # segment already retries itself three times with exponential
            # backoff inside `_download_segment_sync`; this pass is the fourth
            # attempt, made one at a time rather than against a host that is
            # refusing sixteen connections — which is the case it exists for.
            if self.failed_segments:
                retry_errors = []
                for seg_id, start, end, *error_info in list(self.failed_segments):
                    if not self._download_segment_sync(seg_id, start, end):
                        error_msg = error_info[0] if error_info else f"Segment {seg_id} retry failed"
                        retry_errors.append(error_msg)
                    else:
                        # It landed this time, so it is not a failure any more.
                        failed_segment_errors = [
                            e for e in failed_segment_errors
                            if not e.startswith(f"Segment {seg_id}")
                        ]
                if retry_errors:
                    error_summary = "; ".join(retry_errors[:3])
                    if len(retry_errors) > 3:
                        error_summary += f" and {len(retry_errors) - 3} more retry failures"
                    raise Exception(f"Segment retry failed: {error_summary}")

            if failed_segment_errors:
                error_summary = "; ".join(failed_segment_errors[:5])  # Show up to 5 errors
                if len(failed_segment_errors) > 5:
                    error_summary += f" and {len(failed_segment_errors) - 5} more errors"
                raise Exception(f"Parallel download failed due to segment errors: {error_summary}")
            
            if progress_bar:
                progress_bar.close()
            # Clean up progress bar reference
            if hasattr(self, '_progress_bar'):
                delattr(self, '_progress_bar')
                        
            return True
        
        except Exception as e:
            if progress_bar:
                try:
                    progress_bar.close()
                except:
                    pass
            # Clean up progress bar reference
            if hasattr(self, '_progress_bar'):
                delattr(self, '_progress_bar')
            # Re-raise the exception to propagate it up to the node
            raise e
        
    def assemble_file(self) -> bool:
        """Assemble segments into final file with integrity verification"""
        try:
            missing_segments = []
            # The file about to be rewritten is not the file any earlier
            # completion record describes.
            clear_part_completion(self.file_path)
            with open(self.file_path, 'wb') as output_file:
                for i in range(len(self.active_segments)):
                    segment_path = self.active_segments.get(i)
                    if not segment_path or not os.path.exists(segment_path):
                        missing_segments.append(i)
                        continue
                        
                    with open(segment_path, 'rb') as segment_file:
                        output_file.write(segment_file.read())
            
            if missing_segments:
                raise Exception(f"Missing segments during assembly: {missing_segments}")
            
            # Cleanup temp files
            for segment_path in self.active_segments.values():
                try:
                    os.remove(segment_path)
                except:
                    pass
                    
            # Verify file size
            final_size = os.path.getsize(self.file_path)
            if final_size != self.total_size:
                raise Exception(f"File size mismatch after assembly: expected {self.total_size}, got {final_size}")

            # Every segment landed and was written in order: this path knows it
            # finished, so it says so where a later process can read it.
            mark_part_complete(self.file_path, self.total_size)

            return True
            
        except Exception as e:
            # Clean up any temp files on error
            for segment_path in self.active_segments.values():
                try:
                    if os.path.exists(segment_path):
                        os.remove(segment_path)
                except:
                    pass
            # Remove incomplete output file
            try:
                if os.path.exists(self.file_path):
                    os.remove(self.file_path)
                    clear_part_completion(self.file_path)
            except:
                pass
            raise Exception(f"Failed to assemble downloaded file: {e}") from e


class RangeNotHonoured(Exception):
    """The server answered a range request with something other than that range."""


class RangeIgnored(RangeNotHonoured):
    """The server answered a range request with the whole file (`200`).

    Worth asking again before believing it: see `RANGE_WARMUP_PAUSES`."""


class SignedUrlRefused(Exception):
    """The resolved (signed) address answered 401/403: its signature expired,
    so the address the card names has to be resolved again."""


class DownloadUrl:
    """
    THE ADDRESS THE SEGMENTS ASK, RESOLVED ONCE.

    `source` is the address the card names (Civitai's
    `/api/download/models/<id>`); `current` is where its redirect leads — a
    signed storage URL. Every segment and every retry used to ask `source`, so
    each one cost a round trip to the API and got a differently signed URL.
    Now they share one. A signature lasts an hour on Civitai (by the timestamps
    in it), which a slow download can outlive: a 401/403 from `current`
    resolves `source` again (`refresh`).
    """

    def __init__(self, source: str, resolved: Optional[str] = None):
        self.source = source
        self.current = resolved or source
        self._refreshing: Optional["asyncio.Lock"] = None

    async def refresh(self, session, stale: str) -> None:
        if self._refreshing is None:
            self._refreshing = asyncio.Lock()
        async with self._refreshing:
            if self.current != stale:
                return  # another segment already resolved it again
            async with session.head(self.source, allow_redirects=True) as response:
                response.raise_for_status()
                self.current = str(response.url)
            print(f"[ANYMATIX DOWNLOAD] The signed address was refused; resolved {redact_url(self.source)} again")


def check_range_response(status: int, content_range: Optional[str], start: int, end: int) -> None:
    """
    A RANGE REQUEST IS ANSWERED BY THAT RANGE, OR NOT AT ALL.

    `200` used to be accepted beside `206`. A `200` is the server IGNORING the
    range and sending the whole file, and the segment loop wrote every byte of
    it from `start` on: on fmt-5000, 2026-10-07, a Civitai LTX 2.3 VAE part file
    grew to 2,722,984,832 of its 1,452,258,578 bytes
    (`bugs/a-parallel-segment-download-accepts-http-200`). Only a `206` whose
    `Content-Range` starts at `start` is this segment; anything else raises,
    the parallel strategy fails, and the downloader falls back to one stream,
    which knows what to do with a `200`.
    """
    if status == 200:
        raise RangeIgnored(f"HTTP 200 to Range bytes={start}-{end}")
    if status != 206:
        raise RangeNotHonoured(f"HTTP {status} to Range bytes={start}-{end}")
    if content_range:
        try:
            unit, _, rest = content_range.strip().partition(" ")
            first = int(rest.split("-", 1)[0])
        except (ValueError, IndexError):
            raise RangeNotHonoured(f"unreadable Content-Range {content_range!r} for bytes={start}-{end}")
        if unit.lower() != "bytes" or first != start:
            raise RangeNotHonoured(f"Content-Range {content_range!r} does not start at {start}")


async def fetch_async_segment(session, url: str, start: int, end: int,
                            segment_id: int, progress_callback: Optional[Callable] = None,
                            part_path: Optional[str] = None,
                            on_landed: Optional[Callable[[int, int], None]] = None,
                            reached: Optional[dict] = None) -> int:
    """
    Download one byte range STRAIGHT TO ITS OFFSET in the part file.

    `reached[segment_id]`, when given, is kept at the absolute offset of the
    last byte written, chunk by chunk — what a retry resumes from
    (`fetch_async_segment_resuming`).

    This used to build the segment in memory — `segment_data += chunk` — and
    hand the bytes back to be written once every segment had arrived. With a
    segment size of total/16 that put the WHOLE file in RAM (2.14 GB for
    t3_mtl23ls_v2, plus the copies that `+=` makes), which is a plausible way
    to have a remote ComfyUI killed while it downloads, and is certainly a way
    to make a shared machine swap. Each task opens its own handle: POSIX is
    happy with concurrent writes to disjoint ranges, so no lock is needed and
    the peak is one chunk per connection.

    `on_landed(segment_id, offset)` is told, every `SEGMENT_JOURNAL_STRIDE`
    bytes and at the end, the absolute offset up to which this segment's bytes
    have been handed to the kernel — flushed, so a process that dies the next
    instant has not lost them. That is what `SegmentJournal` records, and the
    only thing that lets a killed download resume (see there).
    """
    if not AIOHTTP_AVAILABLE:
        raise ImportError("aiohttp not available")
    if not AIOFILES_AVAILABLE:
        raise ImportError("aiofiles not available")
    if not part_path:
        raise ValueError("fetch_async_segment needs the part file to write into")

    headers = {'Range': f'bytes={start}-{end}'}
    address = url.current if isinstance(url, DownloadUrl) else url

    async with session.get(address, headers=headers) as response:
        if response.status in (401, 403) and isinstance(url, DownloadUrl) and address != url.source:
            response.close()
            raise SignedUrlRefused(f"HTTP {response.status} from the signed address")
        try:
            check_range_response(response.status, response.headers.get('Content-Range'), start, end)
        except RangeNotHonoured:
            # Unread: a `200` is the whole file, and reading it to reuse the
            # connection would download it.
            response.close()
            raise

        async with aiofiles.open(part_path, 'r+b') as f:
            await f.seek(start)
            position = start
            unrecorded = 0
            async for chunk in response.content.iter_chunked(8192):
                if position + len(chunk) > end + 1:
                    raise RangeNotHonoured(
                        f"segment {segment_id} received more than its {end - start + 1} bytes"
                    )
                await f.write(chunk)
                position += len(chunk)
                unrecorded += len(chunk)
                if reached is not None:
                    reached[segment_id] = position
                if progress_callback:
                    progress_callback(len(chunk))
                if on_landed and unrecorded >= SEGMENT_JOURNAL_STRIDE:
                    await f.flush()
                    on_landed(segment_id, position)
                    unrecorded = 0
            await f.flush()
            if on_landed:
                on_landed(segment_id, position)

        return segment_id


async def fetch_async_segment_resuming(session, url: str, start: int, end: int,
                                       segment_id: int, progress_callback: Optional[Callable] = None,
                                       part_path: Optional[str] = None,
                                       on_landed: Optional[Callable[[int, int], None]] = None) -> int:
    """
    One segment, carried on from the byte it reached each time its link fails.

    `bugs/a-local-run-stalled-model-download-given` (app repo). On sleipnir,
    2026-10-10, a Civitai VAE stopped at 591M of 1.45G and nothing happened for
    28 minutes: the session had no read timeout, so a segment whose TCP flow
    went silent without a reset waited for ever, and one segment that failed
    failed the whole download. Now the session's `sock_read` bound turns the
    silence into an error, and this resumes the segment from `reached` — the
    exact byte, not the last journal stride — after a short pause, up to
    `DOWNLOAD_ATTEMPTS` times. Only transport failures are retried: a range the
    server will not honour is not going to start honouring it, and is the
    single-stream fallback's job.
    """
    reached = {segment_id: start}
    attempts = max(1, DOWNLOAD_ATTEMPTS)
    warmups = 0
    refreshes = 0
    attempt = 0
    while attempt < attempts:
        if reached[segment_id] > end:
            return segment_id
        try:
            return await fetch_async_segment(
                session, url, reached[segment_id], end, segment_id,
                progress_callback, part_path, on_landed=on_landed, reached=reached
            )
        except RangeIgnored:
            # Asked again, not given up on, and not an attempt: the server is
            # warming the file, not failing (`RANGE_WARMUP_PAUSES`).
            if warmups >= len(RANGE_WARMUP_PAUSES):
                raise
            await asyncio.sleep(RANGE_WARMUP_PAUSES[warmups])
            warmups += 1
            continue
        except SignedUrlRefused:
            if refreshes >= 2:
                raise
            refreshes += 1
            await url.refresh(session, url.current)
            continue
        except RETRYABLE_ASYNC_ERRORS as e:
            if attempt + 1 >= attempts:
                reason = f"{describe_transport_error(e)}, {attempts} attempts"
                raise DownloadStalled(
                    f"segment {segment_id} stopped at byte {reached[segment_id]} of {end + 1}: {reason}",
                    reason
                ) from e
            print(f"[ANYMATIX DOWNLOAD] Segment {segment_id} stopped at byte {reached[segment_id]} "
                  f"({describe_transport_error(e)}); attempt {attempt + 2} of {attempts} resumes there")
            await asyncio.sleep(retry_pause_seconds(attempt))
            attempt += 1
    return segment_id


class AsyncParallelDownloader:
    """Ultra-modern async parallel downloader with HTTP/2 and connection pooling"""
    
    def __init__(self, url: str, file_path: str, total_size: int,
                 progress_callback: Optional[Callable[[int, int], None]] = None,
                 max_connections: int = 16):
        if not AIOHTTP_AVAILABLE or not AIOFILES_AVAILABLE:
            raise ImportError("aiohttp and aiofiles required for async downloading")
            
        self.url = url
        self.file_path = file_path 
        self.total_size = total_size
        self.progress_callback = progress_callback
        self.max_connections = max_connections
        self.downloaded_bytes = 0
        self.lock = asyncio.Lock()
        # How many bytes of resumable prefix the salvage decided to KEEP, if it
        # ran. Zero means there is nothing on disk worth resuming from, and the
        # error path is then free to remove the part file. See `download_async`.
        self.kept_prefix = 0
        # True when a stall ended the attempt and the part file with its
        # journal was kept whole for the next run (see `download_async`).
        self.kept_journal = False
        
    async def download_async(self) -> bool:
        """Execute async parallel download with HTTP/2 optimization"""
        progress_bar = None
        try:
            # Initialize TQDM progress bar
            if TQDM_AVAILABLE:
                try:
                    progress_bar = tqdm(
                        total=self.total_size,
                        desc="Async Download",
                        unit='B',
                        unit_scale=True,
                        leave=True
                    )
                except:
                    progress_bar = None
            
            # Calculate segments. A journal left by a download that was killed
            # fixes the layout: its offsets only mean something in its own.
            journal = read_segment_journal(self.file_path, self.total_size)
            segment_size = journal["segment_size"] if journal else \
                max(1024*1024, self.total_size // self.max_connections)  # At least 1MB per segment
            segments = []
            
            for i in range(0, self.total_size, segment_size):
                start = i
                end = min(i + segment_size - 1, self.total_size - 1)
                segments.append((len(segments), start, end))
            
            # Configure HTTP/2 connector with connection pooling
            # KEEP THE CONNECTIONS ALIVE, WHICH IS THE ENTIRE POINT.
            #
            # force_close=True closes every connection after one response and
            # is mutually exclusive with keepalive_timeout — aiohttp raises
            # "keepalive_timeout cannot be set if force_close is True" the
            # moment the session is built. So the parallel downloader has been
            # dying at construction and silently falling back to one stream on
            # every large download. Keep-alive is what makes segment fetching
            # worth doing at all, so force_close is the one that goes.
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections,
                enable_cleanup_closed=True,
                keepalive_timeout=30
            )
            
            # NO TOTAL — a 27 GB weight takes as long as it takes — BUT A
            # BOUND ON SILENCE. `sock_read` is the gap between two reads on one
            # connection: without it a flow that died without a reset (a route
            # change, a VPN coming up, a NAT that forgot us) left a segment
            # waiting for ever, which is how sleipnir sat at 591M of 1.45G for
            # 28 minutes. `fetch_async_segment_resuming` turns the timeout into
            # a resume from the byte reached.
            timeout = aiohttp.ClientTimeout(
                total=None,
                connect=DOWNLOAD_CONNECT_SECONDS,
                sock_connect=DOWNLOAD_CONNECT_SECONDS,
                sock_read=DOWNLOAD_STALL_SECONDS,
            )
            
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'AnymatixFetcher/2.0 (Parallel)'}
            ) as session:
                
                def progress_update(bytes_read):
                    self.downloaded_bytes += bytes_read
                    if progress_bar:
                        progress_bar.update(bytes_read)
                    if self.progress_callback:
                        self.progress_callback(self.downloaded_bytes, self.total_size)
                
                # A PART FILE, NEVER THE DESTINATION, AND NEVER SPARSE AT THE
                # DESTINATION. Writing at offsets means the file reaches full
                # size immediately, and `download_file` decides a download is
                # complete by comparing the size on disk with the expected
                # one — so a preallocated destination would announce a
                # half-downloaded weight as finished. The part file carries
                # that risk instead, and only a completed download is renamed
                # into place.
                # THE PATH HANDED IN IS ALREADY THE PART PATH. DO NOT DERIVE
                # ANOTHER ONE.
                #
                # `download_file` builds it once with `part_path_for` and passes
                # THAT into `fetch_parallel`. Appending `.part` here built it a
                # second time, so every parallel download wrote to
                # `<name>.safetensors.part.part` — measured 2026-09-05 on a
                # remote, a complete 323 MB SAM2 weight wearing two suffixes.
                #
                # It is not a cosmetic name. `finalize_download` is then handed
                # `<name>.part`, which never existed, and raises "Download
                # produced no data" on a transfer that finished perfectly; the
                # next run reads `local_file_size` from that same absent
                # `.part`, sees 0, and re-downloads the whole file from byte
                # zero. The `.part` scheme exists to make interrupted downloads
                # resumable, and on this path it could not resume once.
                # `bugs/the-parallel-download-writes-part-part-can`.
                part_path = self.file_path
                os.makedirs(os.path.dirname(part_path) or ".", exist_ok=True)
                if journal:
                    # A KILLED DOWNLOAD, CARRIED ON — NOT HASHED, NOT RESTARTED.
                    # `bugs/a-dropped-ssh-link-remote-machine-kills`: see
                    # `SegmentJournal`. The part file is kept as it is; each
                    # segment carries on from the offset it had flushed.
                    landed = journal["done"]
                    already = sum(min(landed.get(i, 0), e - s + 1) for i, s, e in segments)
                    print(f"[ANYMATIX DOWNLOAD] Resuming a parallel download that was interrupted: "
                          f"{already} of {self.total_size} bytes are already on disk")
                    progress_update(already)
                else:
                    # THE MOMENT THIS RUNS, THE FILE IS THE RIGHT SIZE AND EMPTY.
                    # Any completion recorded for an earlier part file at this path
                    # describes bytes that no longer exist, so it goes first: a
                    # marker that survives its own bytes is how a hole gets adopted.
                    clear_part_completion(part_path)
                    async with aiofiles.open(part_path, 'wb') as prealloc:
                        await prealloc.truncate(self.total_size)
                    landed = {}
                record = SegmentJournal(part_path, self.total_size, segment_size, segments, landed)

                # Download all segments concurrently — what is left of each.
                tasks = [
                    fetch_async_segment_resuming(session, self.url, start + landed.get(seg_id, 0), end, seg_id,
                                                 progress_update, part_path, on_landed=record.landed)
                    for seg_id, start, end in segments
                    if start + landed.get(seg_id, 0) <= end
                ]
                
                segment_results = await asyncio.gather(*tasks, return_exceptions=True)

                failed_segments = [str(r) for r in segment_results if isinstance(r, BaseException)]
                stalled = [r for r in segment_results if isinstance(r, DownloadStalled)]
                if stalled and len(stalled) == len([r for r in segment_results if isinstance(r, BaseException)]):
                    # THE LINK IS GONE, NOT THE SERVER'S PATIENCE WITH RANGES.
                    # Every failure is a segment that ran out of attempts on a
                    # silent or broken connection, so a single stream would
                    # only stall the same way — and the salvage below would
                    # throw away every segment past the first hole. The part
                    # file and its journal are kept exactly as they are, and
                    # the next run carries each segment on from its byte.
                    self.kept_journal = True
                    raise DownloadStalled("; ".join(str(r) for r in stalled[:3]), stalled[0].reason)
                if failed_segments:
                    # HAND WHAT LANDED TO THE RESUME PATH, WHICH ALREADY EXISTS.
                    #
                    # Segments finish out of order, so a part file with a hole
                    # in the middle is not a resumable prefix — but the
                    # segments BEFORE the first failure are one, and a single
                    # ordered stream can carry on from there. Truncating to
                    # that boundary is what turns a died-at-18% download into
                    # 18% already done, instead of starting from zero on every
                    # retry (which is what a downloader that only wrote at the
                    # end could never avoid).
                    # By what LANDED, not by task index: segments a resumed
                    # download found already whole have no task at all.
                    completed = set(record.finished)
                    prefix = 0
                    for seg_id, seg_start, seg_end in segments:
                        if seg_id not in completed:
                            break
                        prefix = seg_end + 1
                    try:
                        if prefix > 0:
                            with open(part_path, 'r+b') as trim:
                                trim.truncate(prefix)
                            # SAY SO, BECAUSE THE ERROR PATH BELOW DELETES THIS
                            # FILE OTHERWISE. The raise a few lines down lands
                            # in `except Exception` at the end of this method,
                            # which removed `self.file_path` on the assumption
                            # that a surviving part file must be a holed one.
                            # It is not: it is what we just truncated on
                            # purpose.
                            self.kept_prefix = prefix
                            # A prefix is resumed by ONE stream; the journal's
                            # per-segment offsets describe a file that no longer
                            # exists at this length.
                            clear_segment_journal(part_path)
                            # No rename: `part_path` IS `self.file_path` (the
                            # caller's part file). `os.replace` here was a
                            # no-op that read like a move, which is how the
                            # caller came to look for these bytes under the
                            # model's final name and find nothing.
                            print(
                                f"[ANYMATIX DOWNLOAD] Keeping the {prefix} bytes that landed; "
                                f"a single stream can resume from there"
                            )
                        elif os.path.exists(part_path):
                            os.remove(part_path)
                            clear_part_completion(part_path)
                    except Exception as salvage_error:
                        print(f"[ANYMATIX DOWNLOAD] Could not keep the partial download: {salvage_error}")

                    error_summary = "; ".join(failed_segments[:3])
                    if len(failed_segments) > 3:
                        error_summary += f" and {len(failed_segments) - 3} more async segment errors"
                    raise Exception(f"Async parallel download failed: {error_summary}")

                # Nothing to assemble: every segment wrote its own range.
                written = os.path.getsize(part_path)
                if written != self.total_size:
                    raise Exception(
                        f"Async parallel download wrote {written} bytes, expected {self.total_size}"
                    )
                # Again no rename, and for the same reason: the caller's
                # `finalize_download` is the ONE place a `.part` earns the
                # model's name.
                #
                # THIS IS THE ONLY MOMENT ANYTHING KNOWS THE DOWNLOAD FINISHED.
                # Every segment returned without raising and the file is the
                # declared length: written down here, that fact survives the
                # process. Until 2026-09-17 it was printed and forgotten, and
                # the next run had nothing but the size to go on.
                mark_part_complete(part_path, self.total_size)
                clear_segment_journal(part_path)

                if progress_bar:
                    progress_bar.close()
                return True
                
        except Exception as e:
            if progress_bar:
                try:
                    progress_bar.close()
                except:
                    pass
            # A part file still here was not salvageable: its holes are in the
            # middle, so it is neither a resumable prefix nor a download.
            #
            # UNLESS THE SALVAGE KEPT IT, AND IT DID.
            #
            # The failed-segment branch above truncates to the last byte before
            # the first hole and then raises, so the exception it raises arrives
            # HERE — and this `os.remove` deleted the prefix a few frames after
            # printing "Keeping the N bytes that landed". The caller then found
            # no part file, resumed from 0, and a 27 GB weight interrupted at
            # 18% started again from zero. Fixing the caller to read the part
            # path (`bugs/an-interrupted-parallel-download-restarted-from-zero`)
            # was necessary and was not enough: by the time it looked, there was
            # nothing left to look at.
            try:
                stale = self.file_path
                if self.kept_prefix <= 0 and not self.kept_journal and os.path.exists(stale):
                    os.remove(stale)
                    clear_part_completion(stale)
            except Exception:
                pass
            # Re-raise exception to propagate to node
            raise Exception(f"Async parallel download failed: {e}") from e


def check_range_support(url: str) -> Tuple[bool, Optional[int]]:
    """Whether `url` serves byte ranges, and its size (see `resolve_for_ranges`)."""
    supports, size, _ = resolve_for_ranges(url)
    return supports, size


def probe_range(final_url: str, label: str = "") -> Optional[bool]:
    """
    ASK FOR ONE BYTE, AND ASK AGAIN IF THE ANSWER IS THE WHOLE FILE.

    True on a `206`; False when every warm-up retry still got a `200`, or the
    server answered something else; None when the probe could not be made (no
    network answer), which tells the caller nothing.

    `Accept-Ranges: bytes` is not the answer: Civitai's B2 sends it on the very
    responses that ignore the range (`RANGE_WARMUP_PAUSES`). One real request
    is, and it also warms the file for the segments that follow, so they meet
    `206` from their first request instead of all meeting `200` together.
    """
    for warmup in range(len(RANGE_WARMUP_PAUSES) + 1):
        try:
            with requests.get(final_url, headers={'Range': 'bytes=0-0'}, stream=True,
                              timeout=(DOWNLOAD_CONNECT_SECONDS, DOWNLOAD_STALL_SECONDS),
                              allow_redirects=False) as response:
                status = response.status_code
        except requests.RequestException as e:
            print(f"[ANYMATIX RANGE] Range probe failed: {describe_transport_error(e)}")
            return None
        if status == 206:
            if warmup:
                print(f"[ANYMATIX RANGE] {label}range requests honoured after {warmup} "
                      f"answer(s) of the whole file (a cold file warming up)")
            return True
        if status != 200:
            print(f"[ANYMATIX RANGE] Range probe answered HTTP {status}")
            return False
        if warmup < len(RANGE_WARMUP_PAUSES):
            print(f"[ANYMATIX RANGE] {label}a range request was answered with the whole file; "
                  f"asking again in {RANGE_WARMUP_PAUSES[warmup]}s")
            time.sleep(RANGE_WARMUP_PAUSES[warmup])
            check_interrupted()
    print(f"[ANYMATIX RANGE] {label}the server keeps answering ranges with the whole file")
    return False


def resolve_for_ranges(url: str) -> Tuple[bool, Optional[int], str]:
    """
    Where `url` really lives, whether that place serves byte ranges, and the size.

    The redirect is followed ONCE here and the final address returned, so the
    segments ask it directly (`DownloadUrl`) instead of each going through the
    redirect for a freshly signed copy.
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library not available for range support check")

    try:
        with requests.head(url, allow_redirects=True,
                           timeout=(DOWNLOAD_CONNECT_SECONDS, DOWNLOAD_STALL_SECONDS)) as response:
            response.raise_for_status()
            final_url = response.url
            accepts_ranges = response.headers.get('Accept-Ranges', '').lower() == 'bytes'
            content_length = response.headers.get('Content-Length')
            file_size = int(content_length) if content_length else None
    except requests.RequestException as e:
        raise Exception(f"Failed to check range support for {redact_url(url)}: {e}") from e

    if file_size is not None and file_size <= 1:
        return False, file_size, final_url
    probed = probe_range(final_url)
    if probed is None:
        # No answer to the probe: the header is all there is to go on.
        return accepts_ranges, file_size, final_url
    if probed:
        print(f"[ANYMATIX RANGE] Server supports Range requests ({final_url.split('/')[2]})")
    return probed, file_size, final_url


def fetch_parallel(url: str, file_path: str, callback: Optional[Callable[[int, Optional[int]], None]] = None,
                  local_file_size: int = 0, max_connections: int = 8) -> bool:
    """
    Parallel download with intelligent fallback.

    `file_path` IS THE PATH TO WRITE, and the caller has already made it a part
    path (`part_path_for`). Nothing in here may append `.part` to it: doing so
    produced `<name>.safetensors.part.part`, which `finalize_download` cannot
    see and no later run can resume from.
    """
    
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library not available for parallel download")
    
    # Check server capabilities, at the address the redirect leads to.
    supports_ranges, total_size, final_url = resolve_for_ranges(url)
    address = DownloadUrl(url, final_url)
    
    if not supports_ranges or not total_size:
        # The probe asked for a range and was refused even after the warm-up
        # (`probe_range`), or the size is unknown: one stream it is.
        print(f"[ANYMATIX DOWNLOAD] Parallel download not possible: supports_ranges={supports_ranges}, total_size={total_size}")
        return False
        
    # Skip parallel for small files (< 5MB)
    if total_size and total_size < 5 * 1024 * 1024:
        print(f"[ANYMATIX DOWNLOAD] Skipping parallel for small file ({total_size//1024//1024}MB < 5MB)")
        return False
        
    # Handle resume scenario
    if local_file_size > 0:
        if local_file_size >= total_size:
            return True  # Already complete
        # A resume needs one ordered stream from the byte we stopped at.
        print(f"[ANYMATIX DOWNLOAD] Resuming an interrupted download — one connection, picking up where it stopped")
        return False
    
    # Choose download strategy based on available dependencies
    try:
        # Try async method first (fastest) if available
        if AIOHTTP_AVAILABLE and AIOFILES_AVAILABLE:
            print(f"[ANYMATIX DOWNLOAD] Using async parallel download strategy")

            async def run_async():
                downloader = AsyncParallelDownloader(address, file_path, total_size, callback, max_connections)
                success = await downloader.download_async()
                if not success:
                    raise Exception(f"Async parallel download failed for {redact_url(url)}")
                return success

            # Run the async download in a DEDICATED THREAD with its own loop.
            # A fresh loop in THIS thread is not enough: asyncio refuses to
            # run any loop in a thread that already has one running, and
            # ComfyUI executes nodes with its loop running — so this path
            # raised "Cannot run the event loop while another loop is
            # running" every time, silently demoting every download to
            # single-stream.
            outcome: list = []
            running: dict = {}

            async def run_cancellable():
                running["loop"] = asyncio.get_running_loop()
                running["task"] = asyncio.current_task()
                return await run_async()

            def _runner():
                try:
                    outcome.append(asyncio.run(run_cancellable()))
                except BaseException as e:
                    outcome.append(e)

            t = threading.Thread(target=_runner, name="anymatix-parallel-download")
            t.start()
            # STOP REACHES THE DOWNLOAD. A bare `t.join()` left this node deaf
            # to ComfyUI's interrupt for as long as the transfer lasted — and
            # for ever when it stalled: on sleipnir, 2026-10-10, the app's
            # interrupt at 09:59 ended nothing, and the next prompt queued
            # behind a node that would never return. The node thread now
            # wakes twice a second to ask, and cancels the transfer's task when
            # told; the part file and its journal stay for a resume.
            while t.is_alive():
                t.join(0.5)
                if not t.is_alive():
                    break
                try:
                    check_interrupted()
                except BaseException:
                    loop, task = running.get("loop"), running.get("task")
                    if loop is not None and task is not None:
                        try:
                            loop.call_soon_threadsafe(task.cancel)
                        except RuntimeError:
                            # The loop closed between the join and here: the
                            # transfer ended on its own, nothing to cancel.
                            pass
                    t.join()
                    raise
            if outcome and isinstance(outcome[0], BaseException):
                raise outcome[0]
            return bool(outcome and outcome[0])
        else:
            # Fallback to threaded parallel download
            print(f"[ANYMATIX DOWNLOAD] Using threaded parallel download strategy")
            downloader = SegmentDownloader(url, file_path, total_size, callback, max_connections)
            if not downloader.download_parallel():
                raise Exception(f"Threaded parallel download failed for {redact_url(url)}")
            if not downloader.assemble_file():
                raise Exception(f"Failed to assemble downloaded segments for {redact_url(url)}")
            return True
        
    except Exception as e:
        if is_interrupt(e):
            raise
        print(f"[ANYMATIX DOWNLOAD] Parallel download strategy failed: {e}")
        # Re-raise the exception instead of returning False so it propagates to the node
        raise Exception(f"Parallel download failed: {e}")  from e


def sidecar_url_matches(stored_url, url: str) -> bool:
    """Whether a sidecar's stored base url is the url being deleted.

    `download_file` persists the BASE url and names the sidecar after the
    EFFECTIVE one (base plus any auth tail), so the two are not interchangeable
    and a credentialled url is recognised by prefix.
    """
    if not isinstance(stored_url, str) or not stored_url:
        return False
    return url == stored_url or url.startswith(stored_url + "?") or url.startswith(stored_url + "&")


def read_sidecar(path: str) -> Optional[dict]:
    """The sidecar at `path`, if it is one. Leaves its access time alone.

    A sidecar's atime is the one honest "a card last asked for this model"
    signal a RunPod volume keeps (bootstrap.py's volume auto-clean reads it).
    Every caller of this function is SCANNING -- looking at a sidecar that
    belongs to some url other than the one being served -- and a scan that
    updates atime marks every sibling as just used, so nothing is ever old
    enough to evict. Only `fetch_model`'s own `open(store_path)` is a use.
    """
    try:
        st = os.stat(path)
        with open(path, "r") as contents:
            data = json.load(contents)
    except (OSError, ValueError):
        return None
    try:
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    except OSError:
        pass
    return data if isinstance(data, dict) else None


def model_file_is_spoken_for(dirpath: str, model_file: str, is_doomed) -> bool:
    """Whether a sidecar that SURVIVES this deletion still names `model_file`.

    THE ONE RULE BOTH DELETION PATHS OBEY, kept in one place because they had
    drifted: `delete_files` here and `serve_delete` in `__init__.py` are the
    two ways a model is removed, and a rule written twice is a rule that is
    right once.

    `is_doomed(filename, data)` says whether that sidecar is going too. A
    sidecar being deleted in the same breath does not get a vote: when two of
    them named one file, each counted the other as a referrer, neither
    released the bytes, and both sidecars were deleted anyway -- leaving the
    file on disk with nothing pointing at it.
    """
    try:
        entries = os.listdir(dirpath)
    except OSError:
        # Cannot read the directory, so cannot prove the file is free.
        return True
    for other in entries:
        if not other.endswith(".json"):
            continue
        data = read_sidecar(os.path.join(dirpath, other))
        if not data or data.get("file_name") != model_file:
            continue
        if is_doomed(other, data):
            continue
        return True
    return False


def delete_files(url, dir):
    """Remove one url's model, and ONLY what no other url still needs.

    THE FILE IS SHARED NOW, AND IT DID NOT USED TO BE. A model is stored under
    the sha256 of its bytes and each sidecar under the sha256 of its url, so
    several sidecars legitimately name one file — that is what adoption and
    deduplication both produce. Before the content rename ran on every
    completion path (2026-09-17) large models kept url-derived names, so one
    url meant one file and deleting by name was accidentally safe. It is not
    any more.

    This used to be two sweeps and both were wrong once the file became shared:

    1. `if url_hash in f: delete` — a substring match over `os.walk`, with no
       check of who else points at the file. A legacy `<base>_<url hash><ext>`
       that other urls have since ADOPTED was deleted out from under them.
    2. the referrer-aware sweep, which only ran for sidecars it could still
       find — and sweep 1 had already deleted `<url hash>.json`. For a plain
       Hugging Face url, where the effective url is the base url, that is every
       time: the sidecar went, sweep 2 then matched nothing, and the
       content-named file was LEAKED on disk forever.

    So it is one pass now. Work out which sidecars this url owns, read the
    files they name BEFORE deleting anything, and delete a file only when no
    SURVIVING sidecar still names it.
    """
    log_path = Path(dir) / "expunge_log.txt"
    error_path = Path(dir) / "error.txt"
    # Compute hash early and log only the hash to avoid leaking sensitive query params
    url_hash = hash_string(url)
    with open(log_path, "a") as log:
        log.write(f"delete request received, url_hash={url_hash}\n")

    deleted_dirs = set()

    def remove(path, what):
        try:
            delete_file_and_cleanup_dir(Path(path), dir)
            with open(log_path, "a") as log:
                log.write(f"Deleted {what}: {path}\n")
        except Exception as e:
            with open(error_path, "a") as err:
                err.write(f"Failed to delete {what}: {path} - {e}\n")
        deleted_dirs.add(os.path.dirname(path))

    for root, _, files in os.walk(dir):
        sidecars = [f for f in files if f.endswith(".json")]

        # Which sidecars does this url own? By name, which is the effective
        # url's hash, or by the base url they stored.
        doomed = {}
        for f in sidecars:
            data = read_sidecar(os.path.join(root, f))
            if f == f"{url_hash}.json" or (data and sidecar_url_matches(data.get("url"), url)):
                doomed[f] = data or {}

        # Everything a SURVIVING sidecar names is off limits, whatever else
        # says otherwise. Read before deleting: a name collected after the fact
        # is a name read out of a file that is already gone.
        spoken_for = set()
        for f in sidecars:
            if f in doomed:
                continue
            data = read_sidecar(os.path.join(root, f))
            if data and data.get("file_name"):
                spoken_for.add(data["file_name"])

        def is_doomed(filename, _data):
            return filename in doomed

        for f, data in doomed.items():
            model_file = data.get("file_name")
            if model_file and model_file_is_spoken_for(root, model_file, is_doomed):
                with open(log_path, "a") as log:
                    log.write(f"Keeping model file, another url still names it: {model_file}\n")
            elif model_file:
                model_path = os.path.join(root, model_file)
                if os.path.exists(model_path):
                    remove(model_path, "model file")
                part = part_path_for(model_path)
                if os.path.exists(part):
                    remove(part, "partial download")
                marker = completion_marker_for(part)
                if os.path.exists(marker):
                    remove(marker, "completion record")
            remove(os.path.join(root, f), "sidecar JSON")

        # An orphan sweep for what this url left behind before sidecars
        # recorded content hashes: files still wearing the url hash in their
        # name, with no sidecar of their own. Referrer-checked like everything
        # else, because those are exactly the files adoption reuses.
        for f in files:
            if f.endswith(".json") or f in doomed or f in spoken_for:
                continue
            if url_hash not in f:
                continue
            orphan = os.path.join(root, f)
            if os.path.exists(orphan):
                remove(orphan, "file named by this url")

    # After all deletions, check and remove empty parent directories
    for d in deleted_dirs:
        parent = Path(d)
        if parent.exists() and parent.is_dir() and not any(parent.iterdir()):
            try:
                parent.rmdir()
                with open(log_path, "a") as log:
                    log.write(f"fetch.py: Deleted empty output directory: {parent}\n")
            except Exception as e:
                with open(error_path, "a") as err:
                    err.write(f"fetch.py: Failed to remove output directory: {parent} - {e}\n")


def part_path_for(file_path: str) -> str:
    """
    A DOWNLOAD IN PROGRESS MUST NOT WEAR THE NAME OF A FINISHED ONE.

    This wrote straight to the model's own filename and used the bytes already
    there as the resume offset, so an interrupted transfer left a file that
    LOOKED like the model to everything else in the system. A beta tester found
    `ltx-2-19b-dev-fp8.safetensors` on disk at 141 MB of a declared 27 GB —
    0.5% — sitting there looking valid. The downloader would have resumed it
    next time; ComfyUI, asked to load it in the meantime, fails on something
    that has nothing to do with the real cause.

    So the bytes accumulate in `<name>.part` and the final name is given only
    when the size matches what the sidecar declares. Resume still works: it
    resumes from the `.part`. Nothing else in the system can mistake a `.part`
    for a model.
    """
    return file_path + ".part"


def completion_marker_for(part_path: str) -> str:
    """Where a downloader records that it wrote the LAST byte of a part file."""
    return part_path + ".complete"


def mark_part_complete(part_path: str, total_size) -> None:
    """Record that this part file was written to its end.

    SIZE IS NOT EVIDENCE, and for a parallel download it never was: the part
    file is pre-allocated to the final length before the first byte arrives
    (`AsyncParallelDownloader.download_async`), so from the first instant to
    the last it wears the right size and the wrong contents. On fmt-5000,
    2026-09-17, ComfyUI died mid-transfer and the next run adopted a 12 GB
    z-image weight of exactly the declared 12,309,866,400 bytes whose sha256
    was `50638dd8...` where Hugging Face states `24076130...`; the card
    rendered uniform noise and reported success.

    The downloader is the only thing that KNOWS, and it used to print the fact
    ("Parallel download completed successfully") and throw it away. This is
    that fact, written where the next process can read it.
    """
    marker = completion_marker_for(part_path)
    try:
        with open(marker, "w") as f:
            json.dump({"size": total_size, "completed_at": time.time()}, f)
            f.flush()
            os.fsync(f.fileno())
    except OSError as e:
        # A completion we failed to record costs a re-download, which is the
        # safe direction to fail in. It is worth a line, not an exception.
        print(f"[ANYMATIX DOWNLOAD] Could not record the completion of "
              f"{os.path.basename(part_path)}: {e}")


def clear_part_completion(part_path: str) -> None:
    """Forget any recorded completion for this part file.

    Called wherever the bytes change or go: a marker that outlives the bytes it
    describes is worse than none, because the next run believes it.
    """
    try:
        os.remove(completion_marker_for(part_path))
    except OSError:
        pass
    # The same goes for how far each segment got.
    clear_segment_journal(part_path)


# How often a segment writer flushes and records how far it got. Small enough
# that a kill costs little (8 segments x 32 MB, at most, re-fetched), large
# enough that the journal is rewritten a few times a second at most.
SEGMENT_JOURNAL_STRIDE = 32 * 1024 * 1024


def segment_journal_for(part_path: str) -> str:
    """Where a parallel download records how far each of its segments got."""
    return part_path + ".segments"


def clear_segment_journal(part_path: str) -> None:
    try:
        os.remove(segment_journal_for(part_path))
    except OSError:
        pass


def read_segment_journal(part_path: str, total_size) -> Optional[dict]:
    """The journal of an interrupted parallel download of THIS part file, if
    it can be trusted: same declared size, a part file of that size beside it.
    Anything else is None, and the download starts from zero as it always did."""
    data = read_sidecar(segment_journal_for(part_path))
    if not data or total_size is None:
        return None
    try:
        if int(data.get("size", -1)) != int(total_size):
            return None
        segment_size = int(data["segment_size"])
        done = {int(k): int(v) for k, v in (data.get("done") or {}).items()}
    except (KeyError, TypeError, ValueError):
        return None
    if segment_size <= 0 or any(v < 0 for v in done.values()):
        return None
    try:
        if os.path.getsize(part_path) != int(total_size):
            return None
    except OSError:
        return None
    return {"segment_size": segment_size, "done": done}


class SegmentJournal:
    """HOW FAR EACH SEGMENT GOT, WRITTEN WHERE THE NEXT PROCESS CAN READ IT.

    `bugs/a-dropped-ssh-link-remote-machine-kills`. A parallel download writes
    eight segments at their offsets into a pre-allocated part file and, until
    this, recorded nothing until every one of them had finished. A process that
    died mid-way — the remote's dead-man switch after a long link outage, a
    Stop, an OOM — left a full-length file with holes and no record of where
    they were, so the next run could only hash it (minutes for 12 GB on a
    rotational disk), find it wrong, and fetch every byte again. Measured on
    pc-ciancia, 2026-10-06: `flux1-dev-kontext_fp8_scaled`, 11.9 GB at
    ~12 MB/s, restarted from `0.00/11.9G` after the switch fired at 19:03:05Z.

    Each segment is written IN ORDER from its start, so how far it got is one
    number. The writer flushes before reporting it, so the bytes below the
    recorded offset are in the kernel even if the process dies the next
    instant; the journal is replaced atomically, so it is never half-written.
    """

    def __init__(self, part_path: str, total_size: int, segment_size: int,
                 segments, landed: dict):
        self.part_path = part_path
        self.total_size = total_size
        self.segment_size = segment_size
        self.starts = {sid: s for sid, s, e in segments}
        self.lengths = {sid: e - s + 1 for sid, s, e in segments}
        self.done = {sid: min(landed.get(sid, 0), self.lengths[sid]) for sid in self.starts}
        self.finished = {sid for sid in self.starts if self.done[sid] >= self.lengths[sid]}
        self._write()

    def landed(self, segment_id: int, offset: int) -> None:
        self.done[segment_id] = max(self.done.get(segment_id, 0), offset - self.starts[segment_id])
        if self.done[segment_id] >= self.lengths[segment_id]:
            self.finished.add(segment_id)
        self._write()

    def _write(self) -> None:
        path = segment_journal_for(self.part_path)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump({"size": self.total_size, "segment_size": self.segment_size,
                           "done": {str(k): v for k, v in self.done.items()}}, f)
            os.replace(tmp, path)
        except OSError as e:
            # A journal we could not write costs a restart from zero — what
            # happened before it existed. Worth a line, not a failed download.
            print(f"[ANYMATIX DOWNLOAD] Could not record segment progress for "
                  f"{os.path.basename(self.part_path)}: {e}")


def part_completion_is_recorded(part_path: str, expected_size) -> bool:
    """Whether the downloader that wrote this part file said it finished it."""
    data = read_sidecar(completion_marker_for(part_path))
    if not data:
        return False
    if expected_size is None:
        return True
    try:
        return int(data.get("size", -1)) == int(expected_size)
    except (TypeError, ValueError):
        return False


def resumed_part_is_the_whole_file(part_file: str, expected_size, remote_sha256,
                                   label: str,
                                   progress: Optional[Callable[[int, Optional[int]], None]] = None) -> bool:
    """Whether a full-size `.part` found on disk really holds the file.

    The question this replaces was `os.path.getsize(part) == file_size`, which
    is true of a parallel download from its first byte onwards. Two things can
    answer it honestly, and nothing else can:

    1. THE SERVER'S OWN HASH, where it stated one — `X-Linked-Etag` at Hugging
       Face, `hashes.SHA256` in Civitai's metadata. This is the same proof
       `adopt_existing_file` performs on a candidate before reusing it, and a
       resumed part file deserves no less. It outranks the marker: a marker
       says the bytes all arrived, a hash says they are the right bytes.
    2. THE DOWNLOADER'S OWN RECORD (`mark_part_complete`), for the servers that
       state nothing. It covers the case the old size test was written for —
       the process died between the last byte and the rename.

    Neither available means the file is not provable, and an unprovable weight
    is how a card comes to render static. The caller discards it and downloads
    again; a SHORT part file is untouched by all this and still resumes from
    its real prefix, which is the whole reason the `.part` scheme exists.

    THE RECORD IS ASKED FIRST since 2026-09-23 ("skip verification if size
    matches"). For a part file the size is no evidence at all — it is
    pre-allocated — so the downloader's completion record is what "the size
    matches" means here: it says every byte of the declared length arrived. It
    used to be outranked by the server's hash, which cost a full sha256 pass
    (minutes, on a multi-GB weight) every time a process died between the last
    byte and the rename. A part file with NO record is still hashed when the
    server states a hash: that is the pre-allocated file of a killed transfer,
    exactly the fmt-5000 case above, and it is not relaxed.
    """
    if part_completion_is_recorded(part_file, expected_size):
        print(f"[ANYMATIX DOWNLOAD] The downloader recorded {label} as complete before it died")
        return True
    if remote_sha256:
        try:
            actual = compute_file_sha256(part_file, progress=progress).lower()
        except OSError as e:
            print(f"[ANYMATIX DOWNLOAD] Could not hash the resumed part file for {label}: {e}")
            return False
        if actual == remote_sha256:
            print(f"[ANYMATIX DOWNLOAD] The resumed part file for {label} hashes to "
                  f"{actual}, which is what the server states")
            return True
        print(f"[ANYMATIX DOWNLOAD] The resumed part file for {label} hashes to {actual}, "
              f"not the {remote_sha256} the server states - discarding it")
        return False
    print(f"[ANYMATIX DOWNLOAD] A full-size part file for {label} with no completion record "
          f"and no server hash to check it against proves nothing - downloading again")
    return False


def finalize_download(part: str, file_path: str, expected_size, label: str) -> str:
    """
    Give the `.part` its real name, and only if it earned it.

    A short file KEEPS its `.part` and raises: those bytes are worth resuming
    from, and deleting them would throw away what the download already paid
    for. `os.replace` is atomic on the same filesystem, so no reader ever sees
    a half-named file.
    """
    if not os.path.exists(part):
        raise Exception(f"Download produced no data for {label}")
    written = os.path.getsize(part)
    if expected_size is not None and written != expected_size:
        raise Exception(
            f"Incomplete download for {label}: {written} of {expected_size} bytes. "
            f"Kept as {os.path.basename(part)} to resume from."
        )
    os.replace(part, file_path)
    # The bytes have a name now; the note saying they were finished has nothing
    # left to describe, and a stale one would vouch for the NEXT part file
    # written at this path.
    clear_part_completion(part)
    return file_path


def download_file(url, dir, callback: Optional[Callable[[int, Optional[int]], None]] = None, expand_info: Optional[Callable[[str], dict | None]] = None, effective_url: Optional[str] = None, redact_append: Optional[str] = None, adopt_dirs: Optional[List[str]] = None, phase: Optional[Callable[[str], None]] = None):
    """Return the path of the model for `url`, fetching it only if the machine
    has not got it.

    `adopt_dirs` are extra directories that may already hold the bytes and are
    read but never written into — the durable models dir, when `dir` is the
    NVMe cache. The returned path may be in one of them, so a caller that does
    something with `dir` afterwards (mirroring a cache entry, say) must check
    where the file actually came back from.
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library is required for downloading")

    # WHAT THE WAIT IS, SAID BEFORE THE FIRST BYTE OF IT IS REPORTED.
    #
    # This function does three things a person can sit through, and until now
    # the screen had one word for all of them. Vincenzo watched FETCH CHECKPOINT
    # at 0% with an empty bar and had to ask what the machine was doing, twice
    # in one day; the answer both times was that it was hashing a file it
    # already had, which is the 2026-09-17 adoption work doing its job.
    #
    #   fetching   bytes are coming over the network
    #   verifying  a file already on disk is being hashed to prove it is the
    #              bytes this url serves - the adoption candidate, a resumed
    #              part file, or what a download just wrote
    #   adopting   the proof passed and the file is taken over; no bytes, no wait
    #
    # `phase` is announced BEFORE the progress it describes, never after, so a
    # bar cannot be driven under the label of the thing that finished before it.
    def in_phase(name: str) -> None:
        if phase:
            phase(name)

    def verifying(done, total):
        """A hash reports on the same channel a download does — and announces
        its phase from its own FIRST tick, not in advance.

        Announcing before the call put VERIFYING on screen for an adoption that
        found no candidate and hashed nothing: a label for work that did not
        happen, which is the same defect as a label for work misnamed. Every
        hash opens with `(0, total)`, so the first tick IS the start of real
        work and there is no other way to reach this.
        """
        if done == 0:
            in_phase("verifying")
        if callback:
            callback(done, total)

    effective = effective_url or url
    print("download file", redact_url(effective, redact_append), dir)
    url_hash = hash_string(effective)
    os.makedirs(dir, exist_ok=True)
    store_path = os.path.join(dir, f"{url_hash}.json")

    # A URL THIS MACHINE HAS ALREADY FETCHED MAKES NO REQUEST AT ALL.
    #
    # The sidecar is named `<url hash>.json` INSIDE `dir`, and an adoption
    # writes it beside the file it names — which, on a pod, is the durable
    # volume while `dir` is the NVMe cache — and removes the one in `dir`,
    # because a sidecar in the cache dies with the container and leaves an
    # orphan pointing at nothing. So the very next call for the same url with
    # the same `dir` found no sidecar, went to `fetch_headers` (a request), and
    # then re-hashed the whole weight to adopt it again. Every run: ~23 s per
    # 12 GB, and a network round trip a fully-downloaded card should not need.
    #
    # It did not bite in the shipped app only because
    # `anymatix_checkpoint_fetcher` checks for the sidecar on the volume before
    # it chooses a cache dir — a caller keeping a fetcher's promise, which
    # holds exactly until something calls the fetcher directly.
    #
    # Looking here costs one `stat` per adopt dir on the cold path.
    if not os.path.exists(store_path):
        for other_dir in (adopt_dirs or []):
            if not other_dir or os.path.abspath(other_dir) == os.path.abspath(dir):
                continue
            cached = read_sidecar(os.path.join(other_dir, f"{url_hash}.json"))
            if not cached:
                continue
            satisfied = satisfied_by_sidecar(other_dir, cached)
            if satisfied:
                print(f"[ANYMATIX] {os.path.basename(satisfied)} is already here, "
                      f"recorded for url hash {url_hash}: nothing to fetch")
                return satisfied

    parsed_url = urlparse(effective)
    file_name_default = parsed_url.path.split('/')[-1].split('?')[0]
    # Always persist base URL, never include token-bearing URL
    data = {"url": url}

    with requests.Session() as session:

        if (os.path.exists(store_path)):
            print("loading json", store_path)
            with open(store_path, 'r') as contents:
                data.update(json.load(contents))
            # A sidecar with no `file_size` is one whose server stated no
            # length (`remembered_size`, `satisfied_by_sidecar`). Sidecars
            # written before the key existed omit it rather than storing None,
            # and every read below indexes it: say it once, here.
            data.setdefault("file_size", None)
        else:
            print("fetching headers", redact_url(effective, redact_append))
            data.update(fetch_headers(effective, session))
            if data["file_name"] is None:
                data["file_name"] = f"{file_name_default}"
            f = data["file_name"]
            data["name"] = f # Keep full original filename
            x = f.rsplit(".", 1)
            data["file_name"] = f"{x[0]}_{url_hash}" + \
                ('.' + x[1] if len(x) > 1 else "")
            if expand_info:
                try:
                    info = expand_info(url)
                    if info is not None:
                        data["data"] = info
                except Exception as e:
                    print(f"[WARNING] Failed to fetch model info (non-critical): {e}")
                    print(f"[WARNING] Model download will continue without metadata")
                    # Continue with download anyway - metadata is not essential
            with open(store_path, 'w') as file:
                json.dump(data, file, indent=4)

        # The content hash Civitai states in the model metadata, if there is one.
        metadata_hash = None
        if "data" in data and isinstance(data["data"], dict):
            # Check for hashes in Civitai-style metadata
            if "hashes" in data["data"] and isinstance(data["data"]["hashes"], dict):
                metadata_hash = data["data"]["hashes"].get("SHA256", "").lower()
            elif "files" in data["data"] and isinstance(data["data"]["files"], list):
                # Civitai often has a list of files
                for f in data["data"]["files"]:
                    if "hashes" in f and isinstance(f["hashes"], dict):
                        metadata_hash = f["hashes"].get("SHA256", "").lower()
                        if metadata_hash: break

        file_path = os.path.join(dir, data["file_name"])

        # What the bytes should hash to, if anybody told us: Civitai says it in
        # the model metadata, Hugging Face in `X-Linked-Etag`. Either way it is
        # the only thing that can identify a file we already have, because the
        # url cannot: the url is the sidecar's name, not the file's.
        # `sha256` last: on a warm sidecar it is what a previous run recorded
        # for THIS url, which is still a correct identity for the file if the
        # file itself has since been renamed or moved away.
        target_sha256 = metadata_hash or data.get("remote_sha256") or data.get("sha256")
        if target_sha256:
            data["sha256"] = target_sha256  # Pre-set it

        # WHAT THE SERVER ITSELF STATED, and nothing of our own. `target_sha256`
        # falls back to `data["sha256"]`, which is a hash THIS code measured on
        # a previous run — fine for finding a file we already have, useless as a
        # check on bytes we are about to accept, because a corrupt file that was
        # once canonicalised under its own measurement would then verify against
        # itself forever. Only `metadata_hash` (Civitai) and `remote_sha256`
        # (Hugging Face's `X-Linked-Etag`) are somebody else's claim about the
        # bytes, so only they can catch us being wrong.
        remote_stated_sha256 = metadata_hash or data.get("remote_sha256")

        local_file_size = 0
        part_file = part_path_for(file_path)

        def finish_download():
            """EVERY FINISHED DOWNLOAD LEAVES THE SAME THING BEHIND, and until
            2026-09-17 two of the three did not.

            A file is named by the sha256 of its BYTES; only the sidecar is
            named by the url. That rename lived at the bottom of this function,
            and both the parallel path and the resumed-`.part` path
            `return`ed straight out of `finalize_download` before reaching it —
            so a model fetched in parallel, which is every large model, stayed
            on disk under `<base>_<URL hash><ext>` and its sidecar never
            recorded a `sha256` at all.

            Measured live against Hugging Face on 2026-09-17: fetching
            `krea2_darkbrush.safetensors` left it named after the url hash
            `f77bdb3e...`, not its content hash `f47c4316...`.

            That is why repointing a url could not be recovered from: there was
            no content-named file to find and no hash in the sidecar to find it
            by. The whole store was keyed on the url twice over.
            """
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Download completed but file not found: {file_path}. "
                    f"This may indicate a download failure, filesystem issue, or the file was deleted during download."
                )

            if data["file_size"] is not None:
                actual_size = os.path.getsize(file_path)
                if actual_size != data["file_size"]:
                    # Self-heal: remove the bad file so the next run starts clean instead
                    # of resuming/appending onto it again (the re-download-forever loop).
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
                    raise Exception(
                        f"Downloaded file size mismatch for {data['file_name']}: "
                        f"expected {data['file_size']} bytes, got {actual_size} bytes. "
                        f"The corrupted file was removed and will be re-downloaded on the next run."
                    )

            if file_path.lower().endswith(".json") and not is_valid_json_file(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
                raise Exception(
                    f"Downloaded JSON file is malformed for {data['file_name']}. "
                    f"The corrupted file was removed and will be re-downloaded on the next run."
                )

            # THE LENGTH THE SERVER STATED IS THE PROOF (2026-09-23).
            #
            # Vincenzo watched a fresh download sit on "VERIFYING MODEL 9%" —
            # this hash, over every byte just written — and called it a
            # showstopper: "for now do skip verification if size matches". The
            # size was checked just above against the server's Content-Length,
            # and `finalize_download` only names a part file once the
            # downloader has written its whole declared length.
            #
            # What is given up, deliberately: the comparison with the hash the
            # server states (`remote_stated_sha256`) and the content-addressed
            # rename. The file KEEPS its url-hash name, because a name that
            # asserts a content sha256 may only be produced by computing one —
            # the rule `library-asset-hash-integrity` rests on, and the reason
            # adoption re-hashes whatever it finds under such a name. The
            # sidecar says `verification: "size"` so nothing reads this as
            # hashed, and the Models manager can offer the hash later.
            if data["file_size"] is not None:
                print(f"[ANYMATIX] {data['file_name']} has the {data['file_size']} bytes the "
                      f"server stated: accepted on its size, not hashed")
                data["verification"] = VERIFIED_BY_SIZE
                data["verified_size"] = data["file_size"]
                with open(store_path, 'w') as file:
                    json.dump(data, file, indent=4)
                return file_path

            # POST-DOWNLOAD DEDUPLICATION — only for a file whose server stated
            # no length, so there was nothing to compare it with.
            print(f"[ANYMATIX] Computing hash for deduplication: {file_path}")
            sha256 = compute_file_sha256(file_path, progress=verifying).lower()

            # A SIDECAR MUST NEVER STATE A HASH THE SERVER DID NOT.
            #
            # Where the server told us what the bytes hash to and ours do not
            # match, the file is wrong — and naming it by our own measurement
            # would turn a fault anything could still detect into the store's
            # permanent idea of what this url serves. That is precisely what
            # happened to the 12 GB z-image weight on fmt-5000 on 2026-09-17:
            # recorded as `50638dd8...` where Hugging Face states `24076130...`,
            # and served to the card, which rendered static.
            if remote_stated_sha256 and sha256 != remote_stated_sha256:
                try:
                    os.remove(file_path)
                except Exception:
                    pass
                raise Exception(
                    f"Downloaded file does not match the hash the server states for "
                    f"{data['file_name']}: got {sha256}, expected {remote_stated_sha256}. "
                    f"The file was removed and will be downloaded again on the next run."
                )

            data["sha256"] = sha256
            data["verification"] = VERIFIED_BY_SHA256
            # MEMORIZE THE SIZE: the server stated none, so the next run would
            # otherwise have nothing but a hash to recognise this file by.
            data["verified_size"] = os.path.getsize(file_path)

            canonical_name = canonical_model_name(data["file_name"], sha256)
            canonical_path = os.path.join(dir, canonical_name)

            if os.path.exists(canonical_path) and canonical_path != file_path:
                print(f"[ANYMATIX] Deduplicated model found: {canonical_path}. Reusing.")
                os.remove(file_path)
                data["file_name"] = canonical_name
            else:
                print(f"[ANYMATIX] New unique model. Naming: {canonical_name}")
                os.rename(file_path, canonical_path)
                data["file_name"] = canonical_name

            # Save sidecar with canonical filename and hash
            with open(store_path, 'w') as file:
                json.dump(data, file, indent=4)

            print("Model name:", data["file_name"])

            return os.path.join(dir, data["file_name"])

        if data["file_size"] is not None:
            # THE FINAL NAME MEANS FINISHED. A file wearing it whose size is not
            # the declared one cannot be the model — under the .part scheme it
            # can only be an artifact of the old one — so it is not resumed
            # from, it is removed.
            if os.path.exists(file_path):
                on_disk = os.path.getsize(file_path)
                if on_disk == data["file_size"]:
                    return file_path
                print(
                    f"[ANYMATIX DOWNLOAD] Discarding a file with the final name and the wrong "
                    f"size for {data['file_name']}: {on_disk} != expected {data['file_size']} bytes"
                )
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            if os.path.exists(part_file):
                local_file_size = os.path.getsize(part_file)
                if local_file_size == data["file_size"]:
                    # A FULL-SIZE PART FILE IS A QUESTION, NOT AN ANSWER.
                    #
                    # This used to read "it was complete and nobody renamed it",
                    # which is true of a serial download and meaningless for a
                    # parallel one: that path pre-allocates the part file to the
                    # final length before the first byte arrives, so a transfer
                    # killed at 1% leaves a file of exactly the right size.
                    # Every large model takes the parallel path.
                    # bugs/a-parallel-download-pre-allocates-part-file-so
                    #
                    # UNLESS IT CARRIES A JOURNAL: then it is a parallel
                    # download that was killed, and the journal says exactly
                    # which bytes landed (`SegmentJournal`). It is neither
                    # hashed nor discarded; the parallel path below carries on
                    # each segment from where it stopped.
                    if not part_completion_is_recorded(part_file, data["file_size"]) \
                            and read_segment_journal(part_file, data["file_size"]) is not None:
                        print(f"[ANYMATIX DOWNLOAD] {data['file_name']}: an interrupted parallel "
                              f"download is on disk with its segment journal - resuming it")
                        local_file_size = 0
                    elif resumed_part_is_the_whole_file(
                        part_file, data["file_size"], remote_stated_sha256,
                        data["file_name"], progress=verifying
                    ):
                        finalize_download(
                            part_file, file_path, data["file_size"], data["file_name"]
                        )
                        return finish_download()
                    else:
                        # Unprovable, and there is no prefix to salvage: the holes
                        # of a dead parallel download are wherever its segments were
                        # not, so the only honest offset to resume from is zero.
                        try:
                            os.remove(part_file)
                        except Exception:
                            pass
                        clear_part_completion(part_file)
                        local_file_size = 0
                if local_file_size > data["file_size"]:
                    # Self-heal: a partial larger than the target is corrupt — almost
                    # always a prior resume where the server ignored our Range header and
                    # the full body got appended onto the partial. Discard and start fresh
                    # (otherwise it grows every run and never matches → re-downloads forever).
                    print(
                        f"[ANYMATIX DOWNLOAD] Discarding oversized/corrupt partial for "
                        f"{data['file_name']}: {local_file_size} > expected {data['file_size']} bytes"
                    )
                    try:
                        os.remove(part_file)
                    except Exception:
                        pass
                    clear_part_completion(part_file)
                    local_file_size = 0
        elif data.get("verified_size") is not None and os.path.isfile(file_path) \
                and os.path.getsize(file_path) == data["verified_size"]:
            # No length from the server, but one memorized when this file was
            # proven: a `stat`, not a hash, exactly as for a stated length.
            return file_path
        elif data["file_size"] is None and os.path.exists(file_path) and file_path.lower().endswith(".json"):
            if is_valid_json_file(file_path):
                return file_path
            print(f"[ANYMATIX DOWNLOAD] Removing malformed cached JSON before re-download: {file_path}")
            os.remove(file_path)

        # ADOPTION — the last thing tried before spending bandwidth.
        #
        # A cached file is named by the sha256 of its BYTES; its sidecar is
        # named by the sha256 of the URL. So repointing a url — which is what
        # `a8f13a50b` did to all 75 shipped Hugging Face urls on 2026-09-16,
        # moving them from `/resolve/main/` to `/resolve/<commit>/` — changes
        # the sidecar's name and nothing else. The file was still on disk under
        # its content name; nothing looked for it there, so every machine
        # downloaded its whole model set again (measured on fmt-5000,
        # 2026-09-17: z-image turbo re-fetched from zero).
        #
        # The cache key is NOT normalised to drop the revision, and must not
        # be: two revisions of one repo path may serve different bytes, and a
        # key that ignored the revision would hand back the wrong model in
        # silence. What is stable across revisions is the CONTENT hash, and
        # that is what is matched here — the server states it, and the file on
        # disk is re-hashed to prove it before anything is adopted.
        #
        # This sits after the `.part` and size checks so that a warm run, where
        # our own file is already present under our own name, returns above
        # without hashing anything.
        if target_sha256:
            adopted = adopt_existing_file(
                [dir] + list(adopt_dirs or []),
                canonical_model_name(data["file_name"], target_sha256),
                target_sha256,
                data.get("file_size"),
                f"{url_hash}.json",
                data["file_name"],
                progress=verifying,
            )
            if adopted:
                in_phase("adopting")
                # The adopted file KEEPS ITS NAME. Renaming it to the canonical
                # content name would heal the store, but other sidecars may
                # already point at the old name and would be orphaned by it.
                # Recording the hash in our own sidecar is enough: the next
                # fetch of this url finds the file by name and size and returns
                # above without hashing anything, so the verification is paid
                # once per url, not once per run.
                data["file_name"] = os.path.basename(adopted)
                # Adoption always hashes (it trusts no name and no size), so
                # this file is hash-proven and its length can be memorized.
                data["verification"] = VERIFIED_BY_SHA256
                data["verified_size"] = os.path.getsize(adopted)
                # The sidecar lives beside the file it names, or it dies with a
                # container the file survives — and leaves an orphan behind
                # pointing at nothing.
                adopted_store = os.path.join(os.path.dirname(adopted), f"{url_hash}.json")
                with open(adopted_store, 'w') as file:
                    json.dump(data, file, indent=4)
                if os.path.abspath(adopted_store) != os.path.abspath(store_path) \
                        and os.path.exists(store_path):
                    os.remove(store_path)
                # A partial for bytes we now hold is dead weight.
                if os.path.exists(part_file):
                    os.remove(part_file)
                clear_part_completion(part_file)
                return adopted

        downloaded_size = local_file_size
        in_phase("fetching")

        # PARALLEL DOWNLOAD ATTEMPT — fresh downloads only
        parallel_success = False
        parallel_exception = None
        if local_file_size == 0 and data["file_size"] is not None:  # Only for fresh downloads
            print(f"[ANYMATIX DOWNLOAD] Attempting parallel download for {data['file_name']} ({data['file_size']} bytes)")
            try:
                parallel_success = fetch_parallel(
                    effective,
                    part_file,
                    callback,
                    local_file_size,
                    max_connections=min(8, max(2, data["file_size"] // (10*1024*1024)))  # Adaptive connections
                )
                if parallel_success:
                    mb_total = data["file_size"] / (1024 * 1024) if data["file_size"] else 0
                    print(f"[ANYMATIX DOWNLOAD] Parallel download completed successfully: {data['file_name']} ({mb_total:.0f}MB)")
                    finalize_download(
                        part_file, file_path, data["file_size"], data["file_name"]
                    )
                    return finish_download()
                else:
                    print(f"[ANYMATIX DOWNLOAD] Parallel download was attempted but returned False (likely server doesn't support ranges)")
            except Exception as e:
                if is_interrupt(e):
                    raise
                print(f"[ANYMATIX DOWNLOAD] Parallel download failed with exception, falling back to a single stream: {e}")
                parallel_success = False
                parallel_exception = e

        # SINGLE-STREAM FALLBACK — also the resume path: one ordered stream from
        # the byte we stopped at, which segments cannot express.
        if not parallel_success:
            # The parallel attempt may have left a resumable prefix behind (see
            # AsyncParallelDownloader), and `local_file_size` was measured
            # before it ran. Without re-reading it here the fallback opens the
            # file 'wb' and truncates exactly the bytes we just kept.
            #
            # IT LOOKED IN THE WRONG PLACE, SO IT SALVAGED NOTHING.
            #
            # This read `file_path` — the model's FINAL name. The parallel
            # attempt never writes there: `download_file` hands it `part_file`,
            # so `AsyncParallelDownloader.file_path` IS the part path and the
            # prefix it keeps is left at `<name>.part`. The final name does not
            # exist at this point (it is removed above when its size is wrong),
            # so the check was always false, `local_file_size` stayed 0, the
            # open below took mode 'wb', and it truncated the very bytes the
            # salvage had just been careful to keep. A download interrupted at
            # 18% started again from zero, every time.
            #
            # Same family as the `.part.part` bug fixed in 4383c3da: one path
            # built twice, one step further down the same road.
            # A JOURNALED PART FILE IS NOT A PREFIX. It is full-length with
            # holes, and appending to it from its size would corrupt it. If the
            # parallel attempt failed before it could carry on, the journal is
            # kept for the next run and this one says why; if the server simply
            # does not do ranges, nothing can resume it and it goes.
            if read_segment_journal(part_file, data["file_size"]) is not None:
                stall = stall_in(parallel_exception)
                if stall is not None:
                    raise DownloadStalled(
                        stalled_download_message(data["file_name"], stall), stall.reason
                    ) from parallel_exception
                if parallel_exception is not None:
                    raise Exception(
                        f"Download of {data['file_name']} interrupted; the bytes already on disk "
                        f"are kept and resume next time: {parallel_exception}"
                    ) from parallel_exception
                try:
                    os.remove(part_file)
                except OSError:
                    pass
                clear_part_completion(part_file)
                local_file_size = 0
                downloaded_size = 0
            if os.path.exists(part_file):
                salvaged = os.path.getsize(part_file)
                if salvaged > local_file_size:
                    print(f"[ANYMATIX DOWNLOAD] Resuming from the {salvaged} bytes the parallel attempt left")
                    local_file_size = salvaged
                    downloaded_size = salvaged
            print(f"[ANYMATIX DOWNLOAD] Using single-stream download for {data['file_name']}")
            single_stream_exception = None
            try:
                file_mode = 'ab' if local_file_size > 0 else 'wb'
                with open(part_file, file_mode) as file:
                    progress_bar = None
                    if TQDM_AVAILABLE and data["file_size"]:
                        try:
                            progress_bar = tqdm(total=data["file_size"], initial=local_file_size)
                        except:
                            progress_bar = None
                        
                    def cb(chunk):
                        nonlocal downloaded_size
                        if (chunk):
                            # Self-heal: if a resume blows past the expected size, the
                            # server ignored our Range request and is re-sending the
                            # whole file from byte 0 onto our partial. Abort NOW rather
                            # than appending the full body (which ballooned to 34GB);
                            # the failed/oversized file is removed below so the next run
                            # restarts clean and then caches correctly.
                            if (
                                data["file_size"] is not None
                                and downloaded_size + len(chunk) > data["file_size"]
                            ):
                                raise Exception(
                                    f"Download overshoot for {data['file_name']}: "
                                    f"server ignored the resume Range (re-sending full body). "
                                    f"Aborting to avoid an unbounded append; will restart clean."
                                )
                            file.write(chunk)
                            l = len(chunk)
                            downloaded_size += l
                            if progress_bar:
                                progress_bar.update(l)
                            if callback:
                                callback(downloaded_size, data["file_size"])
                            
                            # Additional console progress for anymatix terminal
                            if data["file_size"] and downloaded_size % (50 * 1024 * 1024) < l:  # Every 50MB
                                mb_downloaded = downloaded_size / (1024 * 1024)
                                mb_total = data["file_size"] / (1024 * 1024)
                                percent = (downloaded_size / data["file_size"]) * 100
                                print(f"[ANYMATIX PROGRESS] {mb_downloaded:.0f}MB / {mb_total:.0f}MB ({percent:.1f}%)")
                                
                    try:
                        # RESUMED FROM THE BYTE REACHED, A BOUNDED NUMBER OF
                        # TIMES. The file stays open, so each attempt appends
                        # where the last one stopped and asks the server for
                        # exactly that range.
                        attempts = max(1, DOWNLOAD_ATTEMPTS)
                        attempt = 0
                        warmups = 0
                        while attempt < attempts:
                            try:
                                fetch(effective, session, cb, downloaded_size)
                                break
                            except RangeIgnored as e:
                                # A COLD FILE FIRST, A SERVER WITHOUT RANGES
                                # ONLY AFTER THE WARM-UP (`RANGE_WARMUP_PAUSES`).
                                # Neither is an attempt: nothing failed.
                                if warmups < len(RANGE_WARMUP_PAUSES):
                                    print(f"[ANYMATIX DOWNLOAD] {e} on resume at byte {downloaded_size}; "
                                          f"asking again in {RANGE_WARMUP_PAUSES[warmups]}s")
                                    time.sleep(RANGE_WARMUP_PAUSES[warmups])
                                    warmups += 1
                                    check_interrupted()
                                    continue
                                # The fallback: the server will not resume, so
                                # this run starts the file over instead of
                                # failing on it.
                                print(f"[ANYMATIX DOWNLOAD] The server keeps answering the resume of "
                                      f"{data['file_name']} with the whole file; starting it again from byte 0")
                                file.seek(0)
                                file.truncate()
                                downloaded_size = 0
                                if progress_bar:
                                    try:
                                        progress_bar.reset()
                                    except Exception:
                                        pass
                                continue
                            except DownloadTransportError as e:
                                file.flush()
                                # A link that never delivered a byte is a wrong
                                # address or no network at all: said at once,
                                # not after a minute of pauses.
                                if downloaded_size == 0:
                                    raise
                                if attempt + 1 >= attempts:
                                    reason = f"{e}, {attempts} attempts"
                                    raise DownloadStalled(
                                        stalled_download_message(data["file_name"], DownloadStalled(reason, reason)),
                                        reason
                                    ) from e
                                print(f"[ANYMATIX DOWNLOAD] Single stream stopped at byte {downloaded_size} ({e}); "
                                      f"attempt {attempt + 2} of {attempts} resumes there")
                                time.sleep(retry_pause_seconds(attempt))
                                check_interrupted()
                                attempt += 1
                    finally:
                        if progress_bar:
                            try:
                                progress_bar.close()
                            except:
                                pass
                        
                        # Final status message for anymatix terminal
                        if data["file_size"] is not None and downloaded_size == data["file_size"]:
                            mb_final = downloaded_size / (1024 * 1024)
                            print(f"[ANYMATIX DOWNLOAD] Single-stream download completed: {mb_final:.0f}MB")
                            # One ordered stream wrote every byte it was asked
                            # for. The rename is the next statement, so this
                            # note is only ever read when the process dies in
                            # between — which is the case the old size test was
                            # written for, and the only one it was right about.
                            mark_part_complete(part_file, data["file_size"])
                            
            except Exception as e:
                single_stream_exception = e
                print(f"[ANYMATIX DOWNLOAD] Single-stream download also failed: {e}")

                # Self-heal: drop the partial so the next run restarts clean — UNLESS
                # this is a user interrupt (then keep the partial for a real resume).
                # Nor when the link died after every attempt: the bytes that
                # arrived are good, and the next run resumes from them.
                keep_partial = is_interrupt(e) or isinstance(e, DownloadStalled)
                if not keep_partial:
                    try:
                        if os.path.exists(part_file):
                            os.remove(part_file)
                            clear_part_completion(part_file)
                            print(f"[ANYMATIX DOWNLOAD] Removed partial after failure: {part_file}")
                    except Exception:
                        pass

                # A link that died after every attempt is the whole story, and
                # its message already names the file.
                if isinstance(single_stream_exception, DownloadStalled):
                    raise single_stream_exception
                # If both the parallel and the single-stream attempt failed, raise the more serious exception
                if parallel_exception and single_stream_exception:
                    # Prefer the parallel exception when it says more; otherwise the single-stream one
                    if "Range" in str(parallel_exception) or "connection" in str(parallel_exception).lower():
                        raise parallel_exception
                    else:
                        raise single_stream_exception
                elif single_stream_exception:
                    raise single_stream_exception
                elif parallel_exception:
                    raise parallel_exception
                else:
                    raise Exception(f"Both the parallel and the single-stream download failed for {data['file_name']}")

        # THE RENAME, AND IT IS THE ONLY ONE.
        #
        # Everything above wrote into `<name>.part`. Reaching here means the
        # single-stream path ran, and this is where its bytes earn the model's
        # name. A short file raises and KEEPS its `.part`, so the next run
        # resumes instead of starting from zero.
        finalize_download(part_file, file_path, data["file_size"], data["file_name"])
        return finish_download()


def expand_info_civitai(url):
    # get the model id from the url using a regex that matches the first /.../ after https://civitai.com/api/download/models
    pattern = r'https://civitai\.com/api/download/models/([^/]+)'
    match = re.search(pattern, url)
    if match:
        model_id = match.group(1)
    else:
        return None
    model_info_url = f"https://civitai.com/api/v1/model-versions/{model_id}"
    
    try:
        with requests.Session() as session:
            response = requests.get(model_info_url, allow_redirects=True, timeout=30)
            
            # Check if the response is successful
            if response.status_code == 200:
                # Check if response has content
                if response.text.strip():
                    try:
                        return response.json()
                    except ValueError as json_error:
                        print(f"[WARNING] Failed to parse Civitai model info JSON for model {model_id}: {json_error}")
                        print(f"[WARNING] Response content (first 200 chars): {response.text[:200]}")
                        return None
                else:
                    print(f"[WARNING] Empty response from Civitai API for model {model_id}")
                    return None
            elif response.status_code == 404:
                print(f"[WARNING] Model {model_id} not found on Civitai (404)")
                return None
            elif response.status_code == 403:
                print(f"[WARNING] Access denied to model {model_id} on Civitai (403) - model may be private or require authentication")
                return None
            elif response.status_code == 429:
                print(f"[WARNING] Rate limited by Civitai API for model {model_id} (429) - too many requests")
                return None
            else:
                print(f"[WARNING] Civitai API returned status {response.status_code} for model {model_id}")
                return None
                
    except requests.exceptions.Timeout:
        print(f"[WARNING] Timeout while fetching model info from Civitai for model {model_id}")
        return None
    except requests.exceptions.ConnectionError as conn_error:
        print(f"[WARNING] Connection error while fetching model info from Civitai for model {model_id}: {conn_error}")
        return None
    except requests.exceptions.RequestException as req_error:
        print(f"[WARNING] Request error while fetching model info from Civitai for model {model_id}: {req_error}")
        return None
    except Exception as e:
        print(f"[WARNING] Unexpected error while fetching model info from Civitai for model {model_id}: {e}")
        return None


def expand_info(url):
    if url.startswith("https://civitai.com/api/download/models"):
        return expand_info_civitai(url)
    return None
