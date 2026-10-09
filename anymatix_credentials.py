"""
A SECRET IS NEVER A NODE INPUT.

Everything in a prompt is kept verbatim by ComfyUI: `GET /queue`, `GET /history`
and the frontend's workflow view all return it, and on a shared server anyone
who can reach the port reads them. The user's Civitai key used to travel as the
`auth` field of every Civitai url leaf, so a queue dump on a lab machine showed
it in clear text (anymatix bug "A remote ComfyUI's queue and history show the
user's Civitai token in clear text", 2026-10-09).

The key now arrives out of band: the app POSTs it to `/anymatix/credentials`
before every prompt, and it lives here, in this process's memory only. It is
never written to disk, never printed, and there is no route that reads it back.
The fetchers ask `civitai_auth_tail(url)` for the query tail they used to read
off their input, so the effective url -- and the sidecar named after it -- is
exactly what it was.
"""

import functools
import re
import threading
import traceback
from typing import Optional
from urllib.parse import urlparse

_lock = threading.Lock()
_civitai_token = ""

CIVITAI_DOWNLOAD_HOSTS = ("civitai.com", "civitai.red")


def set_civitai_token(token) -> None:
    """Replace the key. Anything that is not a non-empty string clears it."""
    global _civitai_token
    value = token.strip() if isinstance(token, str) else ""
    with _lock:
        _civitai_token = value


def has_civitai_token() -> bool:
    with _lock:
        return _civitai_token != ""


def civitai_auth_tail(url) -> Optional[str]:
    """`token=<key>` for a Civitai download url when a key is set; None otherwise.

    A url that already names a token is left alone (the caller strips it: a
    token inside a url is itself a secret in the prompt, and is never used).
    """
    try:
        host = urlparse(str(url or "")).hostname or ""
    except Exception:
        return None
    if host not in CIVITAI_DOWNLOAD_HOSTS:
        return None
    with _lock:
        token = _civitai_token
    return f"token={token}" if token else None


def strip_url_token(url) -> str:
    """The url with any `token=` query pair removed."""
    from urllib.parse import parse_qsl, urlencode, urlunparse

    text = str(url or "")
    try:
        p = urlparse(text)
        pairs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k != "token"]
        if len(pairs) == len(parse_qsl(p.query, keep_blank_values=True)):
            return text
        return urlunparse(p._replace(query=urlencode(pairs)))
    except Exception:
        return text


_URL_SECRET = re.compile(r"([?&](?:token|api_key|apikey|access_token)=)[^&\s\"'#)]+", re.IGNORECASE)


def redact_secrets(text) -> str:
    """`text` with every url credential pair, and the current key, masked.

    A failed download's exception names the url it asked for -- requests says
    "401 Client Error: Unauthorized for url: ...?token=<key>" -- and ComfyUI
    keeps the message in `/history` and prints the traceback to its console.
    """
    out = _URL_SECRET.sub(r"\1<redacted>", str(text))
    with _lock:
        token = _civitai_token
    if token:
        out = out.replace(token, "<redacted>")
    return out


def secrets_redacted(fn):
    """A node method whose failure can never carry a secret out.

    The effective url (base plus the `token=` tail) reaches requests, and
    requests puts it in every error it raises. When the formatted failure --
    message AND the exceptions chained under it, which ComfyUI's console
    traceback prints -- holds a secret, it is re-raised with the masked message
    and no chain. Anything else, ComfyUI's interrupt included, passes untouched.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            try:
                full = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            except Exception:
                full = str(e)
            if redact_secrets(full) == full:
                raise
            raise RuntimeError(redact_secrets(str(e))) from None

    return wrapper
