"""
A secret is never a node input (anymatix bug "A remote ComfyUI's queue and
history show the user's Civitai token in clear text", 2026-10-09).

The key reaches the fetchers through `anymatix_credentials`, set by
`POST /anymatix/credentials`; a failure that would carry it out is masked.
"""

import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import anymatix_credentials as creds  # noqa: E402

FAKE = "fake-civitai-key-0123456789abcdef"


@pytest.fixture(autouse=True)
def _clean():
    creds.set_civitai_token("")
    yield
    creds.set_civitai_token("")


def test_the_tail_comes_from_the_store_and_only_for_civitai():
    url = "https://civitai.com/api/download/models/123?type=Model"
    assert creds.civitai_auth_tail(url) is None
    creds.set_civitai_token(FAKE)
    assert creds.civitai_auth_tail(url) == f"token={FAKE}"
    assert creds.civitai_auth_tail("https://civitai.red/api/download/models/1") == f"token={FAKE}"
    assert creds.civitai_auth_tail("https://huggingface.co/x/y/resolve/main/z.safetensors") is None
    creds.set_civitai_token(None)
    assert creds.civitai_auth_tail(url) is None


def test_a_token_written_into_the_url_is_stripped():
    assert creds.strip_url_token("https://civitai.com/api/download/models/1?token=abc&type=Model") == (
        "https://civitai.com/api/download/models/1?type=Model"
    )
    plain = "https://civitai.com/api/download/models/1?type=Model"
    assert creds.strip_url_token(plain) == plain


def test_redaction_masks_url_pairs_and_the_held_key():
    creds.set_civitai_token(FAKE)
    msg = f"401 Client Error: Unauthorized for url: https://civitai.com/api/download/models/1?token={FAKE}"
    out = creds.redact_secrets(msg)
    assert FAKE not in out and "token=<redacted>" in out
    assert FAKE not in creds.redact_secrets(f"bare {FAKE} in text")


def test_a_failure_carrying_the_key_leaves_without_it_or_its_chain():
    creds.set_civitai_token(FAKE)

    @creds.secrets_redacted
    def node():
        try:
            raise ValueError(f"for url: https://civitai.com/x?token={FAKE}")
        except ValueError as e:
            raise Exception("download failed") from e

    with pytest.raises(RuntimeError) as info:
        node()
    import traceback

    full = "".join(traceback.format_exception(info.type, info.value, info.tb))
    assert FAKE not in full


def test_a_clean_failure_passes_untouched():
    @creds.secrets_redacted
    def node():
        raise KeyError("nothing secret")

    with pytest.raises(KeyError):
        node()


def test_no_fetcher_declares_or_reads_an_auth_input():
    src = open(os.path.join(os.path.dirname(HERE), "anymatix_checkpoint_fetcher.py")).read()
    assert '"auth"' not in src, "a fetcher still declares or reads an `auth` input"
    assert len(re.findall(r"@secrets_redacted\n    def download_model", src)) == 2
