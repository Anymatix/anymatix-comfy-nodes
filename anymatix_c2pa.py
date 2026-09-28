"""
The C2PA manifest every generated result carries, ON TOP of its IPTC label.

WHY IT EXISTS

  `anymatix_ai_disclosure.py` writes the unsigned IPTC label (AI Act
  art. 50(2)). On 2026-09-28 Anymatix decided to add a C2PA (Content
  Credentials) manifest as well -- TRACKERS item
  `c2pa-content-credentials-marking-decided-2026-08-13-not`, section
  "REVISED". This module is that manifest. It never replaces the label: the
  label is written first, and this module signs the labelled file.

WHY THE SIGNATURE IS "UNTRUSTED", ON PURPOSE

  A desktop app that signs locally keeps its key on the user's machine, where
  it can be extracted; a certificate on a public trust list would prove
  nothing there. So the key is generated ON THE MACHINE, once per install,
  under a root certificate made on the spot -- and this repository, which is
  public, never holds a secret. Every C2PA reader therefore reports the
  signer as not on its trust list (`signingCredential.untrusted`), which is
  exactly true, while the manifest itself validates: it is well formed, and
  the file has not changed since it was signed.

  The root's private key is used once, to issue the signing certificate, and
  never written anywhere: nothing can issue a second certificate under it.

  Consequence, stated so nobody rediscovers it: files made on the same
  install carry the same certificate, so they can be recognised as coming
  from one install. Nothing in the certificate names a person or a machine.

WHAT IS IN THE MANIFEST -- a module constant, like the label

  claim generator `Anymatix`, and one `c2pa.actions` assertion: `c2pa.created`
  with digitalSourceType `trainedAlgorithmicMedia`. No thumbnail (measured
  2026-09-28: +577 KB on a PNG with one, +13 KB without). No title, no path,
  no user, no prompt, no card name: the file is signed from a stream, so
  c2pa-rs never sees a file name either. Disclosure, not surveillance.

THE TIMESTAMP

  From a public TSA when reachable. Reachability is a TCP connect with a short
  timeout, because c2pa-rs fails the whole signature when its TSA request
  fails, and waits 30 s before it does when packets are dropped (measured).
  Unreachable -> signed without a timestamp, and the TSA is not tried again
  for `TSA_RETRY_AFTER_SECONDS`, so an offline machine pays the probe once,
  not once per file. A caller writing a large batch passes `timestamp=False`:
  one TSA round-trip per frame would add minutes to a sequence.

WHAT IS NOT NEGOTIABLE HERE

  Signing never costs the result. The signed file is written to a sibling and
  `os.replace()`d over the staged one only when complete; any failure --
  c2pa-python missing, a key that cannot be written, a format c2pa-rs
  rejects -- leaves the file exactly as it was, IPTC label included, and
  prints one line.

  | extension           | C2PA manifest                                    |
  |---------------------|--------------------------------------------------|
  | png, jpg, jpeg,     | yes -- verified one by one,                      |
  | webp, gif, tif,     | `tests/test_c2pa_manifest.py`                    |
  | tiff, avif, mp4,    |                                                  |
  | mov, m4a, mp3, wav, |                                                  |
  | flac                |                                                  |
  | exr, mkv            | no -- c2pa-rs has no handler; IPTC label only    |
  | bmp                 | no -- no metadata at all, not even the label     |
  | json (sidecars)     | exempt -- not media                              |

  Environment overrides, for tests and for an operator who needs them:
  `ANYMATIX_C2PA_DIR` (where the signing credential lives) and
  `ANYMATIX_C2PA_TSA` (the TSA URL; empty = never timestamp).

This module imports neither `comfy` nor `folder_paths`, and imports `c2pa`
and `cryptography` only when it signs, so a missing wheel can never stop the
node pack from loading. Tested outside ComfyUI: `tests/test_c2pa_manifest.py`.
"""

import datetime
import io
import json
import os
import socket
import sys
import time
import urllib.parse

try:
    from .anymatix_atomic_write import cleanup_temp, temp_path_for
    from .anymatix_ai_disclosure import DIGITAL_SOURCE_TYPE, SOFTWARE
except ImportError:
    # Loaded standalone (tests via spec_from_file_location), with no package
    # context for a relative import to resolve against.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from anymatix_atomic_write import cleanup_temp, temp_path_for
    from anymatix_ai_disclosure import DIGITAL_SOURCE_TYPE, SOFTWARE


#: extension -> the MIME type c2pa-rs embeds a manifest into. Every entry is
#: signed and read back as valid by the tests; an extension that is not here
#: keeps its IPTC label only (see the table above).
MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "avif": "image/avif",
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "m4a": "audio/mp4",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
}

#: The whole manifest definition. A constant: nothing a caller passes can
#: reach it.
MANIFEST = json.dumps(
    {
        "claim_generator_info": [{"name": SOFTWARE}],
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": DIGITAL_SOURCE_TYPE,
                            "softwareAgent": {"name": SOFTWARE},
                        }
                    ]
                },
            }
        ],
    }
)

#: c2pa-rs settings for signing: no thumbnail.
SETTINGS = {"builder": {"thumbnail": {"enabled": False}}}

TSA_URL = "http://timestamp.digicert.com"
TSA_PROBE_SECONDS = 3.0
TSA_RETRY_AFTER_SECONDS = 600.0

#: One file holds the signing key and its certificate chain, so the two can
#: never disagree: it is replaced whole or not at all.
CREDENTIAL_FILE = "anymatix-c2pa-signer.pem"

ROOT_COMMON_NAME = "Anymatix self-signed root (untrusted)"
SIGNER_COMMON_NAME = "Anymatix self-signed content credentials (untrusted)"
ORGANIZATION = "Anymatix"
ROOT_YEARS = 20
SIGNER_YEARS = 10

#: monotonic time before which the TSA is not tried again.
_tsa_unreachable_until = 0.0


def _extension(extension):
    return extension.lower().lstrip(".")


def _not_signed(what, error):
    print(f"anymatix: C2PA manifest not written to a {what}: {error!r}")


# ------------------------------------------------------ the signing credential ----


def credential_dir():
    """Per-install, per-user data directory; never inside a repository."""
    override = os.environ.get("ANYMATIX_C2PA_DIR")
    if override:
        return override
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        return os.path.join(base, "Anymatix", "content-credentials")
    if sys.platform == "darwin":
        return os.path.join(
            home, "Library", "Application Support", "Anymatix", "content-credentials"
        )
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share")
    return os.path.join(base, "anymatix", "content-credentials")


def credential_path():
    return os.path.join(credential_dir(), CREDENTIAL_FILE)


def _name(common_name):
    from cryptography import x509
    from cryptography.x509.oid import NameOID

    return x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, ORGANIZATION),
        ]
    )


def _key_usage(**granted):
    from cryptography import x509

    usages = dict.fromkeys(
        (
            "digital_signature",
            "content_commitment",
            "key_encipherment",
            "data_encipherment",
            "key_agreement",
            "key_cert_sign",
            "crl_sign",
            "encipher_only",
            "decipher_only",
        ),
        False,
    )
    usages.update(granted)
    return x509.KeyUsage(**usages)


def generate_credential(now=None, signer_days=None):
    """
    A fresh ES256 signing key and its chain [signer, root], as one PEM.

    The signer certificate has the profile C2PA requires of a claim signer:
    not a CA, digitalSignature, an extended key usage (emailProtection) and
    an authority key identifier. `now` and `signer_days` exist for the tests.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID

    now = now or datetime.datetime.now(datetime.timezone.utc)
    start = now - datetime.timedelta(days=1)
    signer_days = signer_days if signer_days is not None else 365 * SIGNER_YEARS

    root_key = ec.generate_private_key(ec.SECP256R1())
    root_name = _name(ROOT_COMMON_NAME)
    root_ski = x509.SubjectKeyIdentifier.from_public_key(root_key.public_key())
    root = (
        x509.CertificateBuilder()
        .subject_name(root_name)
        .issuer_name(root_name)
        .public_key(root_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(start)
        .not_valid_after(now + datetime.timedelta(days=365 * ROOT_YEARS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(_key_usage(key_cert_sign=True, crl_sign=True), critical=True)
        .add_extension(root_ski, critical=False)
        .sign(root_key, hashes.SHA256())
    )

    signer_key = ec.generate_private_key(ec.SECP256R1())
    signer = (
        x509.CertificateBuilder()
        .subject_name(_name(SIGNER_COMMON_NAME))
        .issuer_name(root_name)
        .public_key(signer_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(start)
        .not_valid_after(now + datetime.timedelta(days=signer_days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(_key_usage(digital_signature=True), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(signer_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(root_ski),
            critical=False,
        )
        .sign(root_key, hashes.SHA256())
    )
    # root_key goes out of scope here and is never serialised.
    key_pem = signer_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    pem = serialization.Encoding.PEM
    return key_pem + signer.public_bytes(pem) + root.public_bytes(pem)


def parse_credential(data):
    """
    (key PEM, chain PEM) from a credential file's bytes. Raises ValueError
    when it is malformed, when the key is not the signer's, or when the
    signer certificate expires within a day.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    at = data.find(b"-----BEGIN CERTIFICATE-----")
    if at <= 0:
        raise ValueError("no key or no certificate")
    key_pem, chain_pem = data[:at], data[at:]
    key = serialization.load_pem_private_key(key_pem, password=None)
    chain = x509.load_pem_x509_certificates(chain_pem)
    if len(chain) != 2:
        raise ValueError(f"expected [signer, root], found {len(chain)} certificates")
    signer = chain[0]
    if signer.public_key().public_numbers() != key.public_key().public_numbers():
        raise ValueError("the key is not the signer certificate's key")
    horizon = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
    if signer.not_valid_after_utc < horizon:
        raise ValueError("the signer certificate has expired")
    return key_pem, chain_pem


def _write_private(path, data):
    """Write `data` at `path`, owner-only, whole or not at all."""
    directory = os.path.dirname(path)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    if os.name == "posix":
        os.chmod(directory, 0o700)
    staged = temp_path_for(path)
    # O_BINARY: without it Windows opens the descriptor in text mode and
    # rewrites every newline of the PEM.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(staged, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(staged, path)
    except BaseException:
        cleanup_temp(staged)
        raise


def load_credential():
    """
    (key PEM, chain PEM) for this install, generated on first use.

    A file that is missing is created; one that is malformed or expired is
    replaced, and says so. Two processes creating it at the same moment each
    sign with a consistent pair, and the last write is the one that stays.
    """
    path = credential_path()
    try:
        with open(path, "rb") as f:
            return parse_credential(f.read())
    except FileNotFoundError:
        print(f"anymatix: generating this install's C2PA signing credential in {os.path.dirname(path)}")
    except ValueError as error:
        print(f"anymatix: replacing the C2PA signing credential: {error}")
    data = generate_credential()
    _write_private(path, data)
    return parse_credential(data)


# ------------------------------------------------------------- the timestamp ----


def _tsa_url():
    return os.environ.get("ANYMATIX_C2PA_TSA", TSA_URL)


def _tsa_reachable(url):
    """A TCP connect to the TSA within `TSA_PROBE_SECONDS`."""
    global _tsa_unreachable_until
    if time.monotonic() < _tsa_unreachable_until:
        return False
    parts = urllib.parse.urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        socket.create_connection((parts.hostname, port), timeout=TSA_PROBE_SECONDS).close()
        return True
    except OSError as error:
        _tsa_unreachable_until = time.monotonic() + TSA_RETRY_AFTER_SECONDS
        print(f"anymatix: C2PA timestamp authority unreachable, signing without a timestamp: {error!r}")
        return False


# ------------------------------------------------------------------- signing ----


def _sign(mime, source, dest, credential, tsa_url):
    import c2pa

    key_pem, chain_pem = credential
    info = c2pa.C2paSignerInfo(
        alg=b"es256", sign_cert=chain_pem, private_key=key_pem, ta_url=tsa_url
    )
    with c2pa.Signer.from_info(info) as signer, c2pa.Context.from_dict(SETTINGS) as context:
        with c2pa.Builder(MANIFEST, context=context) as builder:
            builder.sign(signer, mime, source, dest)


def _sign_with_timestamp_if_possible(mime, source, dest, timestamp):
    """
    Sign `source` into `dest`, with a timestamp when one can be had. Both are
    seekable binary streams; `dest` is rewound and truncated for a retry.
    """
    credential = load_credential()
    url = _tsa_url() if timestamp else ""
    if url and _tsa_reachable(url):
        try:
            _sign(mime, source, dest, credential, url)
            return
        except Exception as error:  # the timestamp is optional; the manifest is not
            # Retried without the TSA, and only once: a failure that is not
            # the TSA's fails the retry too, and is reported by the caller.
            print(f"anymatix: C2PA timestamp failed, signing without one: {error!r}")
            source.seek(0)
            dest.seek(0)
            dest.truncate()
    _sign(mime, source, dest, credential, None)


def sign_bytes(data, extension, timestamp=True):
    """
    `data` with a C2PA manifest embedded, or `data` itself when the format
    cannot carry one or anything at all goes wrong. Never raises.
    """
    extension = _extension(extension)
    mime = MIME_TYPES.get(extension)
    if mime is None:
        return data
    try:
        dest = io.BytesIO()
        _sign_with_timestamp_if_possible(mime, io.BytesIO(bytes(data)), dest, timestamp)
        return dest.getvalue()
    except Exception as error:  # the manifest is best-effort; the output is not
        _not_signed(extension, error)
        return data


def sign_file(path, extension, timestamp=True):
    """
    Embed a C2PA manifest in the file at `path`, in place. True when it now
    carries one.

    Meant for a staged temp file, after the IPTC label and before the caller
    publishes it. The signed file is written to a sibling and `os.replace()`d
    over `path` only when complete, so `path` is either signed or exactly as
    it was. Never raises.
    """
    extension = _extension(extension)
    mime = MIME_TYPES.get(extension)
    if mime is None:
        return False
    staged = None
    try:
        staged = temp_path_for(path)
        with open(path, "rb") as source, open(staged, "w+b") as dest:
            _sign_with_timestamp_if_possible(mime, source, dest, timestamp)
            dest.flush()
        os.replace(staged, path)
        staged = None
        return True
    except Exception as error:  # the manifest is best-effort; the output is not
        _not_signed(extension, error)
        return False
    finally:
        if staged is not None:
            cleanup_temp(staged)
