from pathlib import Path
from .expunge import delete_file_and_cleanup_dir
import hashlib
import json
import os
import re
from typing import Callable, Iterator, Optional

from requests import Session

from tqdm import tqdm
import requests

from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode


def hash_string(input_string):
    encoded_string = input_string.encode()
    hash_object = hashlib.sha256(encoded_string)
    return hash_object.hexdigest()


def redact_url(u: str, appended: Optional[str] = None) -> str:
    """Return a safe-to-log URL string.
    Remove only the query parameters contained in 'appended' (if any), preserving all other params.
    If appended is None, return u unchanged.
    """
    try:
        if not appended:
            return u
        # Parse both URL and appended query tail
        p = urlparse(u)
        current = parse_qsl(p.query, keep_blank_values=True)
        remove = parse_qsl(appended, keep_blank_values=True)
        remove_keys = set(k for k, _ in remove)
        # Remove only matching key-value pairs from tail; if same key appears with multiple values, remove specific pairs
        remove_pairs = set(remove)
        kept = [kv for kv in current if kv not in remove_pairs]
        new_query = urlencode(kept)
        return urlunparse(p._replace(query=new_query))
    except Exception:
        return u


def fetch_headers(url, session):
    file_name = None
    file_size = None
    # TODO: FIXME: should this be session.head??
    with session.get(url, allow_redirects=True, stream=True) as response:
        response.raise_for_status()
        if "Content-Disposition" in response.headers:
            filename_match = re.search(
                r'filename="(.+)"', response.headers["Content-Disposition"])
            if filename_match:
                file_name = filename_match.group(1)
        if "Content-Length" in response.headers:
            file_size = int(response.headers.get('Content-Length', 0))
    return {"file_name": file_name, "file_size": file_size}


def fetch(url: str, session: Session, callback: Callable[[bytes], None], local_file_size: int = 0, chunk_size=8192) -> None:
    req_headers = {}

    if local_file_size > 0:
        req_headers = {'Range': f'bytes={local_file_size}-'}

    # TODO: what if "Range" is not accepted?
    with session.get(url, headers=req_headers, allow_redirects=True, stream=True) as response_2:
        response_2.raise_for_status()
        for item in response_2.iter_content(chunk_size):
            callback(item)


def delete_files(url, dir):
    log_path = Path(dir) / "expunge_log.txt"
    error_path = Path(dir) / "error.txt"
    # Compute hash early and log only the hash to avoid leaking sensitive query params
    url_hash = hash_string(url)
    with open(log_path, "a") as log:
        log.write(f"delete request received, url_hash={url_hash}\n")

    # Pass 1: delete by hash of the provided URL (works if caller sends effective URL)
    deleted_dirs = set()
    for root, _, files in os.walk(dir):
        for f in files:
            with open(log_path, "a") as log:
                log.write(f"Examining file: {f} in {root}\n")
            if url_hash in f:
                file_path = os.path.join(root, f)
                with open(log_path, "a") as log:
                    log.write(f"Matched hash, deleting file: {file_path}\n")
                try:
                    delete_file_and_cleanup_dir(Path(file_path), dir)
                    with open(log_path, "a") as log:
                        log.write(f"Deleted file and checked parent dir: {file_path}\n")
                except Exception as e:
                    with open(error_path, "a") as err:
                        err.write(f"Failed to delete file: {file_path} - {e}\n")
                deleted_dirs.add(os.path.dirname(file_path))

    # Pass 2: delete by matching JSON sidecars whose base URL is a prefix of the provided URL (or equal)
    for root, _, files in os.walk(dir):
        for f in files:
            if not f.endswith('.json'):
                continue
            json_path = os.path.join(root, f)
            try:
                with open(json_path, 'r') as contents:
                    data = json.load(contents)
                if isinstance(data, dict):
                    base_url = data.get("url")
                else:
                    base_url = None
                if isinstance(base_url, str) and (url == base_url or url.startswith(base_url + "?") or url.startswith(base_url + "&")):
                    # Delete the associated model file and the json itself
                    model_file = data.get("file_name")
                    if model_file:
                        file_path = os.path.join(root, model_file)
                        if os.path.exists(file_path):
                            try:
                                delete_file_and_cleanup_dir(Path(file_path), dir)
                                with open(log_path, "a") as log:
                                    log.write(f"Deleted model file by base-url match: {file_path}\n")
                            except Exception as e:
                                with open(error_path, "a") as err:
                                    err.write(f"Failed to delete model file: {file_path} - {e}\n")
                            deleted_dirs.add(os.path.dirname(file_path))
                    # Delete the json sidecar
                    try:
                        delete_file_and_cleanup_dir(Path(json_path), dir)
                        with open(log_path, "a") as log:
                            log.write(f"Deleted sidecar JSON: {json_path}\n")
                    except Exception as e:
                        with open(error_path, "a") as err:
                            err.write(f"Failed to delete sidecar JSON: {json_path} - {e}\n")
                        deleted_dirs.add(os.path.dirname(json_path))
            except Exception as e:
                with open(error_path, "a") as err:
                    err.write(f"Failed to read/parse JSON: {json_path} - {e}\n")

    # After all deletions, check and remove empty parent directories
    for d in deleted_dirs:
        parent = Path(d)
        with open(log_path, "a") as log:
            log.write(f"Checking if parent directory is empty: {parent}\n")
        if parent.exists() and parent.is_dir() and not any(parent.iterdir()):
            try:
                parent.rmdir()
                with open(log_path, "a") as log:
                    log.write(f"fetch.py: Deleted empty output directory: {parent}\n")
            except Exception as e:
                with open(error_path, "a") as err:
                    err.write(f"fetch.py: Failed to remove output directory: {parent} - {e}\n")
        else:
            with open(log_path, "a") as log:
                log.write(f"Parent directory not empty after deletion: {parent}\n")


def download_file(url, dir, callback: Optional[Callable[[int, Optional[int]], None]] = None, expand_info: Optional[Callable[[str], dict | None]] = None, effective_url: Optional[str] = None, redact_append: Optional[str] = None):
    effective = effective_url or url
    print("download file", redact_url(effective, redact_append), dir)
    url_hash = hash_string(effective)
    os.makedirs(dir, exist_ok=True)
    store_path = os.path.join(dir, f"{url_hash}.json")
    parsed_url = urlparse(effective)
    file_name_default = parsed_url.path.split('/')[-1].split('?')[0]
    # Always persist base URL, never include token-bearing URL
    data = {"url": url}

    with requests.Session() as session:

        if (os.path.exists(store_path)):
            print("loading json", store_path)
            with open(store_path, 'r') as contents:
                data.update(json.load(contents))
        else:
            print("fetching headers", redact_url(effective, redact_append))
            data.update(fetch_headers(effective, session))
            if data["file_name"] is None:
                data["file_name"] = f"{file_name_default}"
            f = data["file_name"]
            x = f.rsplit(".")
            data["name"] = x[0]
            data["file_name"] = f"{x[0]}_{url_hash}" + \
                ('.' + '.'.join(x[1:]) if len(x) > 1 else "")
            if expand_info:
                info = expand_info(url)
                if info is not None:
                    data["data"] = info
            with open(store_path, 'w') as file:
                json.dump(data, file, indent=4)

        file_path = os.path.join(dir, data["file_name"])
        local_file_size = 0

        if data["file_size"] is not None and os.path.exists(file_path):
            local_file_size = os.path.getsize(file_path)
            if local_file_size == data["file_size"]:
                return file_path

        downloaded_size = local_file_size

        with open(file_path, 'ab') as file:
            with tqdm(total=data["file_size"], initial=local_file_size) as progress_bar:
                def cb(chunk):
                    nonlocal downloaded_size
                    if (chunk):
                        file.write(chunk)
                        l = len(chunk)
                        downloaded_size += l
                        progress_bar.update(l)
                        if callback:
                            callback(downloaded_size, data["file_size"])
                fetch(effective, session, cb, local_file_size)

        print("Model name:", data["file_name"])

        return file_path


def expand_info_civitai(url):
    # get the model id from the url using a regex that matches the first /.../ after https://civitai.com/api/download/models
    pattern = r'https://civitai\.com/api/download/models/([^/]+)'
    match = re.search(pattern, url)
    if match:
        model_id = match.group(1)
    else:
        return None
    model_info_url = f"https://civitai.com/api/v1/model-versions/{model_id}"
    with requests.Session() as session:
        return requests.get(model_info_url, allow_redirects=True).json()


def expand_info(url):
    if url.startswith("https://civitai.com/api/download/models"):
        return expand_info_civitai(url)
    return None


if __name__ == "__main__":
    url = "https://civitai.com/api/download/models/128713"
    dir = "tmp"
    model_name = download_file(url, dir, print, expand_info)
    print(f"downloaded model {model_name}")
