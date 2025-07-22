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

from urllib.parse import urlparse


def hash_string(input_string):
    encoded_string = input_string.encode()
    hash_object = hashlib.sha256(encoded_string)
    return hash_object.hexdigest()


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
    print("delete files", url)
    url_hash = hash_string(url)
    for root, _, files in os.walk(dir):
        for f in files:
            print("examining", f)
            if url_hash in f:
                print("deleting", f)
                file_path = os.path.join(root, f)
                print("deleting", file_path)
                # Use the new cleanup logic
                delete_file_and_cleanup_dir(Path(file_path), dir)


def download_file(url, dir, callback: Optional[Callable[[int, Optional[int]], None]] = None, expand_info: Optional[Callable[[str], dict | None]] = None):
    print("download file", url, dir)
    url_hash = hash_string(url)
    os.makedirs(dir, exist_ok=True)
    store_path = os.path.join(dir, f"{url_hash}.json")
    parsed_url = urlparse(url)
    file_name_default = parsed_url.path.split('/')[-1].split('?')[0]
    data = {"url": url}

    with requests.Session() as session:

        if (os.path.exists(store_path)):
            print("loading json", store_path)
            with open(store_path, 'r') as contents:
                data.update(json.load(contents))
        else:
            print("fetching headers", url)
            data.update(fetch_headers(url, session))
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
                fetch(url, session, cb, local_file_size)

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
