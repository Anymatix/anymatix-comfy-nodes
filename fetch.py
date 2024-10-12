#%%
import hashlib
import json
import os
import re
from typing import Callable,Optional

from tqdm import tqdm
import requests

from urllib.parse import urlparse

def hash_string(input_string):
    encoded_string = input_string.encode()
    hash_object = hashlib.sha256(encoded_string)
    return hash_object.hexdigest()

def fetch_headers(url,session):
    file_name = None
    file_size = None
    with session.get(url,allow_redirects=True,stream=True) as response: # TODO: FIXME: should this be session.head??
        response.raise_for_status()
        if "Content-Disposition" in response.headers:
            filename_match = re.search(r'filename="(.+)"', response.headers["Content-Disposition"])
            if filename_match:
                file_name = filename_match.group(1)
        if "Content-Length" in response.headers:
            file_size = int(response.headers.get('Content-Length', 0))            
    return { "file_name": file_name, "file_size": file_size}

def fetch(url,session,local_file_size: int = 0,chunk_size=8192):
        req_headers = {}
                
        if local_file_size > 0:        
            req_headers = {'Range': f'bytes={local_file_size}-'}            
                
        with session.get(url, headers=req_headers,allow_redirects=True, stream=True) as response_2: # TODO: what if "Range" is not accepted?
            response_2.raise_for_status()                
            return response_2.iter_content(chunk_size)
                    
def download_file(url,dir,store,callback: Optional[Callable[[int,Optional[int]],None]] = None,expand_info: Optional[Callable[[str],dict|None]] = None):   
    url_hash = hash_string(url)
    os.makedirs(store,exist_ok=True)
    store_path = os.path.join(store,f"{url_hash}.json")
    parsed_url = urlparse(url)
    file_name_default = parsed_url.path.split('/')[-1].split('?')[0]        
    data = { "url": url }
    
    with requests.Session() as session:
        
        if (os.path.exists(store_path)):
            with open(store_path,'r') as contents:
                data.update(json.load(contents))
        else:
            data.update(fetch_headers(url,session))
        
        if data["file_name"] is None:
            data["file_name"] = f"{url_hash}_{file_name_default}"
            
        file_path=os.path.join(dir,data["file_name"])
        local_file_size = 0        
                
        if data["file_size"] is not None and os.path.exists(file_path): 
            local_file_size =  os.path.getsize(file_path) 
            if local_file_size == data["file_size"]:
                return data["file_name"]
            
        chunks = fetch(url,session,local_file_size)
        downloaded_size = local_file_size
        
        with open(file_path, 'ab') as file:
            with tqdm(total=data["file_size"],initial=local_file_size) as progress_bar:            
                for chunk in chunks:                                                        
                    if (chunk):
                        file.write(chunk)
                        l=len(chunk)
                        downloaded_size+=l
                        progress_bar.update(l)
                        if callback:
                            callback(downloaded_size,data["file_size"])

        if expand_info:
            info = expand_info(url) 
            if info is not None:
                data.update(info)
            
        with open(store_path, 'w') as file:                
            json.dump(data, file, indent=4)
        
        return data["file_name"]
