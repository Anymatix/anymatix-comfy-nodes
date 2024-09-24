#%%
import os
import re
from typing import Callable,Optional

from tqdm import tqdm
import requests

from urllib.parse import urlparse

def download_file(url,dir,callback: Optional[Callable[[int,Optional[int]],None]] = None):    
    with requests.Session() as session:
        parsed_url = urlparse(url)
        file_name = parsed_url.path.split('/')[-1].split('?')[0]
        file_size = None
        local_file_size = 0
        headers = {}

        with session.get(url,allow_redirects=True,stream=True) as response:
            response.raise_for_status()
            if "Content-Disposition" in response.headers:
                filename_match = re.search(r'filename="(.+)"', response.headers["Content-Disposition"])
                if filename_match:
                    file_name = filename_match.group(1)
            if "Content-Length" in response.headers:
                file_size = int(response.headers.get('Content-Length', 0))            
            
        file_path=os.path.join(dir,file_name)
            
        if os.path.exists(file_path):
            local_file_size = os.path.getsize(file_path)
            headers = {'Range': f'bytes={local_file_size}-'}            
        
        print(file_path,local_file_size,file_size)
        
        if file_size is None or local_file_size < file_size:
            with session.get(url, headers=headers,allow_redirects=True, stream=True) as response_2: # TODO: what if "Range" is not accepted?
                response_2.raise_for_status()
                with open(file_path, 'ab') as file:
                    
                    downloaded_size = local_file_size
                    with tqdm(total=file_size,initial=local_file_size) as progress_bar:
                        for chunk in response_2.iter_content(chunk_size=8192):
                            if (chunk):
                                file.write(chunk)
                                l=len(chunk)
                                downloaded_size+=l
                                progress_bar.update(l)
                                if callback:
                                    callback(downloaded_size,file_size)

        return file_name
