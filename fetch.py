#%%
import os
import re
from typing import Callable,Optional
import requests

from urllib.parse import urlparse

def download_file(url,dir,callback: Callable[[int,Optional[int]],None]):
    
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
                    for chunk in response_2.iter_content(chunk_size=8192):
                        if (chunk):
                            file.write(chunk)
                            downloaded_size+=len(chunk)
                            callback(downloaded_size,file_size)

        return file_path

# def download_file_old(url, dir, callback):
#     callback(0,100)
        
#     wget_process = subprocess.Popen(
#         ['wget', '--content-disposition', '--no-hsts', '-c', '--progress=dot', '-P', dir, url],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE
#     )
#     # Read output in real-time
#     length=-1
#     filename = None
#     while True:
#         line = wget_process.stderr.readline().decode("iso-8859-1")
#         if not line:
#             print("END")
#             break
#         else:
#             print("***", line)          
#         filename_match = re.search(r"Parsed filename from Content-Disposition: (.*)", line)
#         if filename_match:
#             filename = filename_match.group(1)
#             print(f"Detected filename: {filename}")
#         else:
#             length_match = re.search(r"Content-Length:\s*(\d+)", line)
#             if length_match:                
#                 length = int(length_match.group(1))  # Capturing the first number
#                 print(f"Detected length: {length}")
#             else:
#                 progress_match = re.search(r"(\d+)(?=K)",line)
#                 if progress_match:                    
#                     progress=int(progress_match.group(1))
#                     callback(progress*1024,length) # TODO: what if length is not defined??
#     wget_process.wait()    
#     callback(100,100)
#     return filename
        

# def f(progress,length): 
#     print(f"Progress: {100*progress/length:.2f}")

download_file("https://civitai.com/api/download/models/128713","/tmp",lambda x,y: print(x,y))

#download_file("http://httpbin.org/redirect/1","/tmp",lambda x,y: print(x,y))
#download_file("https://via.placeholder.com/150","/tmp",lambda x,y: print(x,y))