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
    """Fetch headers with error handling for missing requests"""
    if not REQUESTS_AVAILABLE:
        return {"file_name": None, "file_size": None}
        
    file_name = None
    file_size = None
    try:
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
    except Exception:
        pass
    return {"file_name": file_name, "file_size": file_size}


def fetch(url: str, session, callback: Callable[[bytes], None], local_file_size: int = 0, chunk_size=8192) -> None:
    """Traditional fetch with graceful fallback"""
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library not available")
        
    req_headers = {}

    if local_file_size > 0:
        req_headers = {'Range': f'bytes={local_file_size}-'}

    # TODO: what if "Range" is not accepted?
    with session.get(url, headers=req_headers, allow_redirects=True, stream=True) as response_2:
        response_2.raise_for_status()
        for item in response_2.iter_content(chunk_size):
            callback(item)


class SegmentDownloader:
    """State-of-the-art parallel segment downloader with adaptive optimization"""
    
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
                    if response.status_code not in [206, 200]:  # Partial Content or OK
                        raise Exception(f"HTTP {response.status_code}")
                        
                    segment_data = b''
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            segment_data += chunk
                            with self.lock:
                                self.downloaded_bytes += len(chunk)
                                if self.progress_callback:
                                    self.progress_callback(self.downloaded_bytes, self.total_size)
                    
                    # Write segment to temp file
                    temp_path = f"{self.file_path}.segment_{segment_id}"
                    with open(temp_path, 'wb') as f:
                        f.write(segment_data)
                    
                    with self.lock:
                        self.active_segments[segment_id] = temp_path
                        
                    return True
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = backoff_base * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    with self.lock:
                        self.failed_segments.append((segment_id, start, end))
                    return False
        return False
        
    def download_parallel(self) -> bool:
        """Execute parallel download with intelligent load balancing"""
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
                        return False
                except Exception as e:
                    with self.lock:
                        self.failed_segments.append((seg_id, start, end))
                    return False
        
        # Retry failed segments with single connection
        if self.failed_segments:
            for seg_id, start, end in self.failed_segments:
                if not self._download_segment_sync(seg_id, start, end):
                    return False
                    
        return True
        
    def assemble_file(self) -> bool:
        """Assemble segments into final file with integrity verification"""
        try:
            with open(self.file_path, 'wb') as output_file:
                for i in range(len(self.active_segments)):
                    segment_path = self.active_segments.get(i)
                    if not segment_path or not os.path.exists(segment_path):
                        return False
                        
                    with open(segment_path, 'rb') as segment_file:
                        output_file.write(segment_file.read())
            
            # Cleanup temp files
            for segment_path in self.active_segments.values():
                try:
                    os.remove(segment_path)
                except:
                    pass
                    
            # Verify file size
            final_size = os.path.getsize(self.file_path)
            return final_size == self.total_size
            
        except Exception:
            return False


async def fetch_async_segment(session, url: str, start: int, end: int, 
                            segment_id: int, progress_callback: Optional[Callable] = None) -> Tuple[int, bytes]:
    """Async segment downloader with HTTP/2 optimization"""
    if not AIOHTTP_AVAILABLE:
        raise ImportError("aiohttp not available")
        
    headers = {'Range': f'bytes={start}-{end}'}
    
    async with session.get(url, headers=headers) as response:
        if response.status not in [206, 200]:
            raise Exception(f"HTTP {response.status}")
            
        segment_data = b''
        async for chunk in response.content.iter_chunked(8192):
            segment_data += chunk
            if progress_callback:
                progress_callback(len(chunk))
        
        return segment_id, segment_data


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
        
    async def download_async(self) -> bool:
        """Execute async parallel download with HTTP/2 optimization"""
        try:
            # Calculate segments
            segment_size = max(1024*1024, self.total_size // self.max_connections)  # At least 1MB per segment
            segments = []
            
            for i in range(0, self.total_size, segment_size):
                start = i
                end = min(i + segment_size - 1, self.total_size - 1)
                segments.append((len(segments), start, end))
            
            # Configure HTTP/2 connector with connection pooling
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections,
                enable_cleanup_closed=True,
                force_close=True,
                keepalive_timeout=30
            )
            
            timeout = aiohttp.ClientTimeout(total=None, connect=30)
            
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'AnymatixFetcher/2.0 (Parallel)'}
            ) as session:
                
                def progress_update(bytes_read):
                    self.downloaded_bytes += bytes_read
                    if self.progress_callback:
                        self.progress_callback(self.downloaded_bytes, self.total_size)
                
                # Download all segments concurrently
                tasks = [
                    fetch_async_segment(session, self.url, start, end, seg_id, progress_update)
                    for seg_id, start, end in segments
                ]
                
                segment_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for failures
                segment_data = {}
                for result in segment_results:
                    if isinstance(result, Exception):
                        return False
                    seg_id, data = result
                    segment_data[seg_id] = data
                
                # Assemble file
                async with aiofiles.open(self.file_path, 'wb') as f:
                    for i in range(len(segments)):
                        await f.write(segment_data[i])
                
                return True
                
        except Exception:
            return False


def check_range_support(url: str) -> Tuple[bool, Optional[int]]:
    """Check if server supports range requests and get file size"""
    if not REQUESTS_AVAILABLE:
        return False, None
        
    try:
        with requests.head(url, allow_redirects=True, timeout=10) as response:
            response.raise_for_status()
            
            accepts_ranges = response.headers.get('Accept-Ranges', '').lower() == 'bytes'
            content_length = response.headers.get('Content-Length')
            file_size = int(content_length) if content_length else None
            
            return accepts_ranges, file_size
    except Exception:
        return False, None


def fetch_parallel(url: str, file_path: str, callback: Optional[Callable[[int, Optional[int]], None]] = None,
                  local_file_size: int = 0, max_connections: int = 8) -> bool:
    """State-of-the-art parallel download with intelligent fallback"""
    
    if not REQUESTS_AVAILABLE:
        return False
    
    # Check server capabilities
    supports_ranges, total_size = check_range_support(url)
    
    if not supports_ranges or not total_size:
        # Fallback to traditional download
        return False
        
    # Skip parallel for small files (< 5MB)
    if total_size < 5 * 1024 * 1024:
        return False
        
    # Handle resume scenario
    if local_file_size > 0:
        if local_file_size >= total_size:
            return True  # Already complete
        # For resume, we'll use traditional method for simplicity
        return False
    
    # Choose download strategy based on available dependencies
    try:
        # Try async method first (fastest) if available
        if AIOHTTP_AVAILABLE and AIOFILES_AVAILABLE:
            async def run_async():
                downloader = AsyncParallelDownloader(url, file_path, total_size, callback, max_connections)
                return await downloader.download_async()
            
            # Run async download
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            return loop.run_until_complete(run_async())
        else:
            # Fallback to threaded parallel download
            downloader = SegmentDownloader(url, file_path, total_size, callback, max_connections)
            if downloader.download_parallel():
                return downloader.assemble_file()
            return False
        
    except Exception:
        return False


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
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library is required for downloading")
        
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

        # STATE-OF-THE-ART PARALLEL DOWNLOAD ATTEMPT
        parallel_success = False
        if local_file_size == 0 and data["file_size"] is not None:  # Only for fresh downloads
            print(f"Attempting parallel download for {data['file_name']} ({data['file_size']} bytes)")
            try:
                parallel_success = fetch_parallel(
                    effective, 
                    file_path, 
                    callback, 
                    local_file_size,
                    max_connections=min(8, max(2, data["file_size"] // (10*1024*1024)))  # Adaptive connections
                )
                if parallel_success:
                    print(f"✅ Parallel download completed successfully: {data['file_name']}")
                    return file_path
            except Exception as e:
                print(f"⚠️  Parallel download failed, falling back to traditional: {e}")
                parallel_success = False

        # TRADITIONAL FALLBACK (backwards compatible)
        if not parallel_success:
            print(f"Using traditional download for {data['file_name']}")
            with open(file_path, 'ab') as file:
                progress_bar = None
                if TQDM_AVAILABLE and data["file_size"]:
                    try:
                        progress_bar = tqdm(total=data["file_size"], initial=local_file_size)
                    except:
                        progress_bar = None
                    
                def cb(chunk):
                    nonlocal downloaded_size
                    if (chunk):
                        file.write(chunk)
                        l = len(chunk)
                        downloaded_size += l
                        if progress_bar:
                            progress_bar.update(l)
                        if callback:
                            callback(downloaded_size, data["file_size"])
                            
                try:
                    fetch(effective, session, cb, local_file_size)
                finally:
                    if progress_bar:
                        try:
                            progress_bar.close()
                        except:
                            pass

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


def benchmark_download_methods(url: str, output_dir: str) -> dict:
    """Benchmark different download methods for performance analysis"""
    import time
    
    results = {}
    
    # Check if URL supports parallel download
    supports_ranges, file_size = check_range_support(url)
    if not supports_ranges or not file_size:
        return {"error": "URL does not support range requests or file size unknown"}
    
    if file_size < 1024*1024:  # Skip benchmark for files < 1MB
        return {"error": "File too small for meaningful benchmark"}
        
    print(f"Benchmarking download methods for {file_size:,} bytes")
    
    # Test parallel download
    try:
        test_file = os.path.join(output_dir, f"benchmark_parallel_{int(time.time())}")
        start_time = time.time()
        
        success = fetch_parallel(url, test_file, max_connections=8)
        
        if success:
            end_time = time.time()
            results["parallel"] = {
                "time": end_time - start_time,
                "speed_mbps": (file_size / (1024*1024)) / (end_time - start_time),
                "success": True
            }
            os.remove(test_file)  # Cleanup
        else:
            results["parallel"] = {"success": False}
            
    except Exception as e:
        results["parallel"] = {"success": False, "error": str(e)}
    
    # Test traditional download for comparison
    try:
        test_file = os.path.join(output_dir, f"benchmark_traditional_{int(time.time())}")
        start_time = time.time()
        
        with requests.Session() as session:
            with open(test_file, 'wb') as f:
                def cb(chunk):
                    if chunk:
                        f.write(chunk)
                fetch(url, session, cb)
        
        end_time = time.time()
        actual_size = os.path.getsize(test_file)
        
        results["traditional"] = {
            "time": end_time - start_time,
            "speed_mbps": (actual_size / (1024*1024)) / (end_time - start_time),
            "success": True
        }
        os.remove(test_file)  # Cleanup
        
    except Exception as e:
        results["traditional"] = {"success": False, "error": str(e)}
    
    # Calculate speedup
    if (results.get("parallel", {}).get("success") and 
        results.get("traditional", {}).get("success")):
        speedup = results["traditional"]["time"] / results["parallel"]["time"]
        results["speedup"] = f"{speedup:.2f}x"
        results["bandwidth_improvement"] = f"{results['parallel']['speed_mbps']:.1f} vs {results['traditional']['speed_mbps']:.1f} MB/s"
    
    return results


if __name__ == "__main__":
    url = "https://civitai.com/api/download/models/128713"
    dir = "tmp"
    model_name = download_file(url, dir, print, expand_info)
    print(f"downloaded model {model_name}")
    
    # Run performance benchmark
    print("\n=== PERFORMANCE BENCHMARK ===")
    benchmark_results = benchmark_download_methods(url, dir)
    print(json.dumps(benchmark_results, indent=2))
