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


def hash_string(input_string):
    encoded_string = input_string.encode()
    hash_object = hashlib.sha256(encoded_string)
    return hash_object.hexdigest()


def compute_file_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hash of a file efficiently."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


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

    try:
        # TODO: what if "Range" is not accepted?
        with session.get(url, headers=req_headers, allow_redirects=True, stream=True) as response_2:
            response_2.raise_for_status()
            for item in response_2.iter_content(chunk_size):
                # Check for ComfyUI interrupt signal before processing each chunk
                check_interrupted()
                callback(item)
    except requests.RequestException as e:
        raise Exception(f"HTTP request failed during traditional download: {e}") from e
    except Exception as e:
        # Re-raise InterruptProcessingException as-is for proper handling
        if "InterruptProcessingException" in type(e).__name__ or "InterruptProcessingException" in str(type(e)):
            raise
        raise Exception(f"Unexpected error during traditional download: {e}") from e


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
            
            # Check if any segments failed
            if failed_segment_errors:
                error_summary = "; ".join(failed_segment_errors[:5])  # Show up to 5 errors
                if len(failed_segment_errors) > 5:
                    error_summary += f" and {len(failed_segment_errors) - 5} more errors"
                raise Exception(f"Parallel download failed due to segment errors: {error_summary}")
            
            # Retry failed segments with single connection
            if self.failed_segments:
                retry_errors = []
                for seg_id, start, end, *error_info in self.failed_segments:
                    if not self._download_segment_sync(seg_id, start, end):
                        error_msg = error_info[0] if error_info else f"Segment {seg_id} retry failed"
                        retry_errors.append(error_msg)
                
                if retry_errors:
                    error_summary = "; ".join(retry_errors[:3])
                    if len(retry_errors) > 3:
                        error_summary += f" and {len(retry_errors) - 3} more retry failures"
                    raise Exception(f"Segment retry failed: {error_summary}")
            
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
            except:
                pass
            raise Exception(f"Failed to assemble downloaded file: {e}") from e


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
                    if progress_bar:
                        progress_bar.update(bytes_read)
                    if self.progress_callback:
                        self.progress_callback(self.downloaded_bytes, self.total_size)
                
                # Download all segments concurrently
                tasks = [
                    fetch_async_segment(session, self.url, start, end, seg_id, progress_update)
                    for seg_id, start, end in segments
                ]
                
                segment_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for failures and collect error details
                segment_data = {}
                failed_segments = []
                for result in segment_results:
                    if isinstance(result, Exception):
                        failed_segments.append(str(result))
                    else:
                        seg_id, data = result
                        segment_data[seg_id] = data
                
                if failed_segments:
                    error_summary = "; ".join(failed_segments[:3])
                    if len(failed_segments) > 3:
                        error_summary += f" and {len(failed_segments) - 3} more async segment errors"
                    raise Exception(f"Async parallel download failed: {error_summary}")
                
                # Assemble file
                try:
                    async with aiofiles.open(self.file_path, 'wb') as f:
                        for i in range(len(segments)):
                            await f.write(segment_data[i])
                except Exception as e:
                    raise Exception(f"Failed to write assembled async segments to file: {e}") from e
                
                if progress_bar:
                    progress_bar.close()
                return True
                
        except Exception as e:
            if progress_bar:
                try:
                    progress_bar.close()
                except:
                    pass
            # Re-raise exception to propagate to node
            raise Exception(f"Async parallel download failed: {e}") from e


def check_range_support(url: str) -> Tuple[bool, Optional[int]]:
    """Check if server supports range requests and get file size"""
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library not available for range support check")
        
    try:
        # First try HEAD request to get the final URL after redirects
        with requests.head(url, allow_redirects=True, timeout=10) as response:
            response.raise_for_status()
            
            # Get the final redirected URL - this is what we'll actually download from
            final_url = response.url
            
            accepts_ranges = response.headers.get('Accept-Ranges', '').lower() == 'bytes'
            content_length = response.headers.get('Content-Length')
            file_size = int(content_length) if content_length else None
            
            # If Accept-Ranges header is present and says 'bytes', we're good
            if accepts_ranges:
                print(f"[ANYMATIX RANGE] Server explicitly supports Range requests via Accept-Ranges header")
                return True, file_size
            
            # Special handling for known cloud storage services that support ranges but may have signed URLs
            if any(domain in final_url.lower() for domain in [
                'cloudflarestorage.com',  # Cloudflare R2
                'amazonaws.com',          # AWS S3
                's3.amazonaws.com',       # AWS S3
                'digitaloceanspaces.com', # DigitalOcean Spaces
                'storage.googleapis.com', # Google Cloud Storage
                'blob.core.windows.net'   # Azure Blob Storage
            ]):
                print(f"[ANYMATIX RANGE] Assuming Range support for cloud storage URL: {final_url.split('/')[2]}")
                return True, file_size
            
            # If no Accept-Ranges header, try a small range request to test
            # IMPORTANT: Use the SAME final_url to avoid different signed URLs
            if file_size and file_size > 1024:
                print(f"[ANYMATIX RANGE] Testing Range request support (no Accept-Ranges header found)")
                try:
                    test_headers = {'Range': 'bytes=0-1023'}  # Request first 1KB
                    # Use final_url directly with allow_redirects=False to test exact same endpoint
                    with requests.get(final_url, headers=test_headers, stream=True, timeout=10, allow_redirects=False) as test_response:
                        if test_response.status_code == 206:  # Partial Content
                            print(f"[ANYMATIX RANGE] Server supports Range requests (tested with small range)")
                            return True, file_size
                        else:
                            print(f"[ANYMATIX RANGE] Server doesn't support Range requests (got status {test_response.status_code})")
                except Exception as e:
                    print(f"[ANYMATIX RANGE] Range test failed: {e}")
                    # Don't raise here, just return False - this is expected for servers that don't support ranges
            
            return False, file_size
    except requests.RequestException as e:
        raise Exception(f"Failed to check range support for {url}: {e}") from e
    except Exception as e:
        raise Exception(f"Unexpected error checking range support for {url}: {e}") from e


def fetch_parallel(url: str, file_path: str, callback: Optional[Callable[[int, Optional[int]], None]] = None,
                  local_file_size: int = 0, max_connections: int = 8) -> bool:
    """State-of-the-art parallel download with intelligent fallback"""
    
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library not available for parallel download")
    
    # Check server capabilities
    supports_ranges, total_size = check_range_support(url)
    
    if not supports_ranges or not total_size:
        # For signed URLs (like Civitai), try a live range test during actual download
        if not supports_ranges and total_size:
            print(f"[ANYMATIX DOWNLOAD] Server capabilities unknown - will attempt range detection during download")
            # We'll try parallel anyway and fall back if it fails
        else:
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
        # For resume, we'll use traditional method for simplicity
        print(f"[ANYMATIX DOWNLOAD] Resume scenario detected - using traditional download")
        return False
    
    # Choose download strategy based on available dependencies
    try:
        # Try async method first (fastest) if available
        if AIOHTTP_AVAILABLE and AIOFILES_AVAILABLE:
            print(f"[ANYMATIX DOWNLOAD] Using async parallel download strategy")
            async def run_async():
                downloader = AsyncParallelDownloader(url, file_path, total_size, callback, max_connections)
                success = await downloader.download_async()
                if not success:
                    raise Exception(f"Async parallel download failed for {url}")
                return success
            
            # Run async download
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            return loop.run_until_complete(run_async())
        else:
            # Fallback to threaded parallel download
            print(f"[ANYMATIX DOWNLOAD] Using threaded parallel download strategy")
            downloader = SegmentDownloader(url, file_path, total_size, callback, max_connections)
            if not downloader.download_parallel():
                raise Exception(f"Threaded parallel download failed for {url}")
            if not downloader.assemble_file():
                raise Exception(f"Failed to assemble downloaded segments for {url}")
            return True
        
    except Exception as e:
        print(f"[ANYMATIX DOWNLOAD] Parallel download strategy failed: {e}")
        # Re-raise the exception instead of returning False so it propagates to the node
        raise Exception(f"Parallel download failed: {e}")  from e


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
                            # REFERENCE-AWARE DELETION
                            # Check if any OTHER sidecar references this file
                            referenced = False
                            for other_f in os.listdir(root):
                                if other_f.endswith('.json') and other_f != f:
                                    try:
                                        with open(os.path.join(root, other_f), 'r') as other_contents:
                                            other_data = json.load(other_contents)
                                        if other_data.get("file_name") == model_file:
                                            referenced = True
                                            break
                                    except:
                                        pass
                            
                            if not referenced:
                                try:
                                    delete_file_and_cleanup_dir(Path(file_path), dir)
                                    with open(log_path, "a") as log:
                                        log.write(f"Deleted model file: {file_path}\n")
                                except Exception as e:
                                    with open(error_path, "a") as err:
                                        err.write(f"Failed to delete model file: {file_path} - {e}\n")
                            else:
                                with open(log_path, "a") as log:
                                    log.write(f"Skipping model file deletion (still referenced): {file_path}\n")
                            
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

        # EARLY DEDUPLICATION CHECK (using metadata hash if available)
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

        if metadata_hash:
            data["sha256"] = metadata_hash # Pre-set it
            print(f"[ANYMATIX] Early hash check for {data['file_name']}: {metadata_hash}")
            for item in os.listdir(dir):
                if item.endswith(".json") and item != f"{url_hash}.json":
                    try:
                        with open(os.path.join(dir, item), 'r') as f:
                            other_data = json.load(f)
                        if other_data.get("sha256") == metadata_hash:
                            other_file_name = other_data.get("file_name")
                            if other_file_name:
                                other_file_path = os.path.join(dir, other_file_name)
                                if os.path.exists(other_file_path):
                                    print(f"[ANYMATIX] Found existing model with matching hash: {other_file_path}. Using it.")
                                    # Update current sidecar to point to the EXISTING file
                                    data["file_name"] = other_file_name
                                    with open(store_path, 'w') as file:
                                        json.dump(data, file, indent=4)
                                    return other_file_path
                    except Exception as e:
                        print(f"[WARNING] Early deduplication check failed for {item}: {e}")
        local_file_size = 0

        if data["file_size"] is not None and os.path.exists(file_path):
            local_file_size = os.path.getsize(file_path)
            if local_file_size == data["file_size"]:
                return file_path

        downloaded_size = local_file_size

        # STATE-OF-THE-ART PARALLEL DOWNLOAD ATTEMPT
        parallel_success = False
        parallel_exception = None
        if local_file_size == 0 and data["file_size"] is not None:  # Only for fresh downloads
            print(f"[ANYMATIX DOWNLOAD] Attempting parallel download for {data['file_name']} ({data['file_size']} bytes)")
            try:
                parallel_success = fetch_parallel(
                    effective, 
                    file_path, 
                    callback, 
                    local_file_size,
                    max_connections=min(8, max(2, data["file_size"] // (10*1024*1024)))  # Adaptive connections
                )
                if parallel_success:
                    mb_total = data["file_size"] / (1024 * 1024) if data["file_size"] else 0
                    print(f"[ANYMATIX DOWNLOAD] Parallel download completed successfully: {data['file_name']} ({mb_total:.0f}MB)")
                    return file_path
                else:
                    print(f"[ANYMATIX DOWNLOAD] Parallel download was attempted but returned False (likely server doesn't support ranges)")
            except Exception as e:
                print(f"[ANYMATIX DOWNLOAD] Parallel download failed with exception, falling back to traditional: {e}")
                parallel_success = False
                parallel_exception = e

        # TRADITIONAL FALLBACK (backwards compatible)
        if not parallel_success:
            print(f"[ANYMATIX DOWNLOAD] Using traditional download for {data['file_name']}")
            traditional_exception = None
            try:
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
                            
                            # Additional console progress for anymatix terminal
                            if data["file_size"] and downloaded_size % (50 * 1024 * 1024) < l:  # Every 50MB
                                mb_downloaded = downloaded_size / (1024 * 1024)
                                mb_total = data["file_size"] / (1024 * 1024)
                                percent = (downloaded_size / data["file_size"]) * 100
                                print(f"[ANYMATIX PROGRESS] {mb_downloaded:.0f}MB / {mb_total:.0f}MB ({percent:.1f}%)")
                                
                    try:
                        fetch(effective, session, cb, local_file_size)
                    finally:
                        if progress_bar:
                            try:
                                progress_bar.close()
                            except:
                                pass
                        
                        # Final status message for anymatix terminal
                        if downloaded_size == data["file_size"]:
                            mb_final = downloaded_size / (1024 * 1024)
                            print(f"[ANYMATIX DOWNLOAD] Traditional download completed: {mb_final:.0f}MB")
                            
            except Exception as e:
                traditional_exception = e
                print(f"[ANYMATIX DOWNLOAD] Traditional download also failed: {e}")
                
                # If both parallel and traditional failed, raise the more serious exception
                if parallel_exception and traditional_exception:
                    # Prefer parallel exception if it's more descriptive, otherwise use traditional
                    if "Range" in str(parallel_exception) or "connection" in str(parallel_exception).lower():
                        raise parallel_exception
                    else:
                        raise traditional_exception
                elif traditional_exception:
                    raise traditional_exception
                elif parallel_exception:
                    raise parallel_exception
                else:
                    raise Exception(f"Both parallel and traditional download methods failed for {data['file_name']}")

        # Final verification: ensure file exists and has correct size before returning
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Download completed but file not found: {file_path}. "
                f"This may indicate a download failure, filesystem issue, or the file was deleted during download."
            )
        
        if data["file_size"] is not None:
            actual_size = os.path.getsize(file_path)
            if actual_size != data["file_size"]:
                raise Exception(
                    f"Downloaded file size mismatch for {data['file_name']}: "
                    f"expected {data['file_size']} bytes, got {actual_size} bytes. "
                    f"The download may have been interrupted or corrupted."
                )

        # POST-DOWNLOAD DEDUPLICATION
        print(f"[ANYMATIX] Computing hash for deduplication: {file_path}")
        sha256 = compute_file_sha256(file_path).lower()
        data["sha256"] = sha256
        
        # Determine canonical filename: original_sha256.ext
        name_with_hash = data["file_name"]
        parts = name_with_hash.rsplit("_", 1)
        if len(parts) > 1:
            basename = parts[0]
            suffix = parts[1]
            ext_parts = suffix.split(".", 1)
            ext = ("." + ext_parts[1]) if len(ext_parts) > 1 else ""
        else:
            basename_parts = name_with_hash.rsplit(".", 1)
            basename = basename_parts[0]
            ext = ("." + basename_parts[1]) if len(basename_parts) > 1 else ""
        
        canonical_name = f"{basename}_{sha256}{ext}"
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
