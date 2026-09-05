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


def is_valid_json_file(file_path: str) -> bool:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except Exception:
        return False


def compute_file_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hash of a file efficiently."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


CREDENTIAL_QUERY_KEYS = {"token", "api_key", "apikey", "access_token"}


def redact_url(u: str, appended: Optional[str] = None) -> str:
    """Return a safe-to-log URL string.

    Always strips credential query parameters (token, api_key, ...), and in
    addition removes the specific parameters contained in 'appended' when the
    caller knows what it appended.

    It used to return 'u' unchanged when 'appended' was None, which made the
    name a promise it did not keep: the download URL carries the user's
    civitai key as a 'token=' tail, and every message interpolating a raw URL
    put that key into comfyui.runtime.log in cleartext. Measured 2026-08-26 in
    the Anymatix security audit (F3b). The request itself is always built from
    the untouched URL; only what gets LOGGED passes through here.
    """
    try:
        # Parse both URL and appended query tail
        p = urlparse(u)
        current = parse_qsl(p.query, keep_blank_values=True)
        remove_pairs = set(parse_qsl(appended, keep_blank_values=True)) if appended else set()
        # Drop the caller's appended pairs, and mask every credential parameter
        # whoever put it there.
        kept = []
        for key, value in current:
            if (key, value) in remove_pairs:
                continue
            if key.lower() in CREDENTIAL_QUERY_KEYS:
                kept.append((key, "<redacted>"))
                continue
            kept.append((key, value))
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
    """One connection, start to finish — and the only path that can resume."""
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
        raise Exception(f"HTTP request failed during single-stream download: {e}") from e
    except Exception as e:
        # Re-raise InterruptProcessingException as-is for proper handling
        if "InterruptProcessingException" in type(e).__name__ or "InterruptProcessingException" in str(type(e)):
            raise
        raise Exception(f"Unexpected error during single-stream download: {e}") from e


class SegmentDownloader:
    """Many connections at once, each fetching its own byte range."""
    
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
                            segment_id: int, progress_callback: Optional[Callable] = None,
                            part_path: Optional[str] = None) -> int:
    """
    Download one byte range STRAIGHT TO ITS OFFSET in the part file.

    This used to build the segment in memory — `segment_data += chunk` — and
    hand the bytes back to be written once every segment had arrived. With a
    segment size of total/16 that put the WHOLE file in RAM (2.14 GB for
    t3_mtl23ls_v2, plus the copies that `+=` makes), which is a plausible way
    to have a remote ComfyUI killed while it downloads, and is certainly a way
    to make a shared machine swap. Each task opens its own handle: POSIX is
    happy with concurrent writes to disjoint ranges, so no lock is needed and
    the peak is one chunk per connection.
    """
    if not AIOHTTP_AVAILABLE:
        raise ImportError("aiohttp not available")
    if not AIOFILES_AVAILABLE:
        raise ImportError("aiofiles not available")
    if not part_path:
        raise ValueError("fetch_async_segment needs the part file to write into")

    headers = {'Range': f'bytes={start}-{end}'}

    async with session.get(url, headers=headers) as response:
        if response.status not in [206, 200]:
            raise Exception(f"HTTP {response.status}")

        async with aiofiles.open(part_path, 'r+b') as f:
            await f.seek(start)
            async for chunk in response.content.iter_chunked(8192):
                await f.write(chunk)
                if progress_callback:
                    progress_callback(len(chunk))

        return segment_id


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
            # KEEP THE CONNECTIONS ALIVE, WHICH IS THE ENTIRE POINT.
            #
            # force_close=True closes every connection after one response and
            # is mutually exclusive with keepalive_timeout — aiohttp raises
            # "keepalive_timeout cannot be set if force_close is True" the
            # moment the session is built. So the parallel downloader has been
            # dying at construction and silently falling back to one stream on
            # every large download. Keep-alive is what makes segment fetching
            # worth doing at all, so force_close is the one that goes.
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections,
                enable_cleanup_closed=True,
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
                
                # A PART FILE, NEVER THE DESTINATION, AND NEVER SPARSE AT THE
                # DESTINATION. Writing at offsets means the file reaches full
                # size immediately, and `download_file` decides a download is
                # complete by comparing the size on disk with the expected
                # one — so a preallocated destination would announce a
                # half-downloaded weight as finished. The part file carries
                # that risk instead, and only a completed download is renamed
                # into place.
                # THE PATH HANDED IN IS ALREADY THE PART PATH. DO NOT DERIVE
                # ANOTHER ONE.
                #
                # `download_file` builds it once with `part_path_for` and passes
                # THAT into `fetch_parallel`. Appending `.part` here built it a
                # second time, so every parallel download wrote to
                # `<name>.safetensors.part.part` — measured 2026-09-05 on a
                # remote, a complete 323 MB SAM2 weight wearing two suffixes.
                #
                # It is not a cosmetic name. `finalize_download` is then handed
                # `<name>.part`, which never existed, and raises "Download
                # produced no data" on a transfer that finished perfectly; the
                # next run reads `local_file_size` from that same absent
                # `.part`, sees 0, and re-downloads the whole file from byte
                # zero. The `.part` scheme exists to make interrupted downloads
                # resumable, and on this path it could not resume once.
                # `bugs/the-parallel-download-writes-part-part-can`.
                part_path = self.file_path
                os.makedirs(os.path.dirname(part_path) or ".", exist_ok=True)
                async with aiofiles.open(part_path, 'wb') as prealloc:
                    await prealloc.truncate(self.total_size)

                # Download all segments concurrently
                tasks = [
                    fetch_async_segment(session, self.url, start, end, seg_id, progress_update, part_path)
                    for seg_id, start, end in segments
                ]
                
                segment_results = await asyncio.gather(*tasks, return_exceptions=True)

                failed_segments = [str(r) for r in segment_results if isinstance(r, BaseException)]
                if failed_segments:
                    # HAND WHAT LANDED TO THE RESUME PATH, WHICH ALREADY EXISTS.
                    #
                    # Segments finish out of order, so a part file with a hole
                    # in the middle is not a resumable prefix — but the
                    # segments BEFORE the first failure are one, and a single
                    # ordered stream can carry on from there. Truncating to
                    # that boundary is what turns a died-at-18% download into
                    # 18% already done, instead of starting from zero on every
                    # retry (which is what a downloader that only wrote at the
                    # end could never avoid).
                    completed = {
                        idx for idx, r in enumerate(segment_results)
                        if not isinstance(r, BaseException)
                    }
                    prefix = 0
                    for seg_id, seg_start, seg_end in segments:
                        if seg_id not in completed:
                            break
                        prefix = seg_end + 1
                    try:
                        if prefix > 0:
                            with open(part_path, 'r+b') as trim:
                                trim.truncate(prefix)
                            os.replace(part_path, self.file_path)
                            print(
                                f"[ANYMATIX DOWNLOAD] Keeping the {prefix} bytes that landed; "
                                f"a single stream can resume from there"
                            )
                        elif os.path.exists(part_path):
                            os.remove(part_path)
                    except Exception as salvage_error:
                        print(f"[ANYMATIX DOWNLOAD] Could not keep the partial download: {salvage_error}")

                    error_summary = "; ".join(failed_segments[:3])
                    if len(failed_segments) > 3:
                        error_summary += f" and {len(failed_segments) - 3} more async segment errors"
                    raise Exception(f"Async parallel download failed: {error_summary}")

                # Nothing to assemble: every segment wrote its own range.
                written = os.path.getsize(part_path)
                if written != self.total_size:
                    raise Exception(
                        f"Async parallel download wrote {written} bytes, expected {self.total_size}"
                    )
                os.replace(part_path, self.file_path)

                if progress_bar:
                    progress_bar.close()
                return True
                
        except Exception as e:
            if progress_bar:
                try:
                    progress_bar.close()
                except:
                    pass
            # A part file still here was not salvageable: its holes are in the
            # middle, so it is neither a resumable prefix nor a download.
            try:
                stale = self.file_path
                if os.path.exists(stale):
                    os.remove(stale)
            except Exception:
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
        raise Exception(f"Failed to check range support for {redact_url(url)}: {e}") from e
    except Exception as e:
        raise Exception(f"Unexpected error checking range support for {redact_url(url)}: {e}") from e


def fetch_parallel(url: str, file_path: str, callback: Optional[Callable[[int, Optional[int]], None]] = None,
                  local_file_size: int = 0, max_connections: int = 8) -> bool:
    """
    Parallel download with intelligent fallback.

    `file_path` IS THE PATH TO WRITE, and the caller has already made it a part
    path (`part_path_for`). Nothing in here may append `.part` to it: doing so
    produced `<name>.safetensors.part.part`, which `finalize_download` cannot
    see and no later run can resume from.
    """
    
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
        # A resume needs one ordered stream from the byte we stopped at.
        print(f"[ANYMATIX DOWNLOAD] Resuming an interrupted download — one connection, picking up where it stopped")
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
                    raise Exception(f"Async parallel download failed for {redact_url(url)}")
                return success

            # Run the async download in a DEDICATED THREAD with its own loop.
            # A fresh loop in THIS thread is not enough: asyncio refuses to
            # run any loop in a thread that already has one running, and
            # ComfyUI executes nodes with its loop running — so this path
            # raised "Cannot run the event loop while another loop is
            # running" every time, silently demoting every download to
            # single-stream.
            outcome: list = []

            def _runner():
                try:
                    outcome.append(asyncio.run(run_async()))
                except BaseException as e:
                    outcome.append(e)

            t = threading.Thread(target=_runner, name="anymatix-parallel-download")
            t.start()
            t.join()
            if outcome and isinstance(outcome[0], BaseException):
                raise outcome[0]
            return bool(outcome and outcome[0])
        else:
            # Fallback to threaded parallel download
            print(f"[ANYMATIX DOWNLOAD] Using threaded parallel download strategy")
            downloader = SegmentDownloader(url, file_path, total_size, callback, max_connections)
            if not downloader.download_parallel():
                raise Exception(f"Threaded parallel download failed for {redact_url(url)}")
            if not downloader.assemble_file():
                raise Exception(f"Failed to assemble downloaded segments for {redact_url(url)}")
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


def part_path_for(file_path: str) -> str:
    """
    A DOWNLOAD IN PROGRESS MUST NOT WEAR THE NAME OF A FINISHED ONE.

    This wrote straight to the model's own filename and used the bytes already
    there as the resume offset, so an interrupted transfer left a file that
    LOOKED like the model to everything else in the system. A beta tester found
    `ltx-2-19b-dev-fp8.safetensors` on disk at 141 MB of a declared 27 GB —
    0.5% — sitting there looking valid. The downloader would have resumed it
    next time; ComfyUI, asked to load it in the meantime, fails on something
    that has nothing to do with the real cause.

    So the bytes accumulate in `<name>.part` and the final name is given only
    when the size matches what the sidecar declares. Resume still works: it
    resumes from the `.part`. Nothing else in the system can mistake a `.part`
    for a model.
    """
    return file_path + ".part"


def finalize_download(part: str, file_path: str, expected_size, label: str) -> str:
    """
    Give the `.part` its real name, and only if it earned it.

    A short file KEEPS its `.part` and raises: those bytes are worth resuming
    from, and deleting them would throw away what the download already paid
    for. `os.replace` is atomic on the same filesystem, so no reader ever sees
    a half-named file.
    """
    if not os.path.exists(part):
        raise Exception(f"Download produced no data for {label}")
    written = os.path.getsize(part)
    if expected_size is not None and written != expected_size:
        raise Exception(
            f"Incomplete download for {label}: {written} of {expected_size} bytes. "
            f"Kept as {os.path.basename(part)} to resume from."
        )
    os.replace(part, file_path)
    return file_path


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
        part_file = part_path_for(file_path)

        if data["file_size"] is not None:
            # THE FINAL NAME MEANS FINISHED. A file wearing it whose size is not
            # the declared one cannot be the model — under the .part scheme it
            # can only be an artifact of the old one — so it is not resumed
            # from, it is removed.
            if os.path.exists(file_path):
                on_disk = os.path.getsize(file_path)
                if on_disk == data["file_size"]:
                    return file_path
                print(
                    f"[ANYMATIX DOWNLOAD] Discarding a file with the final name and the wrong "
                    f"size for {data['file_name']}: {on_disk} != expected {data['file_size']} bytes"
                )
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            if os.path.exists(part_file):
                local_file_size = os.path.getsize(part_file)
                if local_file_size == data["file_size"]:
                    # It was complete and nobody renamed it — the process died
                    # between the last byte and the rename. Finish the job.
                    return finalize_download(
                        part_file, file_path, data["file_size"], data["file_name"]
                    )
                if local_file_size > data["file_size"]:
                    # Self-heal: a partial larger than the target is corrupt — almost
                    # always a prior resume where the server ignored our Range header and
                    # the full body got appended onto the partial. Discard and start fresh
                    # (otherwise it grows every run and never matches → re-downloads forever).
                    print(
                        f"[ANYMATIX DOWNLOAD] Discarding oversized/corrupt partial for "
                        f"{data['file_name']}: {local_file_size} > expected {data['file_size']} bytes"
                    )
                    try:
                        os.remove(part_file)
                    except Exception:
                        pass
                    local_file_size = 0
        elif data["file_size"] is None and os.path.exists(file_path) and file_path.lower().endswith(".json"):
            if is_valid_json_file(file_path):
                return file_path
            print(f"[ANYMATIX DOWNLOAD] Removing malformed cached JSON before re-download: {file_path}")
            os.remove(file_path)

        downloaded_size = local_file_size

        # PARALLEL DOWNLOAD ATTEMPT — fresh downloads only
        parallel_success = False
        parallel_exception = None
        if local_file_size == 0 and data["file_size"] is not None:  # Only for fresh downloads
            print(f"[ANYMATIX DOWNLOAD] Attempting parallel download for {data['file_name']} ({data['file_size']} bytes)")
            try:
                parallel_success = fetch_parallel(
                    effective,
                    part_file,
                    callback,
                    local_file_size,
                    max_connections=min(8, max(2, data["file_size"] // (10*1024*1024)))  # Adaptive connections
                )
                if parallel_success:
                    mb_total = data["file_size"] / (1024 * 1024) if data["file_size"] else 0
                    print(f"[ANYMATIX DOWNLOAD] Parallel download completed successfully: {data['file_name']} ({mb_total:.0f}MB)")
                    return finalize_download(
                        part_file, file_path, data["file_size"], data["file_name"]
                    )
                else:
                    print(f"[ANYMATIX DOWNLOAD] Parallel download was attempted but returned False (likely server doesn't support ranges)")
            except Exception as e:
                print(f"[ANYMATIX DOWNLOAD] Parallel download failed with exception, falling back to a single stream: {e}")
                parallel_success = False
                parallel_exception = e

        # SINGLE-STREAM FALLBACK — also the resume path: one ordered stream from
        # the byte we stopped at, which segments cannot express.
        if not parallel_success:
            # The parallel attempt may have left a resumable prefix behind (see
            # AsyncParallelDownloader), and `local_file_size` was measured
            # before it ran. Without re-reading it here the fallback opens the
            # file 'wb' and truncates exactly the bytes we just kept.
            if os.path.exists(file_path):
                salvaged = os.path.getsize(file_path)
                if salvaged > local_file_size:
                    print(f"[ANYMATIX DOWNLOAD] Resuming from the {salvaged} bytes the parallel attempt left")
                    local_file_size = salvaged
                    downloaded_size = salvaged
            print(f"[ANYMATIX DOWNLOAD] Using single-stream download for {data['file_name']}")
            single_stream_exception = None
            try:
                file_mode = 'ab' if local_file_size > 0 else 'wb'
                with open(part_file, file_mode) as file:
                    progress_bar = None
                    if TQDM_AVAILABLE and data["file_size"]:
                        try:
                            progress_bar = tqdm(total=data["file_size"], initial=local_file_size)
                        except:
                            progress_bar = None
                        
                    def cb(chunk):
                        nonlocal downloaded_size
                        if (chunk):
                            # Self-heal: if a resume blows past the expected size, the
                            # server ignored our Range request and is re-sending the
                            # whole file from byte 0 onto our partial. Abort NOW rather
                            # than appending the full body (which ballooned to 34GB);
                            # the failed/oversized file is removed below so the next run
                            # restarts clean and then caches correctly.
                            if (
                                data["file_size"] is not None
                                and downloaded_size + len(chunk) > data["file_size"]
                            ):
                                raise Exception(
                                    f"Download overshoot for {data['file_name']}: "
                                    f"server ignored the resume Range (re-sending full body). "
                                    f"Aborting to avoid an unbounded append; will restart clean."
                                )
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
                        if data["file_size"] is not None and downloaded_size == data["file_size"]:
                            mb_final = downloaded_size / (1024 * 1024)
                            print(f"[ANYMATIX DOWNLOAD] Single-stream download completed: {mb_final:.0f}MB")
                            
            except Exception as e:
                single_stream_exception = e
                print(f"[ANYMATIX DOWNLOAD] Single-stream download also failed: {e}")

                # Self-heal: drop the partial so the next run restarts clean — UNLESS
                # this is a user interrupt (then keep the partial for a real resume).
                is_interrupt = "InterruptProcessingException" in type(e).__name__
                if not is_interrupt:
                    try:
                        if os.path.exists(part_file):
                            os.remove(part_file)
                            print(f"[ANYMATIX DOWNLOAD] Removed partial after failure: {part_file}")
                    except Exception:
                        pass

                # If both the parallel and the single-stream attempt failed, raise the more serious exception
                if parallel_exception and single_stream_exception:
                    # Prefer the parallel exception when it says more; otherwise the single-stream one
                    if "Range" in str(parallel_exception) or "connection" in str(parallel_exception).lower():
                        raise parallel_exception
                    else:
                        raise single_stream_exception
                elif single_stream_exception:
                    raise single_stream_exception
                elif parallel_exception:
                    raise parallel_exception
                else:
                    raise Exception(f"Both the parallel and the single-stream download failed for {data['file_name']}")

        # THE RENAME, AND IT IS THE ONLY ONE.
        #
        # Everything above wrote into `<name>.part`. The parallel path returns
        # through `finalize_download` itself; reaching here means the
        # single-stream path ran, and this is where its bytes earn the model's
        # name. A short file raises and KEEPS its `.part`, so the next run
        # resumes instead of starting from zero.
        finalize_download(part_file, file_path, data["file_size"], data["file_name"])

        # Final verification: ensure file exists and has correct size before returning
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Download completed but file not found: {file_path}. "
                f"This may indicate a download failure, filesystem issue, or the file was deleted during download."
            )
        
        if data["file_size"] is not None:
            actual_size = os.path.getsize(file_path)
            if actual_size != data["file_size"]:
                # Self-heal: remove the bad file so the next run starts clean instead
                # of resuming/appending onto it again (the re-download-forever loop).
                try:
                    os.remove(file_path)
                except Exception:
                    pass
                raise Exception(
                    f"Downloaded file size mismatch for {data['file_name']}: "
                    f"expected {data['file_size']} bytes, got {actual_size} bytes. "
                    f"The corrupted file was removed and will be re-downloaded on the next run."
                )

        if file_path.lower().endswith(".json") and not is_valid_json_file(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
            raise Exception(
                f"Downloaded JSON file is malformed for {data['file_name']}. "
                f"The corrupted file was removed and will be re-downloaded on the next run."
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
    
    # Time a single stream for comparison
    try:
        test_file = os.path.join(output_dir, f"benchmark_single_stream_{int(time.time())}")
        start_time = time.time()
        
        with requests.Session() as session:
            with open(test_file, 'wb') as f:
                def cb(chunk):
                    if chunk:
                        f.write(chunk)
                fetch(url, session, cb)
        
        end_time = time.time()
        actual_size = os.path.getsize(test_file)
        
        results["single_stream"] = {
            "time": end_time - start_time,
            "speed_mbps": (actual_size / (1024*1024)) / (end_time - start_time),
            "success": True
        }
        os.remove(test_file)  # Cleanup
        
    except Exception as e:
        results["single_stream"] = {"success": False, "error": str(e)}
    
    # Calculate speedup
    if (results.get("parallel", {}).get("success") and 
        results.get("single_stream", {}).get("success")):
        speedup = results["single_stream"]["time"] / results["parallel"]["time"]
        results["speedup"] = f"{speedup:.2f}x"
        results["bandwidth_improvement"] = f"{results['parallel']['speed_mbps']:.1f} vs {results['single_stream']['speed_mbps']:.1f} MB/s"
    
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
