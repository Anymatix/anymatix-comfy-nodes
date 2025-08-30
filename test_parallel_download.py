#!/usr/bin/env python3
"""
Test script for the enhanced parallel download functionality
"""

import os
import sys
import tempfile
import time

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock the expunge module for testing
class MockExpunge:
    @staticmethod
    def delete_file_and_cleanup_dir(file_path, base_dir):
        if os.path.exists(file_path):
            os.remove(file_path)

# Inject mock into sys.modules before importing fetch
import types
mock_expunge = types.ModuleType('expunge')
mock_expunge.delete_file_and_cleanup_dir = MockExpunge.delete_file_and_cleanup_dir
sys.modules['expunge'] = mock_expunge

# Now we can import our fetch module
from fetch import check_range_support, download_file

def test_exception_handling():
    """Test that exceptions are properly raised and not swallowed"""
    print("=== Testing Exception Handling ===")
    
    # Test with invalid URL
    invalid_url = "https://this-domain-does-not-exist-12345.com/file.bin"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print(f"Testing exception propagation with invalid URL: {invalid_url}")
            download_file(invalid_url, temp_dir)
            print("❌ ERROR: Expected exception was not raised!")
            return False
        except Exception as e:
            print(f"✅ SUCCESS: Exception properly raised: {e}")
            
    # Test with URL that exists but file doesn't
    not_found_url = "https://httpbin.org/status/404" 
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print(f"Testing 404 error handling: {not_found_url}")
            download_file(not_found_url, temp_dir)
            print("❌ ERROR: Expected 404 exception was not raised!")
            return False
        except Exception as e:
            print(f"✅ SUCCESS: 404 error properly raised: {e}")
            
    # Test with URL that would timeout (simulated network error)
    timeout_url = "https://httpbin.org/delay/30"  # 30 second delay
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print(f"Testing timeout handling (this may take a moment): {timeout_url}")
            # This should timeout and raise an exception
            download_file(timeout_url, temp_dir)
            print("⚠️  WARNING: Timeout test didn't raise exception (maybe the server responded quickly)")
        except Exception as e:
            print(f"✅ SUCCESS: Timeout error properly raised: {e}")
            
    print("Exception handling tests completed.\n")
    return True

def test_range_support():
    """Test range request detection"""
    print("=== Testing Range Support Detection ===")
    
    test_urls = [
        "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
        "https://github.com/Stability-AI/generative-models/raw/main/assets/logo.png"
    ]
    
    for url in test_urls:
        try:
            supports_ranges, file_size = check_range_support(url)
            print(f"URL: {url}")
            print(f"  Range Support: {supports_ranges}")
            print(f"  File Size: {file_size:,} bytes" if file_size else "  File Size: Unknown")
            print()
        except Exception as e:
            print(f"Error testing {url}: {e}")

def test_small_download():
    """Test download with a small file to verify backwards compatibility"""
    print("=== Testing Small File Download (Traditional Method) ===")
    
    # Small file that should use traditional download
    url = "https://github.com/Stability-AI/generative-models/raw/main/assets/logo.png"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Downloading to: {temp_dir}")
        
        start_time = time.time()
        try:
            result = download_file(url, temp_dir, callback=None)
            end_time = time.time()
            
            if result and os.path.exists(result):
                size = os.path.getsize(result)
                print(f"✅ Download successful!")
                print(f"   File: {result}")
                print(f"   Size: {size:,} bytes")
                print(f"   Time: {end_time - start_time:.2f} seconds")
            else:
                print(f"❌ Download failed - no file created")
                
        except Exception as e:
            print(f"❌ Download failed with error: {e}")

def test_parallel_capabilities():
    """Test the parallel download system components"""
    print("=== Testing Parallel Download Components ===")
    
    # Test with a URL that supports ranges
    url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
    
    supports_ranges, file_size = check_range_support(url)
    print(f"Range Support: {supports_ranges}")
    print(f"File Size: {file_size:,} bytes" if file_size else "File Size: Unknown")
    
    if supports_ranges and file_size:
        print(f"✅ This file would benefit from parallel download")
        print(f"   Estimated segments: {max(1, file_size // (8*1024*1024))}")
        print(f"   Recommended connections: {min(8, max(2, file_size // (10*1024*1024)))}")
    else:
        print("ℹ️  This file would use traditional download")

if __name__ == "__main__":
    print("🚀 AnymatixFetcher Parallel Download Test Suite")
    print("=" * 60)
    
    test_exception_handling()
    test_range_support()
    test_small_download() 
    test_parallel_capabilities()
    
    print("=" * 60)
    print("✅ All tests completed!")
    print("\nNote: For full parallel download testing, run with a large model URL")
    print("The system will automatically choose the best download method.")
