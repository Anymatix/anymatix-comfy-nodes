import sys
import asyncio
from .fetch import delete_files
from .expunge import *
import json
from typing import Dict
import app.logger
import folder_paths
from aiohttp import web
import os
import hashlib
import time
import socket
from server import PromptServer
import app

from .anymatix_checkpoint_fetcher import (
    AnymatixCheckpointFetcher,
    AnymatixCheckpointLoader,
    AnymatixFetcher,
    AnymatixLoraLoader,
    AnymatixUpscaleModelLoader,
    AnymatixCLIPLoader,
    AnymatixUNETLoader,
    AnymatixVAELoader,
    AnymatixCLIPVisionLoader,
    AnymatixUNETLoaderGGUF,
    AnymatixLoraLoaderModelOnly,
    AnymatixDualCLIPLoader,
    AnymatixTripleCLIPLoader,
    AnymatixQuadrupleCLIPLoader,
    AnymatixCLIPLoader2,
    AnymatixAudioEncoderLoader,
    AnymatixSAM2Loader,
    AnymatixSeedVR2LoadDiTModel,
 #   AnymatixSeedVR2LoadVAEModel,
)
from .anymatix_image_save import Anymatix_Image_Save
from .anymatix_maskimage import AnymatixMaskImage
from .anymatix_image_to_video import AnymatixImageToVideo
from .anymatix_save_animated_mp4 import AnymatixSaveAnimatedMP4
from .anymatix_save_audio import AnymatixSaveAudio
from .anymatix_save_json import AnymatixSaveJson
from .anymatix_mask2SAM import AnymatixMaskToSAMcoord

NODE_CLASS_MAPPINGS = {
    "AnymatixCheckpointFetcher": AnymatixCheckpointFetcher,
    "AnymatixCheckpointLoader": AnymatixCheckpointLoader,
    "AnymatixLoraLoader": AnymatixLoraLoader,
    "AnymatixFetcher": AnymatixFetcher,
    "AnymatixImageSave": Anymatix_Image_Save,
    "AnymatixUpscaleModelLoader": AnymatixUpscaleModelLoader,
    "AnymatixMaskImage": AnymatixMaskImage,
    "AnymatixCLIPLoader": AnymatixCLIPLoader,
    "AnymatixUNETLoader": AnymatixUNETLoader,
    "AnymatixVAELoader": AnymatixVAELoader,
    "AnymatixCLIPVisionLoader": AnymatixCLIPVisionLoader,
    "AnymatixImageToVideo": AnymatixImageToVideo,
    "AnymatixSaveAnimatedMP4": AnymatixSaveAnimatedMP4,
    "AnymatixSaveAudio": AnymatixSaveAudio,
    "AnymatixUNETLoaderGGUF": AnymatixUNETLoaderGGUF,
    "AnymatixLoraLoaderModelOnly": AnymatixLoraLoaderModelOnly,
    "AnymatixDualCLIPLoader": AnymatixDualCLIPLoader,
    "AnymatixTripleCLIPLoader": AnymatixTripleCLIPLoader,
    "AnymatixQuadrupleCLIPLoader": AnymatixQuadrupleCLIPLoader,
    "AnymatixCLIPLoader2": AnymatixCLIPLoader2,
    "AnymatixAudioEncoderLoader": AnymatixAudioEncoderLoader,
    "AnymatixSAM2Loader": AnymatixSAM2Loader,
    "AnymatixSaveJson": AnymatixSaveJson,
    "AnymatixMaskToSAMcoord": AnymatixMaskToSAMcoord,
    "AnymatixSeedVR2LoadDiTModel": AnymatixSeedVR2LoadDiTModel,
 #   "AnymatixSeedVR2LoadVAEModel": AnymatixSeedVR2LoadVAEModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnymatixCheckpointFetcher": "Anymatix Checkpoint Fetcher",
    "AnymatixCheckpointLoader": "Anymatix Checkpoint Loader",
    "AnymatixFetcher": "Anymatix Fetcher",
    "AnymatixLoraLoader": "Anymatix Lora Loader",
    "AnymatixImageSave": "Anymatix Image Save",
    "AnymatixUpscaleModelLoader": "Anymatix Upscale Model Loader",
    "AnymatixMaskImage": "Anymatix Mask Image",
    "AnymatixCLIPLoader": "Anymatix CLIP Loader",
    "AnymatixUNETLoader": "Anymatix UNET Loader",
    "AnymatixVAELoader": "Anymatix VAE Loader",
    "AnymatixCLIPVisionLoader": "Anymatix CLIP Vision Loader",
    "AnymatixImageToVideo": "Anymatix Image To Video",
    "AnymatixSaveAnimatedMP4": "Anymatix Save Animated MP4",
    "AnymatixSaveAudio": "Anymatix Save Audio",
    "AnymatixUNETLoaderGGUF": "Anymatix UNET Loader GGUF",
    "AnymatixLoraLoaderModelOnly": "Anymatix Lora Loader Model Only",
    "AnymatixDualCLIPLoader": "Anymatix Dual CLIP Loader",
    "AnymatixTripleCLIPLoader": "Anymatix Triple CLIP Loader",
    "AnymatixQuadrupleCLIPLoader": "Anymatix Quadruple CLIP Loader",
    "AnymatixCLIPLoader2": "Anymatix CLIP Loader 2",
    "AnymatixAudioEncoderLoader": "Anymatix Audio Encoder Loader",
    "AnymatixSAM2Loader": "Anymatix SAM2 Loader",
    "AnymatixSaveJson": "Anymatix Save Json",
    "AnymatixMaskToSAMcoord": "Anymatix Mask To SAM coord",
    "AnymatixSeedVR2LoadDiTModel": "Anymatix SeedVR2 Load DiT Model",
 #   "AnymatixSeedVR2LoadVAEModel": "Anymatix SeedVR2 Load VAE Model",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"anymatix: running on host {socket.gethostname()}")
allowed_dirs = ["output", "input", "models"]

routes = PromptServer.instance.routes

# Heartbeat monitoring state (timer-reset model)
# The client sends the timeout in the POST body, and the server will exit
# if no heartbeat is received within that timeout
_heartbeat_timer_task: asyncio.Task | None = None
_heartbeat_timeout_seconds: int = 300  # Last received timeout, for reporting
_heartbeat_lock = asyncio.Lock()


async def _heartbeat_timer(timeout_seconds: int):
    """Wait timeout_seconds and then exit the process.

    This task is created on first heartbeat and recreated on each subsequent
    heartbeat. If it ever completes, it will force-exit the process.
    """
    try:
        await asyncio.sleep(timeout_seconds)
        print(f"anymatix: heartbeat timeout (no heartbeat for {timeout_seconds}s) - exiting")
        os._exit(1)
    except asyncio.CancelledError:
        # Timer was reset/cancelled by a newer heartbeat; that's normal
        return
    except Exception as e:
        print(f"anymatix: heartbeat timer error: {e}")


@routes.get("/anymatix/log")
async def get_log(request):
    return web.json_response(list(app.logger.get_logs()))


@routes.post('/anymatix/heartbeat')
async def serve_heartbeat(request):
    """Receive heartbeat pings from the Anymatix app.

    The client sends the desired timeout in the POST body as {"timeout": seconds}.
    Each POST resets the server-side watchdog timer to that timeout.
    If no heartbeat arrives within the timeout, the process exits.
    """
    global _heartbeat_timer_task, _heartbeat_timeout_seconds
    try:
        # Read timeout from request body
        data = await request.json()
        timeout = data.get("timeout", 60)  # Default 60s if not specified
        _heartbeat_timeout_seconds = timeout
        print(f"anymatix: heartbeat received, timeout={timeout}s")
        
        # Reset the watchdog timer: cancel previous timer task and create a new one
        async with _heartbeat_lock:
            if _heartbeat_timer_task is not None:
                try:
                    _heartbeat_timer_task.cancel()
                except Exception:
                    pass
            loop = asyncio.get_event_loop()
            _heartbeat_timer_task = loop.create_task(_heartbeat_timer(timeout))
        return web.json_response({"status": "ok", "seconds_until_death": timeout})
    except Exception as e:
        print(f"anymatix: heartbeat endpoint error: {e}")
        return web.json_response({"status": "error", "error": str(e)}, status=500)


@routes.post("/anymatix/uploadAsset")
async def upload_asset(request):
    """
    Upload an asset file to the ComfyUI input directory.
    Uses atomic writes and hash verification to prevent incomplete uploads.
    Supports resumable uploads for large files.
    """
    reader = await request.multipart()

    hash_value = None
    file_extension = None
    temp_path = None
    resume_offset = 0

    try:
        # Iterate over the parts in the multipart form
        async for part in reader:
            if part.name == "hash":
                hash_value = await part.text()
            elif part.name == "extension":
                file_extension = await part.text()
            elif part.name == "resume":
                # Client requests to resume from previous upload
                resume_requested = await part.text()
                resume_offset = int(resume_requested) if resume_requested.isdigit() else 0
            elif part.name == "file":
                if not hash_value or not file_extension:
                    return web.Response(status=400, text="Hash and extension must be provided before file")
                
                # Final destination path
                file_path = os.path.join(
                    folder_paths.input_directory, f"{hash_value}.{file_extension}"
                )
                
                # Write to temporary file first (atomic write pattern)
                # Use same .tmp convention as downloads (fetch.py uses .segment_N for parallel, .tmp for temp)
                temp_path = f"{file_path}.tmp"
                
                # Check if partial upload exists (resumable upload)
                existing_size = 0
                if resume_offset > 0 and os.path.exists(temp_path):
                    existing_size = os.path.getsize(temp_path)
                    if existing_size == resume_offset:
                        print(f"anymatix: resuming upload of {hash_value}.{file_extension} from {existing_size} bytes")
                    else:
                        # Size mismatch, start over
                        print(f"anymatix: resume offset mismatch (expected {resume_offset}, found {existing_size}), restarting upload")
                        existing_size = 0
                        resume_offset = 0
                
                try:
                    # Open in append mode if resuming, otherwise write mode
                    mode = "ab" if existing_size > 0 and resume_offset > 0 else "wb"
                    with open(temp_path, mode) as output_file:
                        bytes_written = 0
                        while chunk := await part.read_chunk():
                            output_file.write(chunk)
                            bytes_written += len(chunk)
                    
                    total_written = existing_size + bytes_written
                    print(f"anymatix: uploaded {bytes_written} bytes ({total_written} total) for {hash_value}.{file_extension}")
                    
                except Exception as write_error:
                    # Don't delete temp file on write error - allows resume
                    print(f"anymatix: upload write error (temp file preserved for resume): {write_error}")
                    current_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
                    return web.Response(
                        status=500, 
                        text=f"Upload write failed: {str(write_error)}",
                        headers={"X-Resume-Offset": str(current_size)}
                    )
                
                # Verify the uploaded file hash matches expected hash
                try:
                    computed_hash = hashlib.sha256()
                    with open(temp_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(8192), b''):
                            computed_hash.update(chunk)
                    
                    if computed_hash.hexdigest() != hash_value:
                        # Hash mismatch - delete temp file
                        os.remove(temp_path)
                        print(f"anymatix: upload hash mismatch - expected {hash_value}, got {computed_hash.hexdigest()}")
                        return web.Response(status=400, text="Hash verification failed - file corrupted during upload")
                except Exception as hash_error:
                    # Don't delete temp file on hash error - might be partial upload
                    print(f"anymatix: upload hash verification error: {hash_error}")
                    current_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
                    return web.Response(
                        status=500, 
                        text=f"Hash verification failed: {str(hash_error)}",
                        headers={"X-Resume-Offset": str(current_size)}
                    )
                
                # Atomic rename - if this succeeds, the file is complete and verified
                try:
                    os.rename(temp_path, file_path)
                    print(f"anymatix: successfully uploaded and verified {hash_value}.{file_extension} ({total_written} bytes)")
                except Exception as rename_error:
                    # Don't delete temp file - can retry rename
                    print(f"anymatix: upload rename error (temp file preserved): {rename_error}")
                    return web.Response(status=500, text=f"Failed to finalize upload: {str(rename_error)}")
        
        if not hash_value or not file_extension:
            return web.Response(status=400, text="Missing required fields: hash, extension, and file")
        
        return web.Response(status=200, text="Upload successful")
    
    except Exception as e:
        # Don't clean up temp file on unexpected error - allows resume/retry
        print(f"anymatix: upload error (temp file preserved for resume): {e}")
        if temp_path and os.path.exists(temp_path):
            current_size = os.path.getsize(temp_path)
            return web.Response(
                status=500, 
                text=f"Upload failed: {str(e)}",
                headers={"X-Resume-Offset": str(current_size)}
            )


outdir = f"{folder_paths.output_directory}/anymatix/results"
os.makedirs(outdir, exist_ok=True)


@routes.get("/anymatix/storage_location")
async def serve_storage_location(request):
    return web.json_response({
        "models_dir": folder_paths.models_dir
    })


@routes.get("/anymatix/reboot")
async def serve_reboot(request):
    # Check if deep restart is requested
    deep_restart = request.rel_url.query.get("deep", "false").lower() == "true"
    
    if deep_restart:
        print("anymatix: scheduling deep restart (same PID, module reload)")
        # Schedule the deep restart asynchronously
        asyncio.create_task(deep_restart_after_delay(2))
        return web.json_response({"status": "scheduled", "message": "Deep restart scheduled", "type": "deep"})
    else:
        print("anymatix: performing soft restart - clearing queue and freeing memory")
        
        try:
            # Import the server instance to access internal methods
            from server import PromptServer
            server_instance = PromptServer.instance
            
            # 1. Interrupt any current processing
            import nodes
            nodes.interrupt_processing()
            
            # 2. Clear the queue
            if hasattr(server_instance, 'prompt_queue'):
                server_instance.prompt_queue.wipe_queue()
                print("anymatix: queue cleared")
            
            # 3. Free memory and unload models
            import comfy.model_management
            comfy.model_management.soft_empty_cache()
            comfy.model_management.unload_all_models()
            print("anymatix: models unloaded and memory freed")
            
            # 4. Close and reopen websocket connections to refresh clients
            if hasattr(server_instance, 'sockets'):
                for sid in list(server_instance.sockets.keys()):
                    try:
                        await server_instance.sockets[sid].close()
                    except:
                        pass
                server_instance.sockets.clear()
                print("anymatix: websocket connections refreshed")
            
            # 5. Clear any remaining execution state
            import execution
            if hasattr(execution, 'current_executed'):
                execution.current_executed.clear()
                
            print("anymatix: soft restart completed successfully")
            return web.json_response({"status": "success", "message": "Soft restart completed", "type": "soft"})
            
        except Exception as e:
            print(f"anymatix: error during soft restart: {e}")
            import traceback
            traceback.print_exc()
            return web.json_response({"status": "error", "message": str(e)}, status=500)


# DISABLED: This function creates a new process that escapes PowerShell job object control
# async def reboot_after_delay(delay: int):
#     await asyncio.sleep(delay)
#     os.execv(sys.executable, [sys.executable] + sys.argv)

async def deep_restart_after_delay(delay: int):
    """
    Attempt a deeper restart by reloading core modules while keeping the same PID.
    This maintains job object control while refreshing the ComfyUI state.
    """
    await asyncio.sleep(delay)
    
    try:
        print("anymatix: performing deep restart - reloading core modules")
        
        # 1. Stop all current processing
        import nodes
        nodes.interrupt_processing()
        
        # 2. Clear execution state
        import execution
        if hasattr(execution, 'current_executed'):
            execution.current_executed.clear()
            
        # 3. Clear model cache
        import comfy.model_management
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        
        # 4. Clear node mappings and reload
        nodes.NODE_CLASS_MAPPINGS.clear()
        nodes.NODE_DISPLAY_NAME_MAPPINGS.clear()
        
        # 5. Reload core modules
        import importlib
        import sys
        
        modules_to_reload = [
            'nodes',
            'execution', 
            'comfy.model_management',
            'folder_paths'
        ]
        
        for module_name in modules_to_reload:
            if module_name in sys.modules:
                try:
                    importlib.reload(sys.modules[module_name])
                    print(f"anymatix: reloaded {module_name}")
                except Exception as e:
                    print(f"anymatix: failed to reload {module_name}: {e}")
        
        # 6. Re-initialize nodes
        import nodes
        nodes.init_extra_nodes()
        
        # 7. Clear server state
        from server import PromptServer
        server_instance = PromptServer.instance
        if hasattr(server_instance, 'prompt_queue'):
            server_instance.prompt_queue.wipe_queue()
            server_instance.prompt_queue.wipe_history()
            
        print("anymatix: deep restart completed - same PID maintained")
        
    except Exception as e:
        print(f"anymatix: error during deep restart: {e}")
        import traceback
        traceback.print_exc()


@routes.post("/anymatix/expunge")
async def serve_expunge(request):
    print("anymatix: expunging cache")
    data = await request.json()
    input_assets: list[str] = data.get("inputAssets", [])
    computation_results: list[str] = data.get("computationResults", [])
    delete_hashes: list[str] = data.get("delete", [])

    # Perform normal expunge
    await expunge_differentiated(
        input_assets, computation_results, outdir, folder_paths.input_directory
    )

    # Clean up temporary upload files (.tmp) in input directory
    # These are incomplete uploads that should be removed during cache cleanup
    temp_files_deleted = []
    try:
        for tmp_file in Path(folder_paths.input_directory).glob("*.tmp"):
            try:
                # Check if file is old (more than 24 hours)
                # Recent .tmp files might be active uploads
                if time.time() - tmp_file.stat().st_mtime > 86400:  # 24 hours
                    tmp_file.unlink()
                    temp_files_deleted.append(str(tmp_file.name))
            except Exception as e:
                print(f"anymatix: failed to delete temp upload file {tmp_file}: {e}")
        if temp_files_deleted:
            print(f"anymatix: cleaned up {len(temp_files_deleted)} old temporary upload files")
    except Exception as e:
        print(f"anymatix: error during temp file cleanup: {e}")

    # If delete parameter is present, delete those hashes from results and input directories
    from .expunge import hash_pattern, input_asset_pattern
    deleted = []
    for h in delete_hashes:
        # Delete computation result dir if hash matches
        if hash_pattern.match(h):
            result_path = Path(outdir) / h
            if result_path.exists() and result_path.is_dir():
                try:
                    shutil.rmtree(result_path)
                    deleted.append(str(result_path))
                except Exception as e:
                    print(f"Failed to delete computation result {result_path}: {e}")
        # Delete input asset file if hash matches asset pattern
        for f in Path(folder_paths.input_directory).glob(f"{h}.*"):
            if input_asset_pattern.match(f.name):
                try:
                    os.remove(f)
                    deleted.append(str(f))
                except Exception as e:
                    print(f"Failed to delete input asset {f}: {e}")
        # Also delete any associated .tmp file for this hash
        tmp_file = Path(folder_paths.input_directory) / f"{h}.*.tmp"
        for tmp in Path(folder_paths.input_directory).glob(f"{h}.*.tmp"):
            try:
                tmp.unlink()
                deleted.append(str(tmp))
            except Exception as e:
                print(f"Failed to delete temp file {tmp}: {e}")
    if deleted:
        print(f"anymatix: deleted hashes: {deleted}")
    return web.Response(status=200)


@routes.get("/anymatix/cache_size")
async def serve_cache_size(request):
    result = await count_outputs(outdir)
    return web.json_response(result)


@routes.post("/anymatix/delete_resource")
async def serve_delete(request):
    print("anymatix: deleting resource")
    data = await request.json()
    url = data["url"]
    delete_files(url, folder_paths.models_dir)
    return web.Response(status=200)


@routes.get("/anymatix/{basedir}/{filename:.+}")
async def serve_file(request):
    response = web.Response(text="File not found", status=404)

    # TODO: check also dirmap in anymatix_checkpoint_fetcher, reconcile that with "allowed_dirs"
    basedir = request.match_info["basedir"]

    if basedir in allowed_dirs:
        # TODO: the check on getcwd is plain wrong (if the cwd is not what I expected). Determine from the current script?
        file_path = os.path.abspath(f"{basedir}/{request.match_info['filename']}")
        base = os.path.abspath(os.path.join(os.getcwd(), basedir))
        if (
            file_path.startswith(base)
            and os.path.isfile(file_path)
            and os.access(file_path, os.R_OK)
        ):
            # TODO: TEMPORARY, WAITING FOR BETTER CLEANUP STRATEGIES
            if "forget" in request.rel_url.query:
                try:
                    os.remove(file_path)
                    print(f"anymatix: deleted resource {file_path}")
                    return web.Response(text="Resource deleted (forget)", status=200)
                except Exception as e:
                    return web.Response(text=f"Failed to delete resource: {e}", status=500)
            response = web.FileResponse(file_path)
        # else:
        #     print("****** anymatix", file_path, "is not allowed or does not exist", os.getcwd())

    return response


# resource_extensions = [".ckpt", ".safetensors"]


@routes.get("/anymatix/resources")
async def serve_resources(_request):
    json_files = (
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(folder_paths.models_dir)
        for filename in filenames
        if filename.endswith(".json")
    )

    def get_json_data(path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    def get_type(path: str) -> str:
        return path.removeprefix(folder_paths.models_dir).split(os.sep)[1]

    def get_contents(path: str):
        data = get_json_data(path)
        type = get_type(path)
        file_path = path.removeprefix(os.path.join(folder_paths.models_dir, type))
        return (data, type, file_path)

    contents = map(get_contents, json_files)

    result: Dict[str, list] = {}

    for data, type, file_path in contents:
        if type in result:
            result[type].append(data)
        else:
            result[type] = [data]

    return web.json_response(result)
