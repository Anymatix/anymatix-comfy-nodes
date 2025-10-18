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
)
from .anymatix_image_save import Anymatix_Image_Save
from .anymatix_maskimage import AnymatixMaskImage
from .anymatix_image_to_video import AnymatixImageToVideo
from .anymatix_save_animated_mp4 import AnymatixSaveAnimatedMP4

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
    "AnymatixUNETLoaderGGUF": AnymatixUNETLoaderGGUF,
    "AnymatixLoraLoaderModelOnly": AnymatixLoraLoaderModelOnly,
    "AnymatixDualCLIPLoader": AnymatixDualCLIPLoader,
    "AnymatixTripleCLIPLoader": AnymatixTripleCLIPLoader,
    "AnymatixQuadrupleCLIPLoader": AnymatixQuadrupleCLIPLoader,
    "AnymatixCLIPLoader2": AnymatixCLIPLoader2,
    "AnymatixAudioEncoderLoader": AnymatixAudioEncoderLoader,
    "AnymatixSAM2Loader": AnymatixSAM2Loader,
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
    "AnymatixUNETLoaderGGUF": "Anymatix UNET Loader GGUF",
    "AnymatixLoraLoaderModelOnly": "Anymatix Lora Loader Model Only",
    "AnymatixDualCLIPLoader": "Anymatix Dual CLIP Loader",
    "AnymatixTripleCLIPLoader": "Anymatix Triple CLIP Loader",
    "AnymatixQuadrupleCLIPLoader": "Anymatix Quadruple CLIP Loader",
    "AnymatixCLIPLoader2": "Anymatix CLIP Loader 2",
    "AnymatixAudioEncoderLoader": "Anymatix Audio Encoder Loader",
    "AnymatixSAM2Loader": "Anymatix SAM2 Loader",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"anymatix: running on host {socket.gethostname()}")
allowed_dirs = ["output", "input", "models"]

routes = PromptServer.instance.routes

# Heartbeat monitoring state (timer-reset model)
_heartbeat_timeout_seconds: int = 300
_heartbeat_timer_task: asyncio.Task | None = None
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

    Each POST resets the server-side watchdog timer. The endpoint returns the
    configured number of seconds after which the process will exit if no
    further heartbeats arrive. The timeout is server-configured only.
    """
    global _heartbeat_timer_task, _heartbeat_timeout_seconds
    try:
        # Reset the watchdog timer: cancel previous timer task and create a new one
        async with _heartbeat_lock:
            if _heartbeat_timer_task is not None:
                try:
                    _heartbeat_timer_task.cancel()
                except Exception:
                    pass
            loop = asyncio.get_event_loop()
            _heartbeat_timer_task = loop.create_task(_heartbeat_timer(_heartbeat_timeout_seconds))
        # Return the server-configured timeout so the client can schedule refreshes
        return web.json_response({"status": "ok", "seconds_until_death": _heartbeat_timeout_seconds})
    except Exception as e:
        print(f"anymatix: heartbeat endpoint error: {e}")
        return web.json_response({"status": "error", "error": str(e)}, status=500)


@routes.post("/anymatix/uploadAsset")
async def upload_asset(request):
    reader = await request.multipart()

    hash_value = None
    file_extension = None

    # Iterate over the parts in the multipart form
    async for part in reader:
        if part.name == "hash":  # Retrieve the 'hash' field
            hash_value = await part.text()
        elif part.name == "extension":  # Retrieve the 'extension' field
            file_extension = await part.text()
        elif part.name == "file":  # Handle the file upload
            file_path = os.path.join(
                folder_paths.input_directory, f"{hash_value}.{file_extension}"
            )
            with open(file_path, "wb") as output_file:
                while chunk := await part.read_chunk():  # Read file in chunks
                    output_file.write(chunk)
    # TODO: choose a temporary file name first, then check file hash, then rename

    # This is done from the client in js:
    # const formData = new FormData()
    # formData.append('file', file)
    # formData.append('hash', h)
    # file = data.get('file')

    # if not file:
    #     return web.Response(status=400, text="File is required")

    # extension = data.get('extension')
    # if not extension:
    #     return web.Response(status=400, text="extension is required")

    # import hashlib

    # # Compute the SHA256 hash of the file
    # file_hash = hashlib.sha256()
    # while True:
    #     chunk = file.read(8192)
    #     if not chunk:
    #         break
    #     file_hash.update(chunk)
    # hash_value = file_hash.hexdigest()

    # file_path = os.path.join(
    #     folder_paths.input_directory, f"{hash_value}.{extension}")

    # with open(file_path, 'wb') as f:
    #     file.seek(0)
    #     while True:
    #         chunk = file.read(8192)
    #         if not chunk:
    #             break
    #         f.write(chunk)

    return web.Response(status=200)


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
