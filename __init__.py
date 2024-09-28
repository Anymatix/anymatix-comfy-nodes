import os

from .anymatix_checkpoint_fetcher import AnymatixCheckpointFetcher, AnymatixCheckpointLoader

NODE_CLASS_MAPPINGS = {
    "AnymatixCheckpointFetcher": AnymatixCheckpointFetcher,   
    "AnymatixCheckpointLoader": AnymatixCheckpointLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnymatixCheckpointFetcher": "Anymatix Checkpoint Fetcher",    
    "AnymatixCheckpointLoader": "Anymatix Checkpoint Loader"
}

from aiohttp import web
from server import PromptServer
routes = PromptServer.instance.routes

@routes.get("/output/{filename:.+}")
async def serve_output(request):
    file_path = os.path.abspath(f"output/{request.match_info['filename']}")    
    if file_path.startswith(f"{os.path.abspath(os.getcwd())}/output") and os.path.isfile(file_path) and os.access(file_path, os.R_OK):
        return web.FileResponse(file_path)
    else:
        return web.Response(text="File not found", status=404)
    
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']