import os
import folder_paths

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


    

allowed_dirs=["output","input","models"] 

@routes.get("/{basedir}/{filename:.+}")
async def serve_file(request):
    response = web.Response(text="File not found", status=404)
    
    basedir = request.match_info['basedir']
    
    if basedir in allowed_dirs:
        file_path = os.path.abspath(f"{basedir}/{request.match_info['filename']}")    
        if file_path.startswith(f"{os.path.abspath(os.getcwd())}/{basedir}") and os.path.isfile(file_path) and os.access(file_path, os.R_OK):
            response = web.FileResponse(file_path)
        
    return response
    
resource_extensions=[".ckpt",".safetensors"]
@routes.get("/resources")
async def serve_resources(request):
    print("resources?")
    md = folder_paths.models_dir
    def find(path):
        return os.path.join(md,path)
    dirs = [d for d in os.listdir(md) if os.path.isdir(find(d))]    
    resources = {
        dir: [f for f in os.listdir(find(dir)) if any(f.endswith(ext) for ext in resource_extensions)] for dir in dirs
    }
    return web.json_response(resources)
    

    
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']