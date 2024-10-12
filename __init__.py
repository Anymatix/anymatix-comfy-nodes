import json
import os
import socket
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

@routes.get("/anymatix/{basedir}/{filename:.+}")
async def serve_file(request):
    print("SERVING FILE",request.url)
    response = web.Response(text="File not found", status=404)
    
    basedir = request.match_info['basedir']
    
    if basedir in allowed_dirs:        
        # TODO: the check on getcwd is plain wrong (if the cwd is not what I expected). Determine from the current script?        
        file_path = os.path.abspath(f"{basedir}/{request.match_info['filename']}")    
        if file_path.startswith(f"{os.path.abspath(os.getcwd())}/{basedir}") and os.path.isfile(file_path) and os.access(file_path, os.R_OK):
            response = web.FileResponse(file_path)
        else:
            print("****",f"{os.path.abspath(os.getcwd())}/{basedir}",os.path.isfile(file_path),os.access(file_path, os.R_OK))
            
    return response
    
resource_extensions=[".ckpt",".safetensors"]

@routes.get("/anymatix/resources") # TODO -> anymatix/resources
async def serve_resources(request):
    md = folder_paths.models_dir
    
    resources = {}
    
    def find(path: str):
        return os.path.join(md,path)
    
    def get_json_data(path: str) -> dict:
        with open(find(path),'r') as f:
            return json.load(f)
    
    def get_json_filename(path: str):
        data = get_json_data(path)
        if "file_name" in data:
            return data["file_name"]
        return None
    
    def get_files(path: str):
        return (f for f in os.listdir(find(path)) if any(f.endswith(ext) for ext in resource_extensions))
    
    def get_json_files(path):
        return [os.path.join(path,f) for f in os.listdir(find(path)) if f.endswith(".json")]
    
    def get_resources(path):
        print("get_resources",path)
        json_files = get_json_files(path)
        print("json_files",json_files)
        
        references = { filename: json_file for json_file in json_files if (filename := get_json_filename(json_file)) if filename is not None }
        
        files = [ ]
        for file in get_files(path):
            if file in references:
                files.append(get_json_data(references[file]))
            else:
                files.append(file)
        
        return files
        
    
    for dir in os.listdir(md):
        if os.path.isdir(find(dir)):
            resources[dir] = get_resources(dir)
    
    # TODO: this is not working, as the json files are not being read correctly.
    # TODO: the json files are not being read correctly.
    # TODO: the json files are not being read correctly.

    # Old:
    # def find(path):
    #     return os.path.join(md,path)

    # dirs = [d for d in os.listdir(md) if os.path.isdir(find(d))]    
    
    # resources = {
    #     dir: [f for f in os.listdir(find(dir)) if any(f.endswith(ext) for ext in resource_extensions)] for dir in dirs
    # }
    return web.json_response(resources)
    
print(f"anymatix running on host {socket.gethostname()}")
    
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']