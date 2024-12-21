import json
from typing import Dict
import app.logger
import folder_paths
from aiohttp import web
import logging
import os
import socket
from server import PromptServer
import app

from .anymatix_checkpoint_fetcher import AnymatixCheckpointFetcher, AnymatixCheckpointLoader

NODE_CLASS_MAPPINGS = {
    "AnymatixCheckpointFetcher": AnymatixCheckpointFetcher,
    "AnymatixCheckpointLoader": AnymatixCheckpointLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnymatixCheckpointFetcher": "Anymatix Checkpoint Fetcher",
    "AnymatixCheckpointLoader": "Anymatix Checkpoint Loader"
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"anymatix: running on host {socket.gethostname()}")
allowed_dirs = ["output", "input", "models"]

routes = PromptServer.instance.routes

@routes.get('/anymatix/log')
async def get_log(request):
    return web.json_response(list(app.logger.get_logs()))

from .expunge import * 
    
outdir = f"{folder_paths.output_directory}/anymatix/results"
@routes.post('/anymatix/expunge')
async def serve_expunge(request):    
    print("anymatix: expunging cache")
    data = await request.json()
    keep : list[str] = data['keep']
    await expunge(keep,outdir)    
    return web.Response(status=200)

@routes.get('/anymatix/cache_size')
async def serve_cache_size(request):    
    result = await count_outputs(outdir)    
    return web.json_response(result)


@routes.get("/anymatix/{basedir}/{filename:.+}")
async def serve_file(request):
    response = web.Response(text="File not found", status=404)

    basedir = request.match_info['basedir']

    if basedir in allowed_dirs:
        # TODO: the check on getcwd is plain wrong (if the cwd is not what I expected). Determine from the current script?
        file_path = os.path.abspath(
            f"{basedir}/{request.match_info['filename']}")
        if file_path.startswith(f"{os.path.abspath(os.getcwd())}/{basedir}") and os.path.isfile(file_path) and os.access(file_path, os.R_OK):
            response = web.FileResponse(file_path)
        else:
            print("****", f"{os.path.abspath(os.getcwd())}/{basedir}",
                  os.path.isfile(file_path), os.access(file_path, os.R_OK))

    return response

# resource_extensions = [".ckpt", ".safetensors"]


@routes.get("/anymatix/resources")
async def serve_resources(_request):
    json_files = (os.path.join(dirpath, filename)  
                  for dirpath, _, filenames in os.walk(folder_paths.models_dir)
                  for filename in filenames
                  if filename.endswith(".json"))

    def get_json_data(path: str) -> dict:
        with open(path, 'r') as f:
            return json.load(f)

    def get_type(path: str) -> str:
        return path.removeprefix(folder_paths.models_dir).split(os.sep)[1]

    def get_contents(path: str):
        data = get_json_data(path)
        type = get_type(path)
        file_path = path.removeprefix(os.path.join(folder_paths.models_dir,type))
        return (data,type,file_path)

    contents = map(get_contents, json_files)

    result : Dict[str,list]= {}

    for (data,type,file_path) in contents:        
        if type in result:
            result[type].append(data)
        else:
            result[type] = [data]
                
    return web.json_response(result)

