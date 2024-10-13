import json
import folder_paths
from aiohttp import web
import os
import socket
from server import PromptServer

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

print(f"anymatix running on host {socket.gethostname()}")
allowed_dirs = ["output", "input", "models"]

routes = PromptServer.instance.routes


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
    json_files = (os.path.join(dirpath, filename)  # .removeprefix(folder_paths.models_dir)
                  for dirpath, _, filenames in os.walk(folder_paths.models_dir)
                  for filename in filenames
                  if filename.endswith(".json"))

    def get_json_data(path: str) -> dict:
        with open(path, 'r') as f:
            return json.load(f)

    def get_type(path: str) -> str:
        return path.removeprefix(folder_paths.models_dir).split(os.sep)[1]

    contents = map(lambda x: {"type": get_type(
        x), "data": get_json_data(x)}, json_files)

    result = list(contents)
    return web.json_response(result)

#     def get_json_data(path: str) -> dict:
#         with open(find(path), 'r') as f:
#             return json.load(f)

#     def get_filename_from_json(path: str):
#         data = get_json_data(path)
#         if "file_name" in data:
#             return data["file_name"]
#         return None


#     for (_,_,file) in os.walk(folder_paths.models_dir):
#         print("FULE",file)

#     result = []
#     return web.json_response(result)

#     resources = {}

#     def find(path: str):
#         return os.path.join(md, path)

#     def get_json_data(path: str) -> dict:
#         with open(find(path), 'r') as f:
#             return json.load(f)

#     def get_filename_from_json(path: str):
#         data = get_json_data(path)
#         if "file_name" in data:
#             return data["file_name"]
#         return None

#     def get_files(path: str):
#         return (f for f in os.listdir(find(path)) if any(f.endswith(ext) for ext in resource_extensions))

#     def get_json_files(path):
#         return [os.path.join(path, f) for f in os.listdir(find(path)) if f.endswith(".json")]

#     def get_resources(path):
#         print("get_resources", path)
#         json_files = get_json_files(path)
#         print("json_files", json_files)

#         references = {filename: json_file for json_file in json_files if (
#             filename := get_filename_from_json(json_file)) if filename is not None}

#         files = get_files(path)

#         print("files",path,list(files))
#         result = list(map(lambda file: get_json_data(references[file]) if file in references else file, files))
#         print("result",result) if path == "checkpoints" else None
#         return result

#     for dir in os.listdir(md):
#         if os.path.isdir(find(dir)):
#             resources[dir] = get_resources(os.path.join(md,dir))


#     return web.json_response(resources)
