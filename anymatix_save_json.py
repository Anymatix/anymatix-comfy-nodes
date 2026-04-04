import json as json_module
import os
import folder_paths


def to_json_serializable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): to_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [to_json_serializable(item) for item in value]

    if hasattr(value, "tolist"):
        return to_json_serializable(value.tolist())

    if hasattr(value, "item"):
        try:
            return to_json_serializable(value.item())
        except Exception:
            pass

    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return to_json_serializable(value.cpu().numpy().tolist())

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

class AnymatixSaveJson:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json": ("*",), # Accept any type
                "output_path": (
                    "STRING",
                    {"default": "anymatix/results", "multiline": False},
                ),
                "filename_prefix": ("STRING", {"default": "data"}),
                "save_json": (["true", "false"],),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Anymatix"

    def save(
        self,
        json, # Must match input key
        output_path="anymatix/results",
        filename_prefix="data",
        save_json="true",
    ):
        output_path = os.path.join(folder_paths.get_output_directory(), output_path)
        os.makedirs(output_path, exist_ok=True)

        print(f"anymatix: saving generic data to {output_path}")

        # Although the node is called SaveJson, logically it saves the "json" input.
        # The input "json" can be anything (number, string, dict, list).
        # We will wrap it in a dictionary or save it directly.
        
        # Consistent with ImageSave, we might want to return UI info
        results = []
        
        # Determine filename
        filename = f"{filename_prefix}.json"
        file_path = os.path.join(output_path, filename)

        try:
            with open(file_path, "w") as f:
                data_to_save = to_json_serializable(json)
                json_module.dump(data_to_save, f, indent=2)
                print(f"Data saved to: {file_path}")
                
        except Exception as e:
            print(f"Unable to save data to {file_path}: {e}")

        # Return UI metadata if needed (not strictly required for reading, but good for UI feedback)
        return {"ui": {"json": [filename]}}
