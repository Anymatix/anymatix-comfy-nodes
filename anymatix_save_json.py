import json as json_module
import os
import folder_paths

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
                # We wrap primitive values in a structure if needed, or just dump them.
                # Anymatix app expects the content of the file to be the value? 
                # Or does it expect a specific JSON structure?
                # Looking at AnymatixImageSave, it dumps {"count": ...}.
                # But here we want to observe the VALUE.
                # So we should probably dump the value directly.
                
                # Handling non-serializable objects (like tensors)
                # If it's a tensor, convert to list
                if hasattr(json, "tolist"):
                    data_to_save = json.tolist()
                elif hasattr(json, "cpu"):
                     data_to_save = json.cpu().numpy().tolist()
                else:
                    data_to_save = json
                
                json_module.dump(data_to_save, f, indent=2)
                print(f"Data saved to: {file_path}")
                
        except Exception as e:
            print(f"Unable to save data to {file_path}: {e}")

        # Return UI metadata if needed (not strictly required for reading, but good for UI feedback)
        return {"ui": {"json": [filename]}}
