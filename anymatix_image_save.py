import re
import comfy
import comfy.sd
import comfy.utils
import folder_paths
import os
import numpy as np
import json
from PIL import Image


class Anymatix_Image_Save:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_path": (
                    "STRING",
                    {"default": "[time(%Y-%m-%d)]", "multiline": False},
                ),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "filename_delimiter": ("STRING", {"default": "_"}),
                "filename_number_padding": (
                    "INT",
                    {"default": 4, "min": 1, "max": 9, "step": 1},
                ),
                "filename_number_start": (["false", "true"],),
                "extension": (["png", "jpg", "jpeg", "gif", "tiff", "webp", "bmp"],),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                "lossless_webp": (["false", "true"],),
                "overwrite_mode": (["false", "true", "prefix_as_filename"],),
                "show_previews": (["true", "false"],),
                "save_json": (["true", "false"]),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "was_save_images"

    OUTPUT_NODE = True

    CATEGORY = "Anymatix"

    def was_save_images(
        self,
        images,
        output_path="",
        filename_prefix="ComfyUI",
        filename_delimiter="_",
        extension="png",
        quality=100,
        lossless_webp="false",
        overwrite_mode="false",
        filename_number_padding=4,
        filename_number_start="false",
        show_previews="true",
        save_json="false",
    ):

        delimiter = filename_delimiter
        number_padding = filename_number_padding
        lossless_webp = lossless_webp == "true"

        output_path = os.path.join(folder_paths.get_output_directory(), output_path)
        os.makedirs(output_path, exist_ok=True)

        print(f"anymatix: saving images to {output_path}")

        if overwrite_mode == "false":
            # Find existing counter values
            if filename_number_start == "true":
                pattern = f"(\\d+){re.escape(delimiter)}{re.escape(filename_prefix)}"
            else:
                pattern = f"{re.escape(filename_prefix)}{re.escape(delimiter)}(\\d+)"
            existing_counters = [
                int(re.search(pattern, filename).group(1))
                for filename in os.listdir(output_path)
                if re.match(pattern, os.path.basename(filename))
            ]
            existing_counters.sort(reverse=True)
        else:
            existing_counters = []

        # Set initial counter value
        if existing_counters:
            counter = existing_counters[0] + 1
        else:
            counter = 1

        # Set initial counter value
        if existing_counters:
            counter = existing_counters[0] + 1
        else:
            counter = 1

        ALLOWED_EXT = [".png", ".jpg", ".jpeg", ".gif", ".tiff", ".webp", ".bmp"]
        # Set Extension
        file_extension = "." + extension
        if file_extension not in ALLOWED_EXT:
            print(
                f"The extension `{extension}` is not valid. The valid formats are: {', '.join(sorted(ALLOWED_EXT))}"
            )
            file_extension = "png"

        saved_files = list()
        results = list()
        for image in images:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Delegate the filename stuffs
            if overwrite_mode == "prefix_as_filename":
                file = f"{filename_prefix}{file_extension}"
            else:
                if filename_number_start == "true":
                    file = f"{counter:0{number_padding}}{delimiter}{filename_prefix}{file_extension}"
                else:
                    file = f"{filename_prefix}{delimiter}{counter:0{number_padding}}{file_extension}"
                if os.path.exists(os.path.join(output_path, file)):
                    counter += 1

            # Save the images
            try:
                output_file = os.path.abspath(os.path.join(output_path, file))

                if save_json:
                    saved_files.append(file)

                if extension in ["jpg", "jpeg"]:
                    img.save(output_file, quality=quality, optimize=True)
                elif extension == "webp":
                    img.save(output_file, quality=quality, lossless=lossless_webp)
                elif extension == "png":
                    img.save(output_file, optimize=True)
                elif extension == "bmp":
                    img.save(output_file)
                elif extension == "tiff":
                    img.save(output_file, quality=quality, optimize=True)
                else:
                    img.save(output_file, optimize=True)

                print(f"Image file saved to: {output_file}")

            except OSError as e:
                print(f"Unable to save file to: {output_file}")
                print(e)
            except Exception as e:
                print("Unable to save file due to the to the following error:")
                print(e)

            if overwrite_mode != "prefix_as_filename":
                counter += 1

        if save_json:
            json_output_file = os.path.abspath(
                os.path.join(output_path, f"{filename_prefix}.json")
            )
            json_obj = {"count": len(saved_files)}
            with open(json_output_file, "w") as outfile:
                json.dump(json_obj, outfile)

        filtered_paths = []
        if filtered_paths:
            for image_path in filtered_paths:
                subfolder = self.get_subfolder_path(image_path, self.output_dir)
                image_data = {
                    "filename": os.path.basename(image_path),
                    "subfolder": subfolder,
                    "type": self.type,
                }
                results.append(image_data)

        if show_previews == "true":
            return {"ui": {"images": results}}
        else:
            return {"ui": {"images": []}}

    def get_subfolder_path(self, image_path, output_path):
        output_parts = output_path.strip(os.sep).split(os.sep)
        image_parts = image_path.strip(os.sep).split(os.sep)
        common_parts = os.path.commonprefix([output_parts, image_parts])
        subfolder_parts = image_parts[len(common_parts) :]
        subfolder_path = os.sep.join(subfolder_parts[:-1])
        return subfolder_path
