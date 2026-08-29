import re
import comfy
import comfy.sd
import comfy.utils
import folder_paths
import os
import numpy as np
import json

from anymatix_output_formats import IMAGE_EXTENSIONS, write_image


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
                "extension": (IMAGE_EXTENSIONS,),
                "overwrite_mode": (["false", "true", "prefix_as_filename"],),
                "show_previews": (["true", "false"],),
                "save_json": (["true", "false"]),
            },
            # OPTIONAL, and every one of them for a reason.
            #
            # `bit_depth` is new, and a required input added here would fail
            # validation for every card that predates the rungs. `quality` and
            # `lossless_webp` move here because they are parameters of SOME
            # formats: a rung that writes OpenEXR has no quality to state, and
            # made to state one anyway it would state a meaningless number that
            # somebody would later read as meaning something.
            "optional": {
                # 8 unless a rung asks for more. A separate input rather than a
                # `png16` extension, so the extension stays the file name's —
                # which is what the renderer derives from the type.
                "bit_depth": ("INT", {"default": 8, "min": 8, "max": 16, "step": 8}),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                "lossless_webp": (["false", "true"],),
            },
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
        bit_depth=8,
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

        # An extension this node does not know is a bug upstream. It used to
        # fall back to PNG — and silently, which means an address that says
        # `webp` could hold a PNG, exactly the disagreement the format rungs
        # exist to close.
        if extension not in IMAGE_EXTENSIONS:
            raise RuntimeError(
                f"The extension `{extension}` is not one this node writes. "
                f"Known: {', '.join(IMAGE_EXTENSIONS)}"
            )
        file_extension = "." + extension

        results = list()
        
        # Total image count is deterministic - write JSON FIRST so viewer can display immediately
        total_images = len(images)
        
        if save_json == "true":
            # Write JSON with final count BEFORE saving images
            # This allows the viewer to know the count immediately and show images as they arrive
            json_output_file = os.path.abspath(
                os.path.join(output_path, f"{filename_prefix}.json")
            )
            json_obj = {"count": total_images}
            with open(json_output_file, "w") as outfile:
                json.dump(json_obj, outfile)
            print(f"anymatix: JSON written first with count={total_images}")
        
        # Add progress bar for batch image saving
        pbar = comfy.utils.ProgressBar(total_images) if total_images > 1 else None
        # For batches, disable PNG optimization (compress_level=1 is ~32x faster than optimize=True)
        # optimize=True uses compress_level=9 which is very slow for many images
        fast_batch = total_images > 4
        print(f"anymatix: saving {total_images} images to {output_path} (fast_batch={fast_batch})")
        
        for idx, image in enumerate(images):
            frame = image.cpu().numpy()

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

            output_file = os.path.abspath(os.path.join(output_path, file))
            write_image(
                frame,
                output_file,
                extension=extension,
                quality=quality,
                lossless_webp=lossless_webp,
                bit_depth=bit_depth,
                fast=fast_batch,
            )
            if not fast_batch:
                print(f"Image file saved to: {output_file}")

            if overwrite_mode != "prefix_as_filename":
                counter += 1
            
            # Update progress bar after each image save
            if pbar is not None:
                pbar.update(1)

        # JSON was already written at the start with the correct count

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
