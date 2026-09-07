#based on https://github.com/Glidias/mask2sam
import torch
import numpy as np
import json
from skimage import measure
from shapely.geometry import Polygon
from .poly_decomp import polygonQuickDecomp as bayazit_decomp


class AnymatixMaskToSAMcoord:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "individual_objects": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pos_str", "neg_str")
    FUNCTION = "mask_to_points"
    CATEGORY = "Anymatix"

    def get_reference_point_from_contour(self, contour):
        if len(contour) < 3:
            cx = np.mean(contour[:, 0])
            cy = np.mean(contour[:, 1])
            return float(cx), float(cy)

        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])

        poly_coords = contour[:-1].tolist()

        try:
            convex_pieces = bayazit_decomp(poly_coords)
            max_area = -1
            best_centroid = None

            for piece in convex_pieces:
                if len(piece) < 3:
                    continue
                poly = Polygon(piece)
                if poly.is_valid and poly.area > max_area:
                    max_area = poly.area
                    c = poly.centroid
                    best_centroid = (float(c.x), float(c.y))

            if best_centroid:
                return best_centroid

        except Exception:
            pass

        cx = np.mean(contour[:, 0])
        cy = np.mean(contour[:, 1])
        return float(cx), float(cy)

    def mask_to_points(self, mask, individual_objects=True):
        B, H, W = mask.shape
        pos_points = []
        neg_points = []

        for b in range(B):
            mask_np = mask[b].cpu().numpy()

            pos_mask = (mask_np == 0)
            neg_mask = (mask_np > 0) & (mask_np < 255)

            # POSITIVE POINTS
            labels = measure.label(pos_mask, connectivity=2)
            for region in measure.regionprops(labels):
                region_mask = (labels == region.label)
                contours = measure.find_contours(region_mask, 0.5)
                if contours:
                    contour = np.fliplr(max(contours, key=len))
                    pos_points.append(
                        self.get_reference_point_from_contour(contour)
                    )

            # NEGATIVE POINTS
            neg_labels = measure.label(neg_mask, connectivity=2)
            for region in measure.regionprops(neg_labels):
                cy, cx = region.centroid
                neg_points.append((float(cx), float(cy)))

        if individual_objects and neg_points:
            paired_neg = []
            for p in pos_points:
                dists = [(p[0]-n[0])**2 + (p[1]-n[1])**2 for n in neg_points]
                paired_neg.append(neg_points[int(np.argmin(dists))])
            neg_points = paired_neg

        pos_str = json.dumps([{"x": x, "y": y} for x, y in pos_points])
        neg_str = json.dumps([{"x": x, "y": y} for x, y in neg_points])

        return (pos_str, neg_str)
