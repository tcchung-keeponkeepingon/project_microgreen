"""SEM pore-overlay generation (hybrid Otsu + absolute-darkness threshold).

Distilled from quantify_porosity{,_pva,_figs1}.py. A pore is a dark region
(below both the Otsu threshold and an absolute darkness floor). The PVA series
additionally applies a shape filter to reject scratch-like / ragged regions;
the figs1 series crops the SEM info banner first.
"""
import numpy as np
from PIL import Image
from skimage import color, filters, measure, morphology

OVERLAY_COLOR = np.array([0.275, 0.71, 0.996])  # sky blue
OVERLAY_ALPHA = 0.35


def load_gray_uint8(path):
    img = np.array(Image.open(path).convert("RGB"))
    return (color.rgb2gray(img) * 255).astype(np.uint8)


def _threshold_dark(gray, dark_floor, min_pore_area, min_hole_area):
    denoised = filters.median(gray, morphology.disk(2))
    otsu = filters.threshold_otsu(denoised)
    mask = (denoised < otsu) & (denoised < dark_floor)
    mask = morphology.remove_small_objects(mask, max_size=min_pore_area)
    mask = morphology.remove_small_holes(mask, max_size=min_hole_area)
    return mask


def _shape_filter(mask, max_aspect, min_solidity):
    lbl = measure.label(mask)
    keep = np.zeros_like(mask)
    for p in measure.regionprops(lbl):
        minor = max(p.axis_minor_length, 1e-6)
        if p.axis_major_length / minor <= max_aspect and p.solidity >= min_solidity:
            keep[lbl == p.label] = True
    return keep


def pore_overlay(path, dark_floor=60, min_pore_area=30, min_hole_area=30,
                 crop_top=None, shape_filter=None):
    """Return (overlay_rgb_float, porosity_pct).

    shape_filter=(max_aspect, min_solidity) enables the PVA-series filter; None disables it.
    crop_top=N keeps rows [:N] (drops the SEM info banner).
    """
    gray = load_gray_uint8(path)
    if crop_top is not None:
        gray = gray[:crop_top]
    mask = _threshold_dark(gray, dark_floor, min_pore_area, min_hole_area)
    if shape_filter is not None:
        mask = _shape_filter(mask, *shape_filter)
    rgb = np.stack([gray] * 3, axis=-1).astype(float) / 255.0
    overlay = rgb.copy()
    overlay[mask] = (1 - OVERLAY_ALPHA) * rgb[mask] + OVERLAY_ALPHA * OVERLAY_COLOR
    return overlay, 100.0 * mask.sum() / mask.size
