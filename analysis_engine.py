

from __future__ import annotations

import cv2
import numpy as np


def analyse_changes(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    threshold: int = 35,
) -> dict:
    
    a, b = np.asarray(before_rgb), np.asarray(after_rgb)
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("Images must be aligned RGB arrays of the same dimensions.")
    if a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError("Image arrays must use uint8 values between 0 and 255.")
    if not (0 <= threshold <= 255):
        raise ValueError("Threshold must be between 0 and 255.")

    a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    a_blur = cv2.GaussianBlur(a_gray, (5, 5), 0)
    b_blur = cv2.GaussianBlur(b_gray, (5, 5), 0)
    difference_map = cv2.absdiff(a_blur, b_blur)
    change_mask = np.where(difference_map >= threshold, 255, 0).astype(np.uint8)

    
    kernel = np.ones((3, 3), dtype=np.uint8)
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)

    return {
        "change_mask": change_mask,
        "difference_map": difference_map,
        "method": "RGB difference baseline (not machine learning)",
        "is_ml": False,
        "explanation": (
            "Aligned RGB photographs are converted to grayscale, smoothed, "
            "subtracted, thresholded, and cleaned. This is only a visual "
            "change-detection baseline, not a glacier/lake classifier."
        ),
    }
