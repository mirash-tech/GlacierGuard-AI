

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np



REFERENCE_PIXEL_AREA_M2 = 99.99184702916689

REQUIRED = (
    'before_B03', 'before_B04', 'before_B08', 'before_B11',
    'after_B03', 'after_B04', 'after_B08', 'after_B11',
    'ndwi_before', 'ndwi_after', 'ndsi_before', 'ndsi_after',
    'ndwi_change', 'ndsi_change', 'valid_mask',
)


def load_sentinel_data(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if path.is_file():
        archives = [path]
    else:
        archives = [path.with_name(f'glacierguard_part_{i}.npz') for i in range(1, 4)]
        if not all(part.is_file() for part in archives):
            raise FileNotFoundError(f'Satellite data not found: {path}')
    data = {}
    for archive_path in archives:
        with np.load(archive_path, allow_pickle=False) as archive:
            data.update({key: archive[key] for key in archive.files if key in REQUIRED})
    missing = set(REQUIRED) - set(data)
    if missing:
        raise ValueError(f"Satellite NPZ is missing: {', '.join(sorted(missing))}")
    shape = data['valid_mask'].shape
    if len(shape) != 2 or not all(a.shape == shape for a in data.values()):
        raise ValueError('Every satellite layer must share the same 2D aligned grid.')
    data['valid_mask'] = data['valid_mask'].astype(bool)
    if not data['valid_mask'].any():
        raise ValueError('No overlapping valid pixels were found.')
    for key, arr in data.items():
        if key != 'valid_mask' and not np.isfinite(arr[data['valid_mask']]).all():
            raise ValueError(f'Non-finite values inside the valid overlap: {key}')
    return data


def false_color(data: dict, prefix: str) -> np.ndarray:
    
    mask = data['valid_mask']
    rgb = np.empty((*mask.shape, 3), dtype=np.uint8)
    for i, band in enumerate(('B04', 'B03', 'B08')):
        values = data[f'{prefix}_{band}']
        lo, hi = np.percentile(values[mask], (2, 98))
        channel = np.zeros(mask.shape, dtype=np.float32)
        if hi > lo:
            channel[mask] = np.clip((values[mask] - lo) / (hi - lo), 0, 1)
        rgb[:, :, i] = np.uint8(np.rint(channel * 255))
    rgb[~mask] = (235, 235, 235)
    return rgb


def index_image(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    
    scaled = np.uint8(np.rint(np.clip((np.nan_to_num(values, nan=-1) + 1) / 2, 0, 1) * 255))
    image = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    image[~valid_mask] = (235, 235, 235)
    return image


def categorical_change(before: np.ndarray, after: np.ndarray, valid: np.ndarray) -> np.ndarray:
    
    image = np.empty((*valid.shape, 3), dtype=np.uint8)
    image[:] = (140, 145, 150)
    image[~valid] = (235, 235, 235)
    image[after & ~before & valid] = (35, 160, 90)
    image[before & ~after & valid] = (234, 115, 45)
    return image


def analyse_spectral_data(data: dict, water_threshold: float = 0.3,
                          snow_threshold: float = 0.4,
                          exclude_snow: bool = False) -> dict:
    
    valid = data['valid_mask']
    snow_before = (data['ndsi_before'] > snow_threshold) & valid
    snow_after = (data['ndsi_after'] > snow_threshold) & valid
    
    
    water_before = (data['ndwi_before'] > water_threshold) & valid
    water_after = (data['ndwi_after'] > water_threshold) & valid
    if exclude_snow:
        water_before &= ~snow_before
        water_after &= ~snow_after
    water_transition = (water_before ^ water_after) & valid
    snow_transition = (snow_before ^ snow_after) & valid
    change_mask = np.uint8(water_transition | snow_transition) * 255
    delta = np.maximum(np.abs(data['ndwi_change']), np.abs(data['ndsi_change']))
    difference_map = np.uint8(np.rint(np.clip(np.nan_to_num(delta, nan=0) * 255, 0, 255)))
    difference_map[~valid] = 0
    counts = {
        'valid_overlap_pixels': int(valid.sum()),
        'water_candidate_before_pixels': int(water_before.sum()),
        'water_candidate_after_pixels': int(water_after.sum()),
        'water_candidate_gain_pixels': int((water_after & ~water_before).sum()),
        'water_candidate_loss_pixels': int((water_before & ~water_after).sum()),
        'water_and_snow_overlap_before_pixels': int(((data['ndwi_before'] > water_threshold) & snow_before).sum()),
        'water_and_snow_overlap_after_pixels': int(((data['ndwi_after'] > water_threshold) & snow_after).sum()),
        'snow_candidate_before_pixels': int(snow_before.sum()),
        'snow_candidate_after_pixels': int(snow_after.sum()),
        'snow_candidate_gain_pixels': int((snow_after & ~snow_before).sum()),
        'snow_candidate_loss_pixels': int((snow_before & ~snow_after).sum()),
    }
    return {
        'water_rule': 'NDWI-only' if not exclude_snow else 'NDWI with NDSI snow exclusion',
        'pixel_area_m2_from_supplied_grid_metadata': REFERENCE_PIXEL_AREA_M2,
        'change_mask': change_mask,
        'difference_map': difference_map,
        'method': 'Sentinel-2 NDWI/NDSI thresholds — ' + ('snow-excluded water' if exclude_snow else 'NDWI-only water') + ' (not ML)',
        'is_ml': False,
        'explanation': (
            'Aligned Sentinel-2 B03/B04/B08/B11 arrays; NDWI=(B03-B08)/(B03+B08), '
            'NDSI=(B03-B11)/(B03+B11). Candidate water NDWI>0.3' +
            (' excluding NDSI>0.4 snow candidates' if exclude_snow else ' without snow exclusion (matches supplied CSV)') +
            '; candidate snow NDSI>0.4. Changes compare '
            '12 Aug with 13 Oct 2026 on overlapping valid pixels. No cloud/shadow '
            'mask, ground truth, independently validated lake outlines, trained '
            'model, or flood prediction. Zero candidate water after does not '
            'establish lake disappearance.'
        ),
        'valid_mask': valid,
        'water_before_mask': water_before,
        'water_after_mask': water_after,
        'snow_before_mask': snow_before,
        'snow_after_mask': snow_after,
        'counts': counts,
        'water_change_image': categorical_change(water_before, water_after, valid),
        'snow_change_image': categorical_change(snow_before, snow_after, valid),
    }


def overlap_bounds(valid_mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(valid_mask)
    if len(rows) == 0:
        raise ValueError("No valid overlapping pixels found.")
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def crop_to_bounds(image: np.ndarray, bounds: tuple[int, int, int, int] | None) -> np.ndarray:
    if bounds is None:
        return image
    top, bottom, left, right = bounds
    return image[top:bottom, left:right]
