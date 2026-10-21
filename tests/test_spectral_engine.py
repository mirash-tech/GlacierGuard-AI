
import unittest
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral_engine import load_sentinel_data, false_color, analyse_spectral_data, index_image, overlap_bounds, crop_to_bounds


class TestSpectralEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_sentinel_data(Path(__file__).resolve().parents[1] / 'data' / 'glacierguard_data.npz')
        cls.result = analyse_spectral_data(cls.data)

    def test_shape_and_valid_overlap(self):
        self.assertEqual(self.data['valid_mask'].shape, (1202, 1829))
        self.assertEqual(self.result['counts']['valid_overlap_pixels'], 558610)

    def test_candidates_match_notebook(self):
        counts = self.result['counts']
        self.assertEqual(counts['water_candidate_before_pixels'], 647)
        self.assertEqual(counts['water_candidate_after_pixels'], 234)
        self.assertEqual(counts['snow_candidate_before_pixels'], 219035)
        self.assertEqual(counts['snow_candidate_after_pixels'], 499070)

    def test_optional_snow_exclusion(self):
        result = analyse_spectral_data(self.data, exclude_snow=True)
        self.assertEqual(result['counts']['water_candidate_before_pixels'], 267)
        self.assertEqual(result['counts']['water_candidate_after_pixels'], 0)

    def test_export_matches_reference_csv(self):
        from spectral_engine import REFERENCE_PIXEL_AREA_M2
        counts = self.result['counts']
        area = REFERENCE_PIXEL_AREA_M2 / 1_000_000
        self.assertAlmostEqual(counts['water_candidate_before_pixels'] * area, 0.06469472502787098)
        self.assertAlmostEqual(counts['water_candidate_after_pixels'] * area, 0.023398092204825055)
        self.assertAlmostEqual(counts['water_candidate_gain_pixels'] * area, 0.0024997961757291724)
        self.assertAlmostEqual(counts['water_candidate_loss_pixels'] * area, 0.0437964289987751)
        self.assertAlmostEqual(counts['snow_candidate_before_pixels'] * area, 21.90171421403357)
        self.assertAlmostEqual(counts['snow_candidate_after_pixels'] * area, 49.902931096846324)

    def test_not_ml_and_mask_is_valid(self):
        self.assertIs(self.result['is_ml'], False)
        self.assertFalse(np.any(self.result['change_mask'][~self.data['valid_mask']]))
        self.assertEqual(self.result['change_mask'].dtype, np.uint8)
        self.assertEqual(self.result['difference_map'].dtype, np.uint8)

    def test_display_crops_to_valid_footprint(self):
        y0, y1, x0, x1 = overlap_bounds(self.data['valid_mask'])
        self.assertEqual((y1-y0, x1-x0), (621, 921))
        display = crop_to_bounds(false_color(self.data, 'before'), (y0, y1, x0, x1))
        self.assertEqual(display.shape, (621, 921, 3))

    def test_display_arrays(self):
        rgb = false_color(self.data, 'before')
        idx = index_image(self.data['ndwi_before'], self.data['valid_mask'])
        self.assertEqual(rgb.shape, (1202, 1829, 3))
        self.assertEqual(idx.shape, rgb.shape)
        self.assertEqual(rgb.dtype, np.uint8)


if __name__ == '__main__':
    unittest.main()
