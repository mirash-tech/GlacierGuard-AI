
import unittest
import numpy as np
from analysis_engine import analyse_changes


class ChangeDetectorTests(unittest.TestCase):
    def test_identical_images_have_no_changes(self):
        before = np.full((64, 64, 3), 140, dtype=np.uint8)
        result = analyse_changes(before, before.copy(), threshold=35)
        self.assertEqual(np.count_nonzero(result['change_mask']), 0)
        self.assertFalse(result['is_ml'])

    def test_large_change_detected(self):
        before = np.zeros((64, 64, 3), dtype=np.uint8)
        after = before.copy()
        after[20:40, 20:40] = 255
        result = analyse_changes(before, after, threshold=35)
        self.assertGreater(np.count_nonzero(result['change_mask']), 250)
        self.assertEqual(result['change_mask'].shape, (64, 64))
        self.assertEqual(result['difference_map'].shape, (64, 64))

    def test_shape_mismatch_rejected(self):
        before = np.zeros((24, 24, 3), dtype=np.uint8)
        after = np.zeros((12, 12, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            analyse_changes(before, after)


if __name__ == '__main__':
    unittest.main()
