import unittest
from pathlib import Path
from spectral_engine import REQUIRED, load_sentinel_data

class SplitArchiveTest(unittest.TestCase):
    def test_all_layers_load(self):
        data = load_sentinel_data(Path(__file__).resolve().parents[1] / 'data' / 'glacierguard_data.npz')
        self.assertEqual(set(data), set(REQUIRED))
        self.assertEqual(data['valid_mask'].shape, (1202, 1829))

if __name__ == '__main__':
    unittest.main()
