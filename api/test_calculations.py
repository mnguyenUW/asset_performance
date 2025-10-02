import unittest
from calculations import calculate_clogging
import pandas as pd

ASSETS = [
    {"asset": "2285", "max_hp": 2250, "max_restriction": 35},
    {"asset": "22270", "max_hp": 2250, "max_restriction": 24.9},
    {"asset": "22665", "max_hp": 2025, "max_restriction": 24.9},
]

class TestCalculateClogging(unittest.TestCase):
    def test_multi_asset_clogging(self):
        for asset_info in ASSETS:
            asset = asset_info["asset"]
            max_hp = asset_info["max_hp"]
            max_restriction = asset_info["max_restriction"]

            # Test with clean filter (restriction at clean baseline)
            hp = max_hp * 0.5
            restriction = max_restriction * 0.5
            result = calculate_clogging(asset, hp, restriction)
            self.assertIsInstance(result, dict)
            self.assertIn("delta", result)
            self.assertIn("HP_max_current", result)
            self.assertIn("percent_clogged", result)
            self.assertGreaterEqual(result["HP_max_current"], 0)
            self.assertLessEqual(result["HP_max_current"], max_hp)
            self.assertGreaterEqual(result["percent_clogged"], 0)
            self.assertLessEqual(result["percent_clogged"], 100)

            # Test with fully clogged (restriction at max)
            restriction = max_restriction
            result = calculate_clogging(asset, hp, restriction)
            self.assertIsInstance(result, dict)
            self.assertLessEqual(result["HP_max_current"], max_hp)
            self.assertLessEqual(result["percent_clogged"], 100)

if __name__ == "__main__":
    unittest.main()