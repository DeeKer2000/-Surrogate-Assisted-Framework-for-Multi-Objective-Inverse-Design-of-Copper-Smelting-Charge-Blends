import importlib.util
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "optimization" / "nsga2_optimization.py"
SPEC = importlib.util.spec_from_file_location("nsga2_optimization", MODULE_PATH)
optimization = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(optimization)


class RatiosToFeaturesTests(unittest.TestCase):
    def test_mass_features_follow_training_input_order(self):
        ratios = np.arange(1, optimization.N_ORES + 1, dtype=float)
        ratios /= ratios.sum()
        sio2_flux_ratio = 0.075

        actual = optimization.ratios_to_features(ratios, sio2_flux_ratio)

        components = [
            col.removesuffix("(g)") for col in optimization.INPUT_COLS[2:-1]
        ]
        expected_pct = ratios @ optimization.ORE_COMPOSITION[components].values
        expected_pct[components.index("SiO2")] += sio2_flux_ratio * 100
        expected_mass = (
            expected_pct
            / (1.0 + sio2_flux_ratio)
            * optimization.TOTAL_CHARGE
            / 100.0
        )

        np.testing.assert_allclose(
            actual[2:-1],
            expected_mass,
            rtol=1e-6,
            err_msg="ratios_to_features must match the model training column order",
        )

    def test_comparison_uses_one_real_representative_solution(self):
        representatives = optimization.pd.DataFrame(
            {
                "slag_Cu_pct": [0.20, 0.25, 0.30, 0.35, 0.40],
                "matte_Cu_pct": [50.0, 55.0, 57.0, 59.0, 60.0],
                # The median is 1.17, while solution C is 1.15.
                "slag_FeO_SiO2_ratio": [1.16, 1.17, 1.15, 1.18, 1.19],
            }
        )

        _, optimized = optimization.generate_comparison(representatives)

        self.assertEqual(optimized["slag_Cu_pct"], 0.30)
        self.assertEqual(optimized["matte_Cu_pct"], 57.0)
        self.assertEqual(optimized["slag_FeO_SiO2_ratio"], 1.15)


if __name__ == "__main__":
    unittest.main()
