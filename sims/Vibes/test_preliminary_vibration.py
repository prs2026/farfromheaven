"""Behavior checks for missing data, scaling, and generated regime outputs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preliminary_vibration import (
    _interpolate_curve, _motor_psd, _pressure_psd, parser, run,
)


EXAMPLE = Path(__file__).with_name("vibration_example_trajectory.csv")


class VibrationScreeningTests(unittest.TestCase):
    def test_reference_motor_psd_scales_by_amplitude_ratio_squared(self):
        reference = np.array([0.01, 0.02])
        scaled = _motor_psd(
            np.array([100.0, 200.0, 0.0]), np.array([1000.0] * 3),
            2.0, reference, 100.0, 1000.0, 2.0, 2,
        )
        np.testing.assert_allclose(scaled, [[0.01, 0.02], [0.04, 0.08], [0, 0]])
        missing = _motor_psd(np.array([100.0, 0.0]), np.array([1000.0, 0.0]),
                             2.0, None, None, None, None, 2)
        self.assertTrue(np.isnan(missing[0]).all())
        np.testing.assert_array_equal(missing[1], [0, 0])

    def test_pressure_rms_and_curve_band_validation(self):
        f = np.geomspace(1e-3, 1e8, 20000)
        q, psd = _pressure_psd(f, np.array([1.2]), np.array([200.0]),
                               1.0, 1.8e-5, 0.02)
        self.assertEqual(q[0], 24000.0)
        # Integrating the normalized one-sided spectrum recovers (Cq*q)^2.
        self.assertAlmostEqual(float(np.trapezoid(psd[0], f)) / (0.02 * q[0])**2,
                               1.0, places=3)
        with self.assertRaisesRegex(ValueError, "must cover"):
            _interpolate_curve((np.array([30., 100.]), np.array([1., 1.])),
                               np.array([20., 100.]), "test curve")

    def test_full_run_marks_unknown_and_writes_complete_result_with_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            common = [str(EXAMPLE), "--vehicle-length-m", "3.3",
                      "--station-from-nose-m", "1.2", "--motor-mass-kg", "8",
                      "--avionics-mass-kg", "1", "--structural-damping-ratio", "0.03",
                      "--frequency-points", "32", "--sample-step-s", "0.5"]
            pressure_output = run(parser().parse_args(
                common + ["--output-dir", str(base / "pressure_only")]))
            partial = pd.read_csv(pressure_output / "powered_ascent_envelope.csv")
            self.assertTrue(partial["combined_psd_g2_per_Hz"].isna().all())
            self.assertTrue(partial["aerodynamic_pressure_psd_Pa2_per_Hz"].gt(0).all())

            frequencies = [20, 100, 500, 2000]
            motor_file = base / "motor.csv"
            transfer_file = base / "transfer.csv"
            tones_file = base / "tones.csv"
            pd.DataFrame({"frequency_Hz": frequencies,
                          "acceleration_psd_g2_per_Hz": [0.01] * 4}).to_csv(
                              motor_file, index=False)
            pd.DataFrame({"frequency_Hz": frequencies,
                          "acceleration_per_pressure_g_per_Pa": [0.001] * 4}).to_csv(
                              transfer_file, index=False)
            pd.DataFrame({"frequency_Hz": [80],
                          "acceleration_rms_g": [0.02]}).to_csv(tones_file, index=False)
            complete_output = run(parser().parse_args(common + [
                "--output-dir", str(base / "complete"),
                "--motor-psd", str(motor_file),
                "--reference-thrust-n", "7000",
                "--reference-exhaust-velocity-mps", "2200",
                "--reference-motor-mass-kg", "8",
                "--transfer-function", str(transfer_file),
                "--motor-tones", str(tones_file),
            ]))
            full = pd.read_csv(complete_output / "powered_ascent_envelope.csv")
            self.assertTrue(full["combined_psd_g2_per_Hz"].notna().all())
            self.assertTrue(full["combined_psd_g2_per_Hz"].ge(
                full["motor_psd_g2_per_Hz"]).all())
            self.assertTrue((complete_output / "mach_6_5_segment_psd.png").exists())
            self.assertTrue((complete_output / "motor_tones_separate.csv").exists())
            summary = (complete_output / "summary.md").read_text(encoding="utf-8")
            self.assertIn("maximum dynamic pressure", summary.lower())
            self.assertIn("combined", summary.lower())
            self.assertIn("PRELIMINARY SCREENING", summary)


if __name__ == "__main__":
    unittest.main()
