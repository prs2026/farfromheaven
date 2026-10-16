"""Small deterministic checks for the Blue Raven trajectory converter."""

import unittest

import numpy as np
import pandas as pd

from Vibes.blue_raven_to_trajectory import (
    PROFILE_COLUMNS,
    _unwrap_signed_16_bit_positions,
    build_trajectory,
)


class BlueRavenTrajectoryTests(unittest.TestCase):
    def test_wrap_undoes_signed_16_bit_discontinuity(self):
        actual, wraps = _unwrap_signed_16_bit_positions(
            np.array([-32744.0, 32765.0, 32740.0])
        )
        np.testing.assert_array_equal(actual, [-32744.0, -32771.0, -32796.0])
        self.assertEqual(wraps, 1)

    def test_profile_units_order_and_quaternion_interpolation(self):
        hr = pd.DataFrame({
            "Flight_Time_(s)": [0.0, 1.0],
            "Quat_1": [0.0, 0.0],
            "Quat_2": [0.0, 0.0],
            "Quat_3": [0.0, 1.0],
            "Quat_4": [1.0, 0.0],
            "Gyro_X": [0.0, 0.0],
            "Gyro_Y": [0.0, 0.0],
            "Gyro_Z": [0.0, 0.0],
        })
        lr = pd.DataFrame({
            "Flight_Time_(s)": [0.0, 0.5, 1.0],
            "Inertial_DR_Position": [0, 10, 20],
            "Inertial_CR_position": [0, -10, -20],
            "Inertial_Altitude": [0, 100, 200],
            "Velocity_DR": [0, 10, 20],
            "Velocity_CR": [0, -10, -20],
            "Velocity_Up": [0, 100, 200],
        })
        profile, quality = build_trajectory(hr, lr)
        self.assertEqual(tuple(profile.columns), PROFILE_COLUMNS)
        self.assertEqual(len(profile), 3)
        self.assertAlmostEqual(profile.loc[1, "x"], 3.048)
        self.assertAlmostEqual(profile.loc[1, "y"], -3.048)
        self.assertAlmostEqual(profile.loc[1, "z"], 30.48)
        self.assertAlmostEqual(profile.loc[1, "vz"], 30.48)
        self.assertAlmostEqual(profile.loc[1, "e0"], 2**-0.5)
        self.assertAlmostEqual(profile.loc[1, "e3"], 2**-0.5)
        self.assertIsNone(quality["first_near_gyro_limit_time_s"])


if __name__ == "__main__":
    unittest.main()
