"""Checks that exported input manifests include GUI settings and model defaults."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from Vibes import preliminary_vibration
from Vibes.vibration_input_manifest import build_manifest, render_manifest, save_manifest


class VibrationInputManifestTests(unittest.TestCase):
    def test_all_model_options_and_missing_input_warnings(self):
        values = {
            "trajectory": "", "hr": "hr.csv", "lr": "lr.csv",
            "end_time": "23.3", "provenance": "measured",
            "latitude": "40.88", "longitude": "-119.08",
            "local_time": "2026-09-26T14:43", "timezone": "America/Los_Angeles",
            "mode": "historical", "elevation": "",
            "motor_curve": "", "exhaust_velocity": "", "ignition_time": "0",
            "length": "1.5", "station": "0.1", "motor_mass": "2",
            "avionics_mass": "0.1", "damping": "0.99",
            "transfer": "", "motor_psd": "", "ref_thrust": "",
            "ref_ve": "", "ref_mass": "", "cq": "0.02",
            "frequency_max": "2000", "output_dir": "output",
        }
        manifest = build_manifest(values)
        parser_dests = {
            action.dest for action in preliminary_vibration.parser()._actions
            if action.dest != "help" and action.default is not argparse.SUPPRESS
        }
        self.assertEqual(set(manifest["preliminary_vibration_inputs"]), parser_dests)
        self.assertEqual(manifest["preliminary_vibration_inputs"]["structural_damping_ratio"], 0.99)
        self.assertEqual(manifest["preliminary_vibration_inputs"]["frequency_min_hz"], 20.0)
        self.assertIn("No thrust curve", " ".join(manifest["input_warnings"]))
        self.assertIn("No structural transfer", " ".join(manifest["input_warnings"]))
        self.assertIn("damping ratio is high", " ".join(manifest["input_warnings"]))
        self.assertIn("## Exact GUI field values", render_manifest(manifest))
        with tempfile.TemporaryDirectory() as temp:
            json_file = save_manifest(manifest, Path(temp) / "inputs.json")
            md_file = save_manifest(manifest, Path(temp) / "inputs.md")
            self.assertEqual(json.loads(json_file.read_text(encoding="utf-8"))["schema"],
                             manifest["schema"])
            self.assertIn("## Effective vibration model inputs",
                          md_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
