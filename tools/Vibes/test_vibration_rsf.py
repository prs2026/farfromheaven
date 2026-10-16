"""RSF persistence and GUI callback checks; no display or network required."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from Vibes.vibration_gui import VibrationApp
from Vibes.vibration_rsf import (
    CSV_FIELDS, SETTING_DEFAULTS, load_rsf, model_options, resolve_csv_paths,
    resolve_model_options, save_rsf,
)


class RsfTests(unittest.TestCase):
    def values(self, root):
        return {**SETTING_DEFAULTS, **dict.fromkeys(CSV_FIELDS, ""),
                "csv_inputs_folder": str(root / "csv_inputs"),
                "output_dir": str(root / "results"), "length": "3.25"}

    def test_round_trip_csv_references_and_all_model_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            values = self.values(root)
            csv_dir = root / "csv_inputs"
            csv_dir.mkdir()
            for key in CSV_FIELDS:
                source = csv_dir / f"{key}.csv"
                source.write_text("column\n123\n", encoding="utf-8")
                values[key] = str(source)
            (csv_dir / "tones.csv").write_text("frequency_Hz,acceleration_rms_g\n100,0.1\n")
            options = {"window_s": 0.5, "frequency_min_hz": 30, "motor_tones": "tones.csv"}
            path = save_rsf(values, root / "setup.RSF", options=options)
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(document["csv_inputs_folder"], "csv_inputs")
            self.assertEqual(document["csv_files"]["trajectory"], "trajectory.csv")
            self.assertNotIn("123", path.read_text(encoding="utf-8"))
            loaded, extras, warnings = load_rsf(path)
            self.assertEqual(resolve_csv_paths(loaded), values)
            self.assertEqual(extras, model_options(options))
            self.assertEqual(warnings, [])
            # Paths remain valid after moving the setup and inputs together.
            moved = root / "moved"
            moved.mkdir()
            csv_dir.rename(moved / "csv_inputs")
            path.rename(moved / path.name)
            loaded, extras, warnings = load_rsf(moved / path.name)
            self.assertEqual(warnings, [])
            self.assertEqual(resolve_csv_paths(loaded)["trajectory"], str(moved / "csv_inputs" / "trajectory.csv"))
            self.assertEqual(resolve_model_options(extras, loaded)["motor_tones"], str(moved / "csv_inputs" / "tones.csv"))

    def test_missing_files_are_recoverable_and_overwrite_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            values = self.values(root)
            values["hr"] = "missing.csv"
            path = save_rsf(values, root / "draft.rsf")
            loaded, _, warnings = load_rsf(path)
            self.assertEqual(loaded["hr"], "missing.csv")
            self.assertEqual(len(warnings), 1)
            self.assertIn("Missing hr CSV", warnings[0])
            with self.assertRaises(FileExistsError):
                save_rsf(values, path)
            save_rsf(values, path, overwrite=True)

    def test_invalid_documents_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = save_rsf(self.values(root), root / "setup.RSF")
            original = json.loads(path.read_text())
            invalid = [[], {**original, "version": 2}, {**original, "settings": {}},
                       {**original, "csv_files": []},
                       {**original, "model_options": {**original["model_options"], "window_s": -1}},
                       {**original, "model_options": {**original["model_options"], "window_s": "NaN"}}]
            for document in invalid:
                with self.subTest(document=document):
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_rsf(path)

    def test_gui_load_restores_fields_without_starting_workflow(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = save_rsf(self.values(root), root / "setup.RSF", options={"window_s": 0.25})
            app = Mock()
            app.busy = False
            app.vars = {key: Mock() for key in self.values(root)}
            with patch("Vibes.vibration_gui.filedialog.askopenfilename", return_value=str(path)), \
                 patch("Vibes.vibration_gui.messagebox.askyesno", return_value=True), \
                 patch("Vibes.vibration_gui.run_workflow") as run:
                VibrationApp._load_rsf(app)
            app.vars["length"].set.assert_called_once_with("3.25")
            self.assertEqual(app.model_options["window_s"], 0.25)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
