"""Create a reproducible, human-readable record of GUI vibration inputs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from . import preliminary_vibration
    from .vibration_rsf import resolve_csv_paths, resolve_model_options
else:
    import preliminary_vibration
    from vibration_rsf import resolve_csv_paths, resolve_model_options


SCHEMA = "projectblaze-preliminary-vibration-inputs-v1"


def _number(value: str) -> float | None | str:
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return number if math.isfinite(number) else value


def _path(value: str) -> str | None:
    if not value:
        return None
    try:
        return str(Path(value).expanduser().resolve())
    except (OSError, ValueError, RuntimeError):
        # The live preview must also work while a path is only partially typed.
        return value


def _model_defaults() -> dict[str, Any]:
    """Read defaults from the actual screening parser, avoiding duplicate constants."""
    parser = preliminary_vibration.parser()
    return {
        action.dest: action.default
        for action in parser._actions
        if action.dest != "help" and action.default is not argparse.SUPPRESS
    }


def build_manifest(values: dict[str, str], *, action: str = "complete",
                   run_dir: Path | None = None,
                   weather_metadata: dict | None = None,
                   trajectory_path: Path | None = None,
                   vibration_input_path: Path | None = None,
                   options: dict | None = None) -> dict[str, Any]:
    """Include every GUI field, bridge assumption, and model default/override."""
    raw = dict(sorted(values.items()))
    values = resolve_csv_paths(values)
    uses_existing = bool(values.get("trajectory"))
    flight = {
        "source_type": "standard_trajectory_csv" if uses_existing else "blue_raven_hr_lr",
        "existing_trajectory_csv": _path(values.get("trajectory", "")),
        "blue_raven_hr_csv": _path(values.get("hr", "")) if not uses_existing else None,
        "blue_raven_lr_csv": _path(values.get("lr", "")) if not uses_existing else None,
        "analysis_end_time_s": _number(values.get("end_time", "")),
        "trajectory_provenance": values.get("provenance") or "unspecified",
    }
    atmosphere = {
        "latitude_deg": _number(values.get("latitude", "")),
        "longitude_deg": _number(values.get("longitude", "")),
        "input_local_time": values.get("local_time") or None,
        "iana_timezone": values.get("timezone") or None,
        "requested_mode": values.get("mode") or "auto",
        "site_elevation_override_m_asl": _number(values.get("elevation", "")),
        "source": "Open-Meteo modeled atmosphere",
    }
    preparation = {
        "motor_thrust_curve_csv": _path(values.get("motor_curve", "")),
        "motor_curve_columns": ["time_s", "thrust_N"],
        "assumed_exhaust_velocity_mps": _number(values.get("exhaust_velocity", "")),
        "motor_ignition_time_s": _number(values.get("ignition_time", "0")),
        "speed_basis": "ground-relative inertial speed; wind not subtracted",
        "altitude_basis": "flight z is AGL; atmosphere profile heights are ASL",
        "atmosphere_time_basis": "nearest Open-Meteo hour held constant over flight",
        "no_altitude_extrapolation": True,
    }
    model = _model_defaults()
    model.update(resolve_model_options(options, values))
    model.update({
        "trajectory": str(vibration_input_path.resolve()) if vibration_input_path else "<generated vibration_input.csv>",
        "output_dir": str((run_dir / "vibration").resolve()) if run_dir else "<timestamped result folder>/vibration",
        "trajectory_provenance": values.get("provenance") or "unspecified",
        "vehicle_length_m": _number(values.get("length", "")),
        "station_from_nose_m": _number(values.get("station", "")),
        "motor_mass_kg": _number(values.get("motor_mass", "")),
        "avionics_mass_kg": _number(values.get("avionics_mass", "")),
        "structural_damping_ratio": _number(values.get("damping", "")),
        "cq": _number(values.get("cq", "0.02")),
        "frequency_max_hz": _number(values.get("frequency_max", "2000")),
        "motor_psd": _path(values.get("motor_psd", "")),
        "transfer_function": _path(values.get("transfer", "")),
        "reference_thrust_n": _number(values.get("ref_thrust", "")),
        "reference_exhaust_velocity_mps": _number(values.get("ref_ve", "")),
        "reference_motor_mass_kg": _number(values.get("ref_mass", "")),
    })
    warnings = ["PRELIMINARY SCREENING ESTIMATES; not a vibration qualification level."]
    if not preparation["motor_thrust_curve_csv"]:
        warnings.append("No thrust curve: powered ascent, ignition, cutoff, coast and motor scaling are unavailable.")
    if not model["transfer_function"]:
        warnings.append("No structural transfer function: aerodynamic acceleration PSD and total PSD are unavailable.")
    if not model["motor_psd"]:
        warnings.append("No reference motor PSD: motor acceleration PSD cannot be estimated from thrust alone.")
    damping = model["structural_damping_ratio"]
    if isinstance(damping, float) and damping >= 0.2:
        warnings.append("Structural damping ratio is high; verify it. The current model records damping but does not apply it without a transfer function.")
    if not flight["analysis_end_time_s"] and flight["source_type"] == "blue_raven_hr_lr":
        warnings.append("Full Blue Raven inertial track selected: late-flight drift or negative altitude may require an analysis end time.")
    resolved = {
        "run_directory": str(run_dir.resolve()) if run_dir else None,
        "weather_csv": str((run_dir / "weather_profile.csv").resolve()) if run_dir else None,
        "trajectory_csv": str(trajectory_path.resolve()) if trajectory_path else None,
        "vibration_input_csv": str(vibration_input_path.resolve()) if vibration_input_path else None,
        "weather_metadata": weather_metadata,
    }
    return {
        "schema": SCHEMA,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_action": action,
        "flight": flight,
        "atmosphere": atmosphere,
        "vibration_preparation": preparation,
        "preliminary_vibration_inputs": model,
        "all_gui_field_values": raw,
        "resolved_files_and_weather": resolved,
        "input_warnings": warnings,
    }


def render_manifest(manifest: dict[str, Any]) -> str:
    """Format all inputs as a compact, readable Markdown page."""
    lines = ["# Preliminary vibration simulation inputs", "",
             f"Captured (UTC): {manifest['captured_at_utc']}",
             f"Action: {manifest['workflow_action']}", ""]
    sections = (
        ("Flight data", "flight"),
        ("Atmosphere", "atmosphere"),
        ("Trajectory preparation", "vibration_preparation"),
        ("Effective vibration model inputs", "preliminary_vibration_inputs"),
        ("Resolved files and fetched weather", "resolved_files_and_weather"),
        ("Exact GUI field values", "all_gui_field_values"),
    )
    for title, key in sections:
        lines.extend((f"## {title}", ""))
        for name, value in manifest[key].items():
            if isinstance(value, dict):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            lines.append(f"- {name}: {value if value is not None else 'not supplied'}")
        lines.append("")
    lines.extend(("## Input warnings", ""))
    lines.extend(f"- {warning}" for warning in manifest["input_warnings"])
    lines.append("")
    return "\n".join(lines)


def save_manifest(manifest: dict[str, Any], output: Path, *, overwrite: bool = False) -> Path:
    """Write JSON or Markdown depending on the chosen suffix."""
    output = output.expanduser().resolve()
    if output.suffix.lower() not in (".json", ".md"):
        raise ValueError("Input export must end in .json or .md")
    if output.exists() and not overwrite:
        raise ValueError(f"Input export already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        content = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    else:
        content = render_manifest(manifest)
    output.write_text(content, encoding="utf-8")
    return output
