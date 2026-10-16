"""UI-independent Blue Raven, weather, and preliminary vibration workflow."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Callable

if __package__:
    from .blue_raven_to_trajectory import convert as convert_blue_raven
    from .openmeteo_profile import fetch_profile, plot_profile, save_profile
    from .prepare_vibration_trajectory import prepare_vibration_trajectory, save_prepared
    from .vibration_input_manifest import build_manifest, save_manifest
    from .vibration_rsf import resolve_csv_paths, resolve_model_options
    from . import preliminary_vibration
else:
    from blue_raven_to_trajectory import convert as convert_blue_raven
    from openmeteo_profile import fetch_profile, plot_profile, save_profile
    from prepare_vibration_trajectory import prepare_vibration_trajectory, save_prepared
    from vibration_input_manifest import build_manifest, save_manifest
    from vibration_rsf import resolve_csv_paths, resolve_model_options
    import preliminary_vibration


def _optional_float(values: dict[str, str], name: str) -> float | None:
    return float(values[name]) if values[name] else None


def _optional_path(values: dict[str, str], name: str) -> Path | None:
    return Path(values[name]) if values[name] else None


def validate_inputs(action: str, values: dict[str, str]) -> None:
    if action not in ("weather", "complete"):
        raise ValueError("action must be weather or complete")
    if not values["output_dir"]:
        raise ValueError("Choose an output parent folder")
    if Path(values["output_dir"]).is_file():
        raise ValueError("Output parent must be a folder, not a file")
    for key in ("latitude", "longitude"):
        try:
            number = float(values[key])
        except ValueError as exc:
            raise ValueError(f"Enter a numeric {key}") from exc
        if not math.isfinite(number):
            raise ValueError(f"{key} must be finite")
    if action == "weather":
        return
    for key in ("length", "station", "motor_mass", "avionics_mass", "damping", "frequency_max"):
        try:
            number = float(values[key])
        except ValueError as exc:
            raise ValueError(f"Enter a numeric value for {key.replace('_', ' ')}") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{key.replace('_', ' ')} must be positive and finite")
    if float(values["station"]) > float(values["length"]):
        raise ValueError("Avionics station cannot exceed vehicle length")
    for key in ("exhaust_velocity", "ref_thrust", "ref_ve", "ref_mass"):
        if values[key]:
            try:
                number = float(values[key])
            except ValueError as exc:
                raise ValueError(f"{key.replace('_', ' ')} must be numeric") from exc
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f"{key.replace('_', ' ')} must be positive and finite")
    try:
        ignition = float(values["ignition_time"])
    except ValueError as exc:
        raise ValueError("Motor ignition time must be numeric") from exc
    if not math.isfinite(ignition):
        raise ValueError("Motor ignition time must be finite")
    if values["end_time"]:
        try:
            end_time = float(values["end_time"])
        except ValueError as exc:
            raise ValueError("Analysis end time must be a positive number") from exc
        if not math.isfinite(end_time) or end_time <= 0:
            raise ValueError("Analysis end time must be positive and finite")
    flight_keys = ("trajectory",) if values["trajectory"] else ("hr", "lr")
    for key in (*flight_keys, "motor_curve", "transfer", "motor_psd"):
        if values[key] and not Path(values[key]).is_file():
            raise ValueError(f"Selected {key.replace('_', ' ')} file does not exist: {values[key]}")
    if not values["trajectory"] and not (values["hr"] and values["lr"]):
        raise ValueError("Choose an existing trajectory or both Blue Raven HR and LR files")
    if bool(values["motor_curve"]) != bool(values["exhaust_velocity"]):
        raise ValueError("Motor curve and assumed exhaust velocity must be supplied together")
    if values["motor_psd"]:
        if not values["motor_curve"]:
            raise ValueError("A reference motor PSD requires a motor thrust curve")
        if not all(values[key] for key in ("ref_thrust", "ref_ve", "ref_mass")):
            raise ValueError("Reference motor PSD requires reference thrust, exhaust velocity and mass")
    elif any(values[key] for key in ("ref_thrust", "ref_ve", "ref_mass")):
        raise ValueError("Reference motor parameters require a reference motor PSD file")


def _write_run_manifest(values: dict[str, str], action: str, run_dir: Path,
                        weather_metadata: dict | None = None,
                        trajectory_path: Path | None = None,
                        vibration_input_path: Path | None = None,
                        options: dict | None = None) -> None:
    manifest = build_manifest(values, action=action, run_dir=run_dir,
                              weather_metadata=weather_metadata,
                              trajectory_path=trajectory_path,
                              vibration_input_path=vibration_input_path, options=options)
    save_manifest(manifest, run_dir / "run_inputs.json", overwrite=True)
    save_manifest(manifest, run_dir / "run_inputs.md", overwrite=True)


def run_workflow(action: str, values: dict[str, str], emit: Callable[[str], None],
                 *, options: dict | None = None) -> Path:
    """Execute the selected workflow and emit progress messages from any thread."""
    raw_values = dict(values)
    options = resolve_model_options(options, values)
    values = resolve_csv_paths(values)
    validate_inputs(action, values)
    if action == "complete" and float(values["frequency_max"]) <= options["frequency_min_hz"]:
        raise ValueError("Maximum PSD frequency must exceed minimum PSD frequency")
    root = Path(values["output_dir"]).expanduser()
    run_dir = root / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    _write_run_manifest(raw_values, action, run_dir, options=options)
    emit(f"Input snapshot: {run_dir / 'run_inputs.md'}")
    emit("Fetching one hourly Open-Meteo profile; a rate-limit retry may take about a minute.")
    profile, weather_notes = fetch_profile(
        float(values["latitude"]), float(values["longitude"]),
        values["local_time"], values["timezone"], values["mode"],
        _optional_float(values, "elevation"),
    )
    weather_csv, _ = save_profile(profile, weather_notes, run_dir / "weather_profile.csv")
    weather_plot = plot_profile(profile, run_dir / "weather_profile.png")
    _write_run_manifest(raw_values, action, run_dir, weather_metadata=weather_notes, options=options)
    emit(f"Weather profile: {weather_csv} ({weather_notes['mode']}, sampled {weather_notes['local_time']})")
    emit(f"Weather plot: {weather_plot}")
    if action == "weather":
        return run_dir

    trajectory = _optional_path(values, "trajectory")
    if trajectory is None:
        trajectory, plot, quality_notes = convert_blue_raven(
            Path(values["hr"]), Path(values["lr"]), run_dir / "blue_raven",
            end_time=_optional_float(values, "end_time"),
        )
        emit(f"Blue Raven trajectory: {trajectory}")
        emit(f"Sensor plot: {plot}")
        emit(f"Quality notes: {quality_notes}")
        quality = json.loads(quality_notes.read_text(encoding="utf-8"))["quality"]
        for warning in quality["warnings"]:
            emit(f"WARNING: {warning}")
    prepared, assumptions = prepare_vibration_trajectory(
        trajectory, weather_csv,
        motor_curve_path=_optional_path(values, "motor_curve"),
        exhaust_velocity_mps=_optional_float(values, "exhaust_velocity"),
        ignition_time_s=float(values["ignition_time"]),
        end_time_s=_optional_float(values, "end_time"),
    )
    prepared_csv, prepared_notes = save_prepared(
        prepared, assumptions, run_dir / "vibration_input.csv")
    _write_run_manifest(raw_values, action, run_dir,
                        weather_metadata=weather_notes,
                        trajectory_path=trajectory,
                        vibration_input_path=prepared_csv, options=options)
    emit(f"Screening input: {prepared_csv}")
    emit(f"Assumptions: {prepared_notes}")
    if not values["motor_curve"]:
        emit("WARNING: No motor curve; powered-flight regimes and motor acceleration cannot be evaluated.")
    if not values["transfer"]:
        emit("WARNING: No structural transfer function; aerodynamic acceleration PSD is unavailable (pressure PSD is still produced).")

    arguments = [
        str(prepared_csv), "--output-dir", str(run_dir / "vibration"),
        "--trajectory-provenance", values["provenance"],
        "--vehicle-length-m", values["length"],
        "--station-from-nose-m", values["station"],
        "--motor-mass-kg", values["motor_mass"],
        "--avionics-mass-kg", values["avionics_mass"],
        "--structural-damping-ratio", values["damping"],
        "--cq", values["cq"],
        "--frequency-max-hz", values["frequency_max"],
    ]
    for key, flag in (("transfer", "--transfer-function"),
                      ("motor_psd", "--motor-psd"),
                      ("ref_thrust", "--reference-thrust-n"),
                      ("ref_ve", "--reference-exhaust-velocity-mps"),
                      ("ref_mass", "--reference-motor-mass-kg")):
        if values[key]:
            arguments.extend((flag, values[key]))
    for key, value in options.items():
        if value is not None:
            arguments.extend(("--" + key.replace("_", "-"), str(value)))
    parsed = preliminary_vibration.parser().parse_args(arguments)
    output = preliminary_vibration.run(parsed)
    emit(f"Vibration report: {output / 'summary.md'}")
    return run_dir
