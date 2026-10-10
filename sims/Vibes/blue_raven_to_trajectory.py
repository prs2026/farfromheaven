"""Plot Blue Raven HR sensors and export its LR inertial flight trajectory.

The output is a measured/reconstructed trajectory, not a validated truth track.
Blue Raven inertial navigation can drift badly after gyro saturation or deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp


PROFILE_COLUMNS = ("time", "x", "y", "z", "vx", "vy", "vz", "e0", "e1", "e2", "e3")
FEET_TO_METERS = 0.3048
HR_COLUMNS = (
    "Flight_Time_(s)", "Accel_X", "Accel_Y", "Accel_Z",
    "Gyro_X", "Gyro_Y", "Gyro_Z",
    "Quat_1", "Quat_2", "Quat_3", "Quat_4",
)
LR_COLUMNS = (
    "Flight_Time_(s)", "Inertial_DR_Position", "Inertial_CR_position",
    "Inertial_Altitude", "Velocity_DR", "Velocity_CR", "Velocity_Up",
)


def _read_numeric(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")
    # Read the header first so missing fields have a useful error message.
    headers = pd.read_csv(path, nrows=0).columns
    missing = [name for name in columns if name not in headers]
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
    frame = pd.read_csv(path, usecols=list(columns)).loc[:, list(columns)]
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path}: required channels contain missing or non-finite values")
    times = frame["Flight_Time_(s)"].to_numpy()
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError(f"{path}: flight time must have at least two strictly increasing samples")
    return frame


def _unwrap_signed_16_bit_positions(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Undo the 65536-foot discontinuities in Blue Raven horizontal positions."""
    jumps = np.diff(values)
    wraps = np.rint(jumps / 65536.0).astype(np.int64)
    wraps[np.abs(jumps) <= 32768] = 0
    corrected = values.astype(float).copy()
    corrected[1:] -= 65536.0 * np.cumsum(wraps)
    return corrected, int(np.count_nonzero(wraps))


def build_trajectory(
    high_rate: pd.DataFrame, low_rate: pd.DataFrame, start_time: float = 0.0,
    end_time: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Use LR ground-relative positions/velocities and interpolate HR attitude."""
    hr_time = high_rate["Flight_Time_(s)"].to_numpy(dtype=float)
    lr_time = low_rate["Flight_Time_(s)"].to_numpy(dtype=float)
    mask = (lr_time >= start_time) & (lr_time >= hr_time[0]) & (lr_time <= hr_time[-1])
    if end_time is not None:
        mask &= lr_time <= end_time
    if not mask.any():
        raise ValueError("No low-rate samples lie in the requested time range and HR overlap")

    # The manual specifies Quat_1..4 as vector x/y/z, then scalar magnitude.
    # SciPy expects that same x/y/z/w ordering; the output profile is w/x/y/z.
    source_quat = high_rate[["Quat_1", "Quat_2", "Quat_3", "Quat_4"]].to_numpy(dtype=float)
    quat_norm = np.linalg.norm(source_quat, axis=1)
    if np.any(quat_norm < 1e-8):
        raise ValueError("High-rate data contains a zero-length quaternion")
    source_quat = source_quat / quat_norm[:, None]
    interpolator = Slerp(hr_time, Rotation.from_quat(source_quat))
    interpolated_quat = interpolator(lr_time[mask]).as_quat()

    # Position fields are signed 16-bit feet in the export and may wrap after
    # +/-32768 ft. Unwrap before selecting a time range to preserve continuity.
    downrange, dr_wraps = _unwrap_signed_16_bit_positions(
        low_rate["Inertial_DR_Position"].to_numpy(dtype=float)
    )
    crossrange, cr_wraps = _unwrap_signed_16_bit_positions(
        low_rate["Inertial_CR_position"].to_numpy(dtype=float)
    )
    profile = pd.DataFrame({
        "time": lr_time[mask],
        "x": downrange[mask] * FEET_TO_METERS,
        "y": crossrange[mask] * FEET_TO_METERS,
        "z": low_rate.loc[mask, "Inertial_Altitude"].to_numpy() * FEET_TO_METERS,
        "vx": low_rate.loc[mask, "Velocity_DR"].to_numpy() * FEET_TO_METERS,
        "vy": low_rate.loc[mask, "Velocity_CR"].to_numpy() * FEET_TO_METERS,
        "vz": low_rate.loc[mask, "Velocity_Up"].to_numpy() * FEET_TO_METERS,
        "e0": interpolated_quat[:, 3],
        "e1": interpolated_quat[:, 0],
        "e2": interpolated_quat[:, 1],
        "e3": interpolated_quat[:, 2],
    }, columns=PROFILE_COLUMNS)
    gyro = high_rate[["Gyro_X", "Gyro_Y", "Gyro_Z"]].to_numpy()
    saturated = np.any(np.abs(gyro) >= 1990.0, axis=1)
    quality = {
        "horizontal_position_wraps_unwrapped": {"downrange": dr_wraps, "crossrange": cr_wraps},
        "first_near_gyro_limit_time_s": float(hr_time[np.argmax(saturated)]) if saturated.any() else None,
        "gyro_limit_threshold_deg_per_s": 1990.0,
        "warnings": [
            "Blue Raven inertial navigation is not an independent position measurement; "
            "gyro saturation, deployment shocks and integration drift can make later positions, "
            "velocities and attitude unreliable. Review against barometric or GPS data."
        ],
    }
    if saturated.any():
        quality["warnings"].append(
            "Gyro readings approach the +/-2000 deg/s sensor range; attitude and "
            "inertial navigation after the first such reading may be unreliable."
        )
    if dr_wraps or cr_wraps:
        quality["warnings"].append(
            "Horizontal position integer rollovers were unwrapped, but this does not "
            "correct inertial-navigation drift."
        )
    return profile, quality


def plot_sensors(high_rate: pd.DataFrame, output: Path) -> None:
    """Plot all six raw, body/sensor-axis HR channels against flight time."""
    times = high_rate["Flight_Time_(s)"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for axis, prefix, ylabel in (
        (axes[0], "Accel", "Acceleration (g)"),
        (axes[1], "Gyro", "Angular rate (deg/s)"),
    ):
        for component in "XYZ":
            axis.plot(times, high_rate[f"{prefix}_{component}"], label=component, linewidth=0.7)
        axis.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right", ncol=3)
    axes[0].set_title("Blue Raven high-rate sensor measurements (sensor axes)")
    axes[1].set_xlabel("Flight time since liftoff (s)")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def convert(
    hr_path: Path, lr_path: Path, output_dir: Path,
    start_time: float = 0.0, end_time: float | None = None,
) -> tuple[Path, Path, Path]:
    high_rate = _read_numeric(hr_path, HR_COLUMNS)
    low_rate = _read_numeric(lr_path, LR_COLUMNS)
    profile, quality = build_trajectory(high_rate, low_rate, start_time, end_time)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "flight_trajectory.csv"
    plot_path = output_dir / "acceleration_gyro.png"
    notes_path = output_dir / "flight_trajectory_notes.json"
    profile.to_csv(trajectory_path, index=False, float_format="%.9g")
    plot_sensors(high_rate, plot_path)
    notes = {
        "high_rate_source": str(hr_path.resolve()),
        "low_rate_source": str(lr_path.resolve()),
        "output_rows": len(profile),
        "coordinate_frame": "Blue Raven launch-relative downrange/crossrange/up, not geographic east/north",
        "position_units": "meters, relative to launch point",
        "velocity_units": "meters/second",
        "attitude": "Blue Raven sensor-frame quaternion, reordered vector-first to scalar-first; "
                    "not transformed into RocketPy body/ENU frame",
        "quality": quality,
    }
    notes_path.write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
    return trajectory_path, plot_path, notes_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("high_rate_csv", type=Path, help="Blue Raven HR CSV")
    parser.add_argument("low_rate_csv", type=Path, help="Matching Blue Raven LR CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("blue_raven_output"))
    parser.add_argument("--start-time", type=float, default=0.0,
                        help="First trajectory time in seconds (default: liftoff, 0 s)")
    parser.add_argument("--end-time", type=float, help="Last trajectory time in seconds")
    args = parser.parse_args(argv)
    if not np.isfinite(args.start_time) or (args.end_time is not None and
        (not np.isfinite(args.end_time) or args.end_time < args.start_time)):
        parser.error("time bounds must be finite and end-time must be >= start-time")
    try:
        trajectory, plot, notes = convert(
            args.high_rate_csv, args.low_rate_csv, args.output_dir,
            args.start_time, args.end_time,
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    print(f"Trajectory: {trajectory}")
    print(f"Sensor plot: {plot}")
    print(f"Quality notes: {notes}")
    quality = json.loads(notes.read_text(encoding="utf-8"))["quality"]
    for warning in quality["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
