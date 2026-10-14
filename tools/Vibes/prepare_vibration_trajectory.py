"""Join a standard flight trajectory and Open-Meteo profile for vibration screening.

Uses ground-relative speed as an explicit approximation; no wind correction is
possible from Blue Raven downrange/crossrange axes without a calibrated azimuth.
Motor thrust is never inferred from accelerometer or altitude data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .preliminary_vibration import TRAJECTORY_COLUMNS
    from .openmeteo_profile import PROFILE_COLUMNS
else:
    from preliminary_vibration import TRAJECTORY_COLUMNS
    from openmeteo_profile import PROFILE_COLUMNS


FLIGHT_COLUMNS = ("time", "x", "y", "z", "vx", "vy", "vz", "e0", "e1", "e2", "e3")
MOTOR_COLUMNS = ("time_s", "thrust_N")


def _read_numeric(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
    data = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if len(data) < 2 or not np.isfinite(data.to_numpy()).all():
        raise ValueError(f"{path}: expected at least two rows of finite numeric data")
    return data


def prepare_vibration_trajectory(
    flight_path: Path, weather_path: Path, *,
    motor_curve_path: Path | None = None,
    exhaust_velocity_mps: float | None = None,
    ignition_time_s: float = 0.0,
    start_time_s: float = 0.0,
    end_time_s: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    flight = _read_numeric(flight_path, FLIGHT_COLUMNS)
    weather = _read_numeric(weather_path, PROFILE_COLUMNS)
    ft = flight["time"].to_numpy()
    altitude_profile = weather["altitude_m_asl"].to_numpy()
    if np.any(np.diff(ft) <= 0) or np.any(np.diff(altitude_profile) <= 0):
        raise ValueError("flight times and weather altitudes must be strictly increasing")
    if not math.isfinite(start_time_s) or (end_time_s is not None and
        (not math.isfinite(end_time_s) or end_time_s < start_time_s)):
        raise ValueError("invalid start/end time")
    if not math.isfinite(ignition_time_s):
        raise ValueError("ignition time must be finite")
    mask = ft >= start_time_s
    if end_time_s is not None:
        mask &= ft <= end_time_s
    flight = flight.loc[mask].copy()
    if len(flight) < 2:
        raise ValueError("time selection leaves fewer than two flight samples")
    # The Blue Raven converter defines z as AGL. This makes the atmosphere's
    # absolute geopotential heights comparable to the flight altitude.
    launch_elevation = float(altitude_profile[0])
    z_agl = flight["z"].to_numpy()
    if np.any(z_agl < 0):
        first = int(np.flatnonzero(z_agl < 0)[0])
        raise ValueError(
            f"flight z falls below launch elevation at t={flight['time'].iloc[first]:g} s; "
            "trim the unreliable segment with --end-time-s or supply a corrected trajectory"
        )
    absolute_height = launch_elevation + z_agl
    if absolute_height.max() > altitude_profile[-1]:
        raise ValueError(
            f"flight reaches {absolute_height.max():g} m ASL, above Open-Meteo "
            f"profile top {altitude_profile[-1]:g} m ASL; no extrapolation is performed"
        )
    density = np.interp(absolute_height, altitude_profile,
                        weather["air_density_kgm3"].to_numpy())
    sound_speed = np.interp(absolute_height, altitude_profile,
                            weather["speed_of_sound_mps"].to_numpy())
    if np.any(density <= 0) or np.any(sound_speed <= 0):
        raise ValueError("weather profile contains non-physical density or sound speed")
    # Blue Raven horizontal axes are not georeferenced, so wind components in
    # east/north cannot be subtracted safely. Use ground speed, label it clearly.
    speed = np.linalg.norm(flight[["vx", "vy", "vz"]].to_numpy(), axis=1)
    times = flight["time"].to_numpy()
    thrust = np.zeros(len(flight))
    ve = np.zeros(len(flight))
    if motor_curve_path is not None:
        if exhaust_velocity_mps is None or not math.isfinite(exhaust_velocity_mps) or exhaust_velocity_mps <= 0:
            raise ValueError("motor curve requires a positive --exhaust-velocity-mps")
        motor = _read_numeric(motor_curve_path, MOTOR_COLUMNS)
        motor_time = motor["time_s"].to_numpy()
        motor_thrust = motor["thrust_N"].to_numpy()
        if np.any(np.diff(motor_time) <= 0) or motor_time[0] < 0 or np.any(motor_thrust < 0):
            raise ValueError("motor curve requires increasing nonnegative time and nonnegative thrust")
        # Outside the supplied motor curve, thrust is exactly zero. The user
        # specifies ignition time and an assumed constant exhaust velocity.
        thrust = np.interp(times - ignition_time_s, motor_time, motor_thrust,
                           left=0.0, right=0.0)
        ve[thrust > 0] = exhaust_velocity_mps
    elif exhaust_velocity_mps is not None:
        raise ValueError("exhaust velocity without a motor curve would not define thrust")
    prepared = pd.DataFrame({
        "time_s": times,
        "altitude_m": z_agl,
        "mach": speed / sound_speed,
        "velocity_mps": speed,
        "air_density_kgm3": density,
        "thrust_N": thrust,
        "exhaust_velocity_mps": ve,
    }, columns=TRAJECTORY_COLUMNS)
    notes = {
        "flight_source": str(flight_path.resolve()),
        "weather_source": str(weather_path.resolve()),
        "motor_curve_source": str(motor_curve_path.resolve()) if motor_curve_path else None,
        "ignition_time_s": ignition_time_s if motor_curve_path else None,
        "exhaust_velocity_mps": exhaust_velocity_mps if motor_curve_path else None,
        "launch_elevation_m_asl": launch_elevation,
        "speed_basis": "Ground-relative inertial speed; wind NOT subtracted",
        "limitations": [
            "This is a preliminary screening input, not a validated flight environment.",
            "Blue Raven inertial velocities may be wrong after gyro saturation or deployment.",
            "Atmosphere is a single modeled hourly profile held fixed over the flight.",
            "Motor thrust is zero when no measured/specified motor curve was supplied.",
            "Mach and dynamic pressure use ground speed because Blue Raven horizontal axes are not georeferenced.",
        ],
    }
    return prepared, notes


def save_prepared(data: pd.DataFrame, notes: dict, output: Path) -> tuple[Path, Path]:
    output = output.resolve()
    notes_path = output.with_name(output.stem + "_metadata.json")
    for path in (output, notes_path):
        if path.exists():
            raise ValueError(f"Output already exists: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False, float_format="%.9g")
    notes_path.write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
    return output, notes_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("flight", type=Path, help="standard trajectory CSV, z in meters AGL")
    parser.add_argument("weather", type=Path, help="CSV from openmeteo_profile.py")
    parser.add_argument("--motor-curve", type=Path, help="CSV: time_s,thrust_N")
    parser.add_argument("--exhaust-velocity-mps", type=float)
    parser.add_argument("--ignition-time-s", type=float, default=0.0)
    parser.add_argument("--start-time-s", type=float, default=0.0)
    parser.add_argument("--end-time-s", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        prepared, notes = prepare_vibration_trajectory(
            args.flight, args.weather,
            motor_curve_path=args.motor_curve,
            exhaust_velocity_mps=args.exhaust_velocity_mps,
            ignition_time_s=args.ignition_time_s,
            start_time_s=args.start_time_s,
            end_time_s=args.end_time_s,
        )
        output, metadata = save_prepared(prepared, notes, args.output)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    print(f"Vibration trajectory: {output}")
    print(f"Assumptions: {metadata}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
