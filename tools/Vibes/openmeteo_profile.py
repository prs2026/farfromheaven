"""Fetch one forecast or historical Open-Meteo vertical atmosphere profile.

Example:
  python Vibes/openmeteo_profile.py --latitude 40.88 --longitude -119.08 \
      --local-time 2026-09-26T14:43 --timezone America/Los_Angeles \
      --mode historical --output Vibes/weather_profile.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LEVELS_HPA = (1000, 975, 950, 925, 900, 850, 800, 700, 600, 500,
              400, 300, 250, 200, 150, 100, 70, 50, 30)
GAS_CONSTANT_DRY_AIR = 287.05  # J/(kg K)
HEAT_CAPACITY_RATIO = 1.4
PROFILE_COLUMNS = (
    "altitude_m_asl", "pressure_pa", "temperature_k", "air_density_kgm3",
    "speed_of_sound_mps", "wind_u_mps", "wind_v_mps",
)


def _request_hour(local_time: str, timezone_name: str, mode: str) -> tuple[datetime, str]:
    try:
        naive = datetime.fromisoformat(local_time)
        zone = ZoneInfo(timezone_name)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError(f"Invalid local time or IANA timezone: {exc}") from exc
    if naive.tzinfo is not None:
        raise ValueError("Give a local time without UTC offset; use --timezone separately")
    requested_local = naive.replace(tzinfo=zone)
    # Reject nonexistent spring-forward clock times, rather than silently
    # interpreting them with the wrong UTC offset.
    if requested_local.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) != naive:
        raise ValueError("local time does not exist in the selected timezone (DST transition)")
    # Open-Meteo supplies hourly profiles. Round in UTC so a DST transition
    # cannot create an invalid local hour; record the chosen hour in metadata.
    rounded_utc = (requested_local.astimezone(timezone.utc) + timedelta(minutes=30)).replace(
        minute=0, second=0, microsecond=0)
    requested = rounded_utc.astimezone(zone)
    if mode == "auto":
        mode = "historical" if requested < datetime.now(zone) else "forecast"
    return requested, mode


def _retry_after_seconds(value: str | None, attempt: int, status_code: int) -> float:
    if value:
        try:
            return min(max(float(value), 0.0), 90.0)
        except ValueError:
            try:
                return min(max((parsedate_to_datetime(value) -
                                datetime.now(parsedate_to_datetime(value).tzinfo)).total_seconds(), 0.0), 90.0)
            except (TypeError, ValueError):
                pass
    # A 429 with no Retry-After normally reflects a rolling minute quota.
    return 65.0 if status_code == 429 else min(5.0 * 2**attempt, 60.0)


def _get_json(url: str, retries: int = 2, timeout_s: float = 30.0) -> dict:
    for attempt in range(retries + 1):
        try:
            with urlopen(url, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(_retry_after_seconds(exc.headers.get("Retry-After"), attempt, exc.code))
                continue
            raise RuntimeError(f"Open-Meteo HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Open-Meteo request failed: {exc.reason}") from exc
    raise RuntimeError("Open-Meteo request failed after retries")


def _first(hourly: dict, key: str) -> float | None:
    values = hourly.get(key)
    if not values or values[0] is None:
        return None
    result = float(values[0])
    return result if math.isfinite(result) else None


def _wind_components(speed_mps: float, from_deg: float) -> tuple[float, float]:
    radians = math.radians(from_deg)
    return -speed_mps * math.sin(radians), -speed_mps * math.cos(radians)


def profile_from_response(payload: dict, requested_hour: datetime) -> pd.DataFrame:
    """Turn pressure levels into an ASL-height table with dry-air properties."""
    hourly = payload.get("hourly") or {}
    returned_times = hourly.get("time") or []
    expected = requested_hour.strftime("%Y-%m-%dT%H:00")
    if len(returned_times) != 1 or returned_times[0] != expected:
        raise ValueError(f"Open-Meteo did not return requested local hour {expected}")
    elevation = float(payload.get("elevation", float("nan")))
    if not math.isfinite(elevation):
        raise ValueError("Open-Meteo response has no usable surface elevation")
    surface_p = _first(hourly, "surface_pressure")
    surface_t = _first(hourly, "temperature_2m")
    surface_speed = _first(hourly, "wind_speed_10m")
    surface_direction = _first(hourly, "wind_direction_10m")
    if surface_p is None or surface_t is None:
        raise ValueError("Open-Meteo surface pressure or temperature is missing")
    if surface_speed is None or surface_direction is None:
        raise ValueError("Open-Meteo surface wind is missing")
    u, v = _wind_components(surface_speed, surface_direction)
    rows = [(elevation, surface_p * 100, surface_t + 273.15, u, v)]
    for level in LEVELS_HPA:
        altitude = _first(hourly, f"geopotential_height_{level}hPa")
        temperature = _first(hourly, f"temperature_{level}hPa")
        speed = _first(hourly, f"wind_speed_{level}hPa")
        direction = _first(hourly, f"wind_direction_{level}hPa")
        if any(value is None for value in (altitude, temperature, speed, direction)):
            continue
        if altitude <= elevation:
            continue  # Subsurface pressure levels are not part of the flight profile.
        u, v = _wind_components(speed, direction)
        rows.append((altitude, level * 100, temperature + 273.15, u, v))
    rows.sort(key=lambda row: row[0])
    monotonic = []
    for row in rows:
        if not monotonic or (row[0] > monotonic[-1][0] and row[1] < monotonic[-1][1]):
            monotonic.append(row)
    if len(monotonic) < 2:
        raise ValueError("Open-Meteo returned fewer than two valid levels above the surface")
    array = np.asarray(monotonic, dtype=float)
    pressure = array[:, 1]
    temperature = array[:, 2]
    if np.any(pressure <= 0) or np.any(temperature <= 0):
        raise ValueError("Open-Meteo returned non-physical pressure or temperature")
    density = pressure / (GAS_CONSTANT_DRY_AIR * temperature)
    sound_speed = np.sqrt(HEAT_CAPACITY_RATIO * GAS_CONSTANT_DRY_AIR * temperature)
    return pd.DataFrame({
        "altitude_m_asl": array[:, 0],
        "pressure_pa": pressure,
        "temperature_k": temperature,
        "air_density_kgm3": density,
        "speed_of_sound_mps": sound_speed,
        "wind_u_mps": array[:, 3],
        "wind_v_mps": array[:, 4],
    }, columns=PROFILE_COLUMNS)


def fetch_profile(latitude: float, longitude: float, local_time: str,
                  timezone_name: str = "UTC", mode: str = "auto",
                  elevation_m: float | None = None) -> tuple[pd.DataFrame, dict]:
    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("latitude must be between -90 and 90")
    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError("longitude must be between -180 and 180")
    if mode not in ("auto", "forecast", "historical"):
        raise ValueError("mode must be auto, forecast, or historical")
    if elevation_m is not None and not math.isfinite(elevation_m):
        raise ValueError("elevation must be finite")
    requested, chosen_mode = _request_hour(local_time, timezone_name, mode)
    hourly = ["surface_pressure", "temperature_2m", "wind_speed_10m", "wind_direction_10m"]
    hourly.extend(f"{field}_{level}hPa" for level in LEVELS_HPA
                  for field in ("geopotential_height", "temperature", "wind_speed", "wind_direction"))
    params = {
        "latitude": latitude, "longitude": longitude,
        "timezone": timezone_name,
        "start_hour": requested.strftime("%Y-%m-%dT%H:00"),
        "end_hour": requested.strftime("%Y-%m-%dT%H:00"),
        "temperature_unit": "celsius", "wind_speed_unit": "ms",
        "hourly": ",".join(hourly),
    }
    if elevation_m is not None:
        params["elevation"] = elevation_m
    endpoint = FORECAST_URL if chosen_mode == "forecast" else HISTORICAL_URL
    url = f"{endpoint}?{urlencode(params)}"
    payload = _get_json(url)
    if payload.get("error"):
        raise RuntimeError(f"Open-Meteo error: {payload.get('reason', payload)}")
    profile = profile_from_response(payload, requested)
    metadata = {
        "source": "Open-Meteo modeled atmosphere (not a sounding or flight measurement)",
        "mode": chosen_mode,
        "request_url": url,
        "local_time": requested.isoformat(),
        "input_local_time": local_time,
        "latitude": latitude, "longitude": longitude,
        "surface_elevation_m_asl": float(payload["elevation"]),
        "model_top_m_asl": float(profile["altitude_m_asl"].iloc[-1]),
        "assumptions": [
            "One hourly profile is held fixed for the flight.",
            "Dry-air density = pressure/(287.05*temperature); sound speed = sqrt(1.4*287.05*temperature).",
            "No altitude extrapolation beyond the returned levels is performed.",
        ],
    }
    return profile, metadata


def save_profile(profile: pd.DataFrame, metadata: dict, output: Path,
                 overwrite: bool = False) -> tuple[Path, Path]:
    output = output.resolve()
    notes = output.with_name(output.stem + "_metadata.json")
    for path in (output, notes):
        if path.exists() and not overwrite:
            raise ValueError(f"Output already exists: {path}; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output, index=False, float_format="%.9g")
    notes.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output, notes


def plot_profile(profile: pd.DataFrame, output: Path, overwrite: bool = False) -> Path:
    """Save an overview PNG of the vertical atmospheric profile."""
    if output.exists() and not overwrite:
        raise ValueError(f"Output already exists: {output}; pass --overwrite to replace it")
    altitude_km_agl = (profile["altitude_m_asl"] - profile["altitude_m_asl"].iloc[0]) / 1000
    fig, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=True, constrained_layout=True)
    axes[0].plot(profile["temperature_k"] - 273.15, altitude_km_agl)
    axes[0].set_xlabel("Temperature (°C)")
    axes[1].plot(profile["pressure_pa"] / 1000, altitude_km_agl)
    axes[1].set_xlabel("Pressure (kPa)")
    axes[2].plot(profile["air_density_kgm3"], altitude_km_agl)
    axes[2].set_xlabel("Density (kg/m³)")
    axes[3].plot(profile["wind_u_mps"], altitude_km_agl, label="East")
    axes[3].plot(profile["wind_v_mps"], altitude_km_agl, label="North")
    axes[3].set_xlabel("Wind component (m/s)")
    axes[3].legend()
    axes[0].set_ylabel("Altitude above site (km)")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.suptitle("Open-Meteo modeled atmosphere — one selected hour")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--local-time", required=True,
                        help="YYYY-MM-DDTHH:MM in the chosen timezone; nearest hour is used")
    parser.add_argument("--timezone", default="UTC", help="IANA timezone, e.g. America/Los_Angeles")
    parser.add_argument("--mode", choices=("auto", "forecast", "historical"), default="auto")
    parser.add_argument("--elevation-m", type=float, help="Optional site elevation override in meters ASL")
    parser.add_argument("--output", type=Path, required=True, help="Output atmosphere CSV")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        profile, metadata = fetch_profile(args.latitude, args.longitude,
                                          args.local_time, args.timezone,
                                          args.mode, args.elevation_m)
        plot_path = args.output.resolve().with_suffix(".png")
        if plot_path.exists() and not args.overwrite:
            raise ValueError(f"Output already exists: {plot_path}; pass --overwrite to replace it")
        csv_path, notes_path = save_profile(profile, metadata, args.output, args.overwrite)
        plot_profile(profile, plot_path, args.overwrite)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Profile: {csv_path}")
    print(f"Metadata: {notes_path}")
    print(f"Plot: {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
