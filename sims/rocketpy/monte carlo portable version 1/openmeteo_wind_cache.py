"""Fetch and cache the Open-Meteo profiles used by Monte Carlo simulations.

Usage::

    python openmeteo_wind_cache.py montecarlo_config.json
    python openmeteo_wind_cache.py montecarlo_config.json --output weather.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from openmeteo_environment import create_openmeteo_environment


CACHE_FORMAT = "projectblaze.openmeteo.wind_profiles"
CACHE_FORMAT_VERSION = 1


def _sample_hours(calls_per_day: int) -> tuple[int, ...]:
    if isinstance(calls_per_day, bool) or not isinstance(calls_per_day, int):
        raise ValueError("wind_sampling.calls_per_day must be an integer")
    if not 1 <= calls_per_day <= 24:
        raise ValueError("wind_sampling.calls_per_day must be in the range [1, 24]")
    return tuple((index * 24) // calls_per_day for index in range(calls_per_day))


def _config_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _output_path(
    config_path: Path, wind_sampling: dict[str, Any], override: Path | None
) -> Path:
    if override is not None:
        path = override
    else:
        value = wind_sampling.get("cache_file")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "environment.wind_sampling.cache_file must be set or --output supplied"
            )
        path = Path(value)
    return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def build_cache(config_path: str | Path, output_override: Path | None = None) -> Path:
    config_path = Path(config_path).expanduser().resolve()
    config = _config_object(
        json.loads(config_path.read_text(encoding="utf-8")), "configuration root"
    )
    environment = _config_object(config.get("environment", {}), "environment")
    wind_sampling = _config_object(
        environment.get("wind_sampling", {}), "environment.wind_sampling"
    )
    launch_date = str(environment["date"])
    launch_time = str(environment["time"])
    launch_hour = datetime.fromisoformat(f"{launch_date}T{launch_time}")
    calls_per_day = wind_sampling.get("calls_per_day", 8)
    sample_hours = _sample_hours(calls_per_day)
    days_either_side = wind_sampling.get("days_either_side", 2)
    if (
        isinstance(days_either_side, bool)
        or not isinstance(days_either_side, int)
        or days_either_side < 0
    ):
        raise ValueError("wind_sampling.days_either_side must be non-negative")

    latitude = float(environment.get("latitude", 40.870683))
    longitude = float(environment.get("longitude", -119.106950))
    elevation_m = float(environment.get("elevation_m", 1191.0))
    timezone_name = str(environment.get("timezone", "America/Los_Angeles"))
    model = environment.get("model", "gfs_seamless")
    endpoint = str(environment.get("endpoint", "forecast"))
    max_expected_height_m = float(environment.get("max_expected_height_m", 80_000.0))
    extend_above_model_top = bool(environment.get("extend_above_model_top", True))
    settings = {
        "latitude": latitude,
        "longitude": longitude,
        "elevation_m": elevation_m,
        "timezone": timezone_name,
        "model": model,
        "endpoint": endpoint,
        "max_expected_height_m": max_expected_height_m,
        "extend_above_model_top": extend_above_model_top,
        "launch_date": launch_date,
        "launch_time": launch_time,
        "calls_per_day": calls_per_day,
        "days_either_side": days_either_side,
    }
    dates = [
        launch_hour.date() + timedelta(days=offset)
        for offset in range(-days_either_side, days_either_side + 1)
    ]
    requests = [(date, hour) for date in dates for hour in sample_hours]
    profiles = []
    print(f"Fetching {len(requests)} Open-Meteo profiles for cache", flush=True)
    for request_index, (sample_date, sample_hour) in enumerate(requests, start=1):
        date_text = sample_date.isoformat()
        time_text = f"{sample_hour:02d}:00"
        print(
            f"Open-Meteo call {request_index}/{len(requests)}: "
            f"{date_text} {time_text} {timezone_name}",
            flush=True,
        )
        result = create_openmeteo_environment(
            latitude=latitude,
            longitude=longitude,
            date=date_text,
            time=time_text,
            timezone_name=timezone_name,
            elevation_m=elevation_m,
            model=model,
            endpoint=endpoint,
            max_expected_height_m=max_expected_height_m,
            extend_above_model_top=extend_above_model_top,
        )
        profiles.append(
            {
                "date": date_text,
                "time": time_text,
                "pressure": result.pressure.tolist(),
                "temperature": result.temperature.tolist(),
                "wind_u": result.wind_u.tolist(),
                "wind_v": result.wind_v.tolist(),
                "model_top_height_m": result.model_top_height_m,
            }
        )
        print(f"Completed Open-Meteo call {request_index}/{len(requests)}", flush=True)

    payload = {
        "format": CACHE_FORMAT,
        "format_version": CACHE_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
        "profiles": profiles,
    }
    output_path = _output_path(config_path, wind_sampling, output_override)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Saved {len(profiles)} weather profiles to {output_path}", flush=True)
    return output_path


@dataclass(frozen=True)
class CachedEnvironmentProvider:
    settings: dict[str, Any]
    profiles: dict[tuple[str, str], dict[str, Any]]
    cache_path: Path

    def __call__(self, **request: Any) -> SimpleNamespace:
        key = (str(request["date"]), str(request["time"]))
        try:
            profile = self.profiles[key]
        except KeyError as exc:
            raise RuntimeError(
                f"weather cache {self.cache_path} has no profile for {key[0]} {key[1]}; "
                "regenerate it with openmeteo_wind_cache.py"
            ) from exc
        checks = {
            "latitude": request["latitude"],
            "longitude": request["longitude"],
            "elevation_m": request["elevation_m"],
            "timezone": request["timezone_name"],
            "model": request["model"],
            "endpoint": request["endpoint"],
            "max_expected_height_m": request["max_expected_height_m"],
            "extend_above_model_top": request["extend_above_model_top"],
        }
        for name, requested in checks.items():
            cached = self.settings.get(name)
            matches = (
                math.isclose(float(cached), float(requested), rel_tol=0, abs_tol=1e-9)
                if isinstance(requested, (int, float)) and not isinstance(requested, bool)
                else cached == requested
            )
            if not matches:
                raise RuntimeError(
                    f"weather cache setting {name!r} is {cached!r}, not {requested!r}; "
                    "regenerate the cache"
                )
        return SimpleNamespace(
            environment=SimpleNamespace(elevation=float(self.settings["elevation_m"])),
            pressure=np.asarray(profile["pressure"], dtype=float),
            temperature=np.asarray(profile["temperature"], dtype=float),
            wind_u=np.asarray(profile["wind_u"], dtype=float),
            wind_v=np.asarray(profile["wind_v"], dtype=float),
            model_top_height_m=float(profile["model_top_height_m"]),
            request_url=f"cache:{self.cache_path}",
            response={},
        )


def load_cached_environment_provider(path: str | Path) -> CachedEnvironmentProvider:
    cache_path = Path(path).expanduser().resolve()
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if payload.get("format") != CACHE_FORMAT or payload.get("format_version") != 1:
        raise ValueError(f"unsupported Open-Meteo cache format: {cache_path}")
    profile_list = payload.get("profiles")
    if not isinstance(profile_list, list) or not profile_list:
        raise ValueError(f"Open-Meteo cache contains no profiles: {cache_path}")
    profiles = {(str(item["date"]), str(item["time"])): item for item in profile_list}
    return CachedEnvironmentProvider(dict(payload.get("settings", {})), profiles, cache_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", type=Path, help="Monte Carlo JSON configuration")
    parser.add_argument("--output", type=Path, help="override wind_sampling.cache_file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        build_cache(args.config_file, args.output)
    except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
