"""Monte Carlo helpers for RocketPy flights.

The pickle produced here stores RocketPy's complete numerical flight solution
for every run. Live ``Flight`` objects are intentionally not pickled because
RocketPy environments contain interpolation callables that standard
``pickle`` cannot serialize reliably.

Standalone usage::

    python montecarlorunner.py montecarlo_config.json

Every entry in ``simulation.parameters`` may be a fixed number, a normal
distribution object with ``mean`` and ``std``, or a deterministic sweep object
with ``min`` and ``max``. ``gaussian`` and the legacy ``std_dev`` field remain
supported. Set ``simulation.workers`` to the number of launches to run
concurrently. Paths are resolved relative to the configuration file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from multiprocessing import get_context
from pathlib import Path
from threading import BrokenBarrierError
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from rocketpy import Environment, Flight, Rocket

try:
    import dill
except ImportError:  # Only required when workers > 1.
    dill = None

from simrunner import runfullstacksim

STATE_COLUMNS = (
    "time",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "e0",
    "e1",
    "e2",
    "e3",
    "omega_x",
    "omega_y",
    "omega_z",
)

BLACK_ROCK_LATITUDE = 40.870683
BLACK_ROCK_LONGITUDE = -119.106950
BLACK_ROCK_ELEVATION_M = 1191.0
BLACK_ROCK_TIMEZONE = "America/Los_Angeles"

SIMULATION_PARAMETER_DEFAULTS = {
    "launch_angle": 89.0,
    "heading": 0.0,
    "rail_length": 6.0,
    "max_time": 600.0,
    "max_time_step": 0.5,
    "rtol": 1e-4,
    "atol": 1e-6,
    "coast_period": 5.0,
    "sustainer_ignition_max_tilt_deg": 90.0,
    "sustainer_booster_impulse_ratio_percent": 100.0,
    "sustainer_booster_mass_ratio_percent": 100.0,
    "booster_wet_mass_percent": 100.0,
    "sustainer_wet_mass_percent": 100.0,
}

_PROCESS_RUN_CASE = None
_NATIVE_THREAD_ENVIRONMENT_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


@contextmanager
def _native_thread_limit(thread_count: int):
    """Limit nested numerical-library threads inherited by spawned workers."""

    previous = {
        variable: os.environ.get(variable)
        for variable in _NATIVE_THREAD_ENVIRONMENT_VARIABLES
    }
    value = str(thread_count)
    for variable in _NATIVE_THREAD_ENVIRONMENT_VARIABLES:
        os.environ[variable] = value
    try:
        yield
    finally:
        for variable, previous_value in previous.items():
            if previous_value is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous_value


def _initialize_process_worker(context_path: str, startup_barrier: Any) -> None:
    """Load a private context and wait until every worker is ready."""

    global _PROCESS_RUN_CASE
    if dill is None:  # pragma: no cover - checked in the parent process
        raise RuntimeError("parallel simulation requires the dill package")
    with Path(context_path).open("rb") as context_file:
        _PROCESS_RUN_CASE = dill.load(context_file)
    print(f"Worker {os.getpid()} loaded simulation context", flush=True)
    try:
        startup_barrier.wait(timeout=300)
    except BrokenBarrierError as exc:
        raise RuntimeError(
            "Monte Carlo workers did not initialize within 300 seconds"
        ) from exc


def _execute_process_case(
    simulation_index: int, parameters: Mapping[str, float]
) -> tuple[Path, int]:
    """Execute one case using the context installed in this worker."""

    if _PROCESS_RUN_CASE is None:
        raise RuntimeError("Monte Carlo worker was not initialized")
    return _PROCESS_RUN_CASE(simulation_index, parameters)


def _as_rocket_tuple(rockets: Rocket | Sequence[Rocket]) -> tuple[Rocket, ...]:
    if isinstance(rockets, Rocket):
        return (rockets,)

    rocket_tuple = tuple(rockets)
    if not rocket_tuple:
        raise ValueError("rockets must contain at least one Rocket object")
    if not all(isinstance(rocket, Rocket) for rocket in rocket_tuple):
        raise TypeError("every item in rockets must be a Rocket object")
    return rocket_tuple


def _selected_rocket_key(rocket: Rocket) -> str | None:
    configuration = getattr(rocket, "source_configuration", None)
    if isinstance(configuration, Mapping):
        selected = configuration.get("selected_rocket")
        if selected is not None:
            return str(selected).strip().lower()
    return None


def _multistage_pair(
    rockets: tuple[Rocket, ...],
) -> tuple[Rocket, Rocket] | None:
    if len(rockets) != 2:
        return None

    identified = {
        key: rocket
        for rocket in rockets
        if (key := _selected_rocket_key(rocket)) is not None
    }
    if set(identified) == {"full_stack", "sustainer"}:
        return identified["full_stack"], identified["sustainer"]
    if set(identified) == {"booster", "sustainer"}:
        raise ValueError(
            "A staged simulation requires the 'full_stack' and 'sustainer' "
            "rocket definitions. The standalone 'booster' is the separated, "
            "motorless booster object."
        )

    by_name = {str(getattr(rocket, "name", "")).lower(): rocket for rocket in rockets}
    full_stack = next(
        (rocket for name, rocket in by_name.items() if "full stack" in name), None
    )
    sustainer = next(
        (rocket for name, rocket in by_name.items() if "sustainer" in name), None
    )
    return (full_stack, sustainer) if full_stack and sustainer else None


def _validate_parameter(name: str, value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"simulation parameter {name!r} must be finite")
    if name == "heading":
        return value % 360.0
    if name == "launch_angle" and not 0 <= value <= 90:
        raise ValueError("launch_angle must be in the range [0, 90] degrees")
    if name in {
        "rail_length",
        "max_time",
        "max_time_step",
        "rtol",
        "atol",
        "sustainer_booster_impulse_ratio_percent",
        "sustainer_booster_mass_ratio_percent",
        "booster_wet_mass_percent",
        "sustainer_wet_mass_percent",
    } and value <= 0:
        raise ValueError(f"{name} must be positive")
    if name == "coast_period" and value < 0:
        raise ValueError("coast_period cannot be negative")
    if name == "sustainer_ignition_max_tilt_deg" and not 0 <= value <= 90:
        raise ValueError(
            "sustainer_ignition_max_tilt_deg must be in the range [0, 90] degrees"
        )
    return value


def _sample_parameter(
    name: str,
    definition: Any,
    simulation_index: int,
    number_of_simulations: int,
    random_generator: random.Random,
) -> float:
    if not isinstance(definition, Mapping):
        return _validate_parameter(name, definition)

    distribution_value = definition.get("distribution")
    if distribution_value is None:
        if "mean" in definition and ("std" in definition or "std_dev" in definition):
            distribution = "normal"
        elif "min" in definition and "max" in definition:
            distribution = "sweep"
        else:
            distribution = "fixed"
    else:
        distribution = str(distribution_value).strip().lower()
    if distribution in {"gaussian", "standard"}:
        distribution = "normal"
    if distribution == "fixed":
        if "value" not in definition:
            raise ValueError(f"fixed parameter {name!r} requires 'value'")
        return _validate_parameter(name, definition["value"])

    if distribution == "sweep":
        if "min" not in definition or "max" not in definition:
            raise ValueError(f"sweep parameter {name!r} requires 'min' and 'max'")
        minimum = float(definition["min"])
        maximum = float(definition["max"])
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError(f"sweep bounds for {name!r} must be finite")
        if maximum < minimum:
            raise ValueError(f"sweep parameter {name!r} has max below min")
        fraction = (
            simulation_index / (number_of_simulations - 1)
            if number_of_simulations > 1
            else 0.0
        )
        return _validate_parameter(name, minimum + fraction * (maximum - minimum))

    if distribution == "normal":
        if "mean" not in definition or not (
            "std" in definition or "std_dev" in definition
        ):
            raise ValueError(
                f"normal parameter {name!r} requires 'mean' and 'std'"
            )
        mean = float(definition["mean"])
        if "std" in definition and "std_dev" in definition:
            std = float(definition["std"])
            legacy_std = float(definition["std_dev"])
            if std != legacy_std:
                raise ValueError(
                    f"normal parameter {name!r} has conflicting 'std' and 'std_dev'"
                )
            standard_deviation = std
        else:
            standard_deviation = float(
                definition.get("std", definition.get("std_dev"))
            )
        if not math.isfinite(mean) or not math.isfinite(standard_deviation):
            raise ValueError(f"normal settings for {name!r} must be finite")
        if standard_deviation < 0:
            raise ValueError(f"normal std for {name!r} cannot be negative")
        minimum = definition.get("min")
        maximum = definition.get("max")
        for _ in range(10_000):
            value = (
                mean
                if standard_deviation == 0
                else random_generator.gauss(mean, standard_deviation)
            )
            if name == "heading":
                value %= 360.0
            if minimum is not None and value < float(minimum):
                continue
            if maximum is not None and value > float(maximum):
                continue
            try:
                return _validate_parameter(name, value)
            except ValueError:
                continue
        raise RuntimeError(
            f"could not sample a valid value for normal parameter {name!r}"
        )

    raise ValueError(
        f"parameter {name!r} has unsupported distribution {distribution!r}; "
        "use fixed, normal, gaussian, standard, or sweep"
    )


def _parameter_plan(
    definitions: Mapping[str, Any],
    number_of_simulations: int,
    random_seed: int | None,
) -> list[dict[str, float]]:
    unknown = set(definitions).difference(SIMULATION_PARAMETER_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown simulation parameters: {sorted(unknown)}")
    complete_definitions = {**SIMULATION_PARAMETER_DEFAULTS, **definitions}
    random_generator = random.Random(random_seed)
    return [
        {
            name: _sample_parameter(
                name,
                definition,
                simulation_index,
                number_of_simulations,
                random_generator,
            )
            for name, definition in complete_definitions.items()
        }
        for simulation_index in range(number_of_simulations)
    ]


def _optional_float(source: Any, attribute: str) -> float | None:
    value = getattr(source, attribute, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _motor_initial_mass(motor: Any) -> float:
    total_mass = getattr(motor, "total_mass", None)
    if callable(total_mass):
        return float(total_mass(0))
    return float(motor.dry_mass) + float(motor.propellant_initial_mass)


def _scale_motor_thrust(motor: Any, scalar: float) -> None:
    """Scale every tabulated thrust sample while retaining its time coordinate."""

    source = motor.thrust.source
    if callable(source):
        raise ValueError("impulse variation requires tabulated thrust curves")
    motor.thrust.set_source(
        [[float(point[0]), float(point[1]) * scalar] for point in source]
    )


def _tabulated_motor_impulse(motor: Any) -> float:
    """Integrate the motor's current tabulated thrust curve in N s."""

    source = motor.thrust.source
    if callable(source):
        raise ValueError("impulse auditing requires tabulated thrust curves")
    points = [(float(point[0]), float(point[1])) for point in source]
    return sum(
        (time_1 - time_0) * (thrust_0 + thrust_1) / 2.0
        for (time_0, thrust_0), (time_1, thrust_1) in zip(points, points[1:])
    )


def _apply_stage_ratio_variations(
    full_stack: Rocket,
    sustainer: Rocket,
    impulse_ratio_percent: float,
    mass_ratio_percent: float,
    booster_wet_mass_percent: float = 100.0,
    sustainer_wet_mass_percent: float = 100.0,
) -> dict[str, float]:
    """Apply impulse and stage-mass scaling and return realized properties.

    The legacy mass-ratio input redistributes a fixed total stack wet mass.
    The two stage-specific inputs instead scale each nominal stage wet mass
    independently, allowing the total stack mass to change.
    """

    booster_motor = full_stack.motor
    sustainer_motor = sustainer.motor
    nominal_booster_impulse = _tabulated_motor_impulse(booster_motor)
    nominal_sustainer_impulse = _tabulated_motor_impulse(sustainer_motor)
    booster_motor_mass = _motor_initial_mass(booster_motor)
    sustainer_motor_mass = _motor_initial_mass(sustainer_motor)
    nominal_sustainer_wet_mass = float(sustainer.mass) + sustainer_motor_mass
    nominal_full_stack_wet_mass = float(full_stack.mass) + booster_motor_mass
    nominal_booster_wet_mass = (
        nominal_full_stack_wet_mass - nominal_sustainer_wet_mass
    )
    if nominal_booster_wet_mass <= 0:
        raise ValueError("full-stack mass must exceed sustainer wet mass")

    if impulse_ratio_percent != 100.0:
        if nominal_booster_impulse <= 0 or nominal_sustainer_impulse <= 0:
            raise ValueError("impulse variation requires two powered stages")
        impulse_scalar = impulse_ratio_percent / 100.0
        _scale_motor_thrust(booster_motor, impulse_scalar)
        _scale_motor_thrust(sustainer_motor, impulse_scalar)

    independent_mass_scaling = (
        booster_wet_mass_percent != 100.0
        or sustainer_wet_mass_percent != 100.0
    )
    if mass_ratio_percent != 100.0 and independent_mass_scaling:
        raise ValueError(
            "sustainer_booster_mass_ratio_percent cannot be varied together with "
            "booster_wet_mass_percent or sustainer_wet_mass_percent"
        )

    if mass_ratio_percent != 100.0:
        mass_ratio = (
            nominal_sustainer_wet_mass
            / nominal_booster_wet_mass
            * mass_ratio_percent
            / 100.0
        )
        varied_booster_wet_mass = nominal_full_stack_wet_mass / (1.0 + mass_ratio)
        varied_sustainer_wet_mass = (
            nominal_full_stack_wet_mass - varied_booster_wet_mass
        )
    else:
        varied_booster_wet_mass = (
            nominal_booster_wet_mass * booster_wet_mass_percent / 100.0
        )
        varied_sustainer_wet_mass = (
            nominal_sustainer_wet_mass * sustainer_wet_mass_percent / 100.0
        )

    if mass_ratio_percent != 100.0 or independent_mass_scaling:
        booster_dry_mass = varied_booster_wet_mass - booster_motor_mass
        sustainer_dry_mass = varied_sustainer_wet_mass - sustainer_motor_mass
        if booster_dry_mass <= 0 or sustainer_dry_mass <= 0:
            raise ValueError(
                "mass-ratio variation leaves a stage with no dry mass; narrow its range"
            )
        sustainer.mass = sustainer_dry_mass
        # A full-stack Rocket excludes its active booster motor but includes
        # the complete sustainer, including the sustainer motor.
        full_stack.mass = booster_dry_mass + varied_sustainer_wet_mass

    realized_booster_impulse = _tabulated_motor_impulse(booster_motor)
    realized_sustainer_impulse = _tabulated_motor_impulse(sustainer_motor)
    realized_sustainer_wet_mass = float(sustainer.mass) + sustainer_motor_mass
    realized_full_stack_wet_mass = float(full_stack.mass) + booster_motor_mass
    realized_booster_wet_mass = (
        realized_full_stack_wet_mass - realized_sustainer_wet_mass
    )
    return {
        "booster_impulse_n_s": realized_booster_impulse,
        "sustainer_impulse_n_s": realized_sustainer_impulse,
        "total_impulse_n_s": realized_booster_impulse + realized_sustainer_impulse,
        "nominal_total_impulse_n_s": (
            nominal_booster_impulse + nominal_sustainer_impulse
        ),
        "impulse_scalar": impulse_ratio_percent / 100.0,
        "booster_wet_mass_kg": realized_booster_wet_mass,
        "sustainer_wet_mass_kg": realized_sustainer_wet_mass,
        "total_wet_mass_kg": realized_full_stack_wet_mass,
        "nominal_booster_wet_mass_kg": nominal_booster_wet_mass,
        "nominal_sustainer_wet_mass_kg": nominal_sustainer_wet_mass,
        "nominal_total_wet_mass_kg": nominal_full_stack_wet_mass,
        "booster_wet_mass_scalar": (
            realized_booster_wet_mass / nominal_booster_wet_mass
        ),
        "sustainer_wet_mass_scalar": (
            realized_sustainer_wet_mass / nominal_sustainer_wet_mass
        ),
    }


def _flight_record(
    flight: Flight,
    *,
    simulation_index: int,
    rocket_index: int,
    launch_angle: float,
    solution: Sequence[Sequence[float]] | None = None,
    rocket_name: str | None = None,
) -> dict[str, Any]:
    return {
        "simulation_index": simulation_index,
        "rocket_index": rocket_index,
        "rocket_name": rocket_name
        or getattr(flight.rocket, "name", f"rocket_{rocket_index}"),
        "flight_name": flight.name,
        "launch_angle": launch_angle,
        "solution": [list(state) for state in (solution or flight.solution)],
        "events": {
            "out_of_rail_time": _optional_float(flight, "out_of_rail_time"),
            "apogee_time": _optional_float(flight, "apogee_time"),
            "apogee": _optional_float(flight, "apogee"),
            "impact_time": _optional_float(flight, "impact_time"),
            "x_impact": _optional_float(flight, "x_impact"),
            "y_impact": _optional_float(flight, "y_impact"),
            "impact_velocity": _optional_float(flight, "impact_velocity"),
        },
    }


def _environment_from_wind_plan(
    plan: Mapping[str, Any], simulation_index: int
) -> Environment:
    """Construct one RocketPy environment from a compact sampled-wind plan."""

    height = np.asarray(plan["height_m"], dtype=float)
    wind_u_values = np.asarray(plan["wind_u_m_s"], dtype=float)[simulation_index]
    wind_v_values = np.asarray(plan["wind_v_m_s"], dtype=float)[simulation_index]
    environment = Environment(
        latitude=float(plan["latitude"]),
        longitude=float(plan["longitude"]),
        elevation=float(plan["elevation_m"]),
        timezone=str(plan["timezone"]),
        date=tuple(plan["date_tuple"]),
        max_expected_height=float(plan["max_expected_height_m"]),
    )
    environment.set_atmospheric_model(
        type="custom_atmosphere",
        pressure=np.asarray(plan["pressure"], dtype=float),
        temperature=np.asarray(plan["temperature"], dtype=float),
        wind_u=np.column_stack((height, wind_u_values)),
        wind_v=np.column_stack((height, wind_v_values)),
    )
    return environment


def _merge_phase_solutions(flights: Sequence[Flight]) -> list[list[float]]:
    solution: list[list[float]] = []
    for flight in flights:
        for state in flight.solution:
            state_copy = list(state)
            if solution and abs(state_copy[0] - solution[-1][0]) <= 1e-9:
                solution[-1] = state_copy
            elif not solution or state_copy[0] > solution[-1][0]:
                solution.append(state_copy)
    return solution


def _validate_powered_flight_motion(flight: Flight, max_time_step: float) -> None:
    total_impulse = _optional_float(flight.rocket.motor, "total_impulse") or 0.0
    if total_impulse <= 0:
        return

    maximum_speed = max(
        math.sqrt(state[4] ** 2 + state[5] ** 2 + state[6] ** 2)
        for state in flight.solution
    )
    maximum_altitude_gain = max(state[3] for state in flight.solution) - float(
        flight.env.elevation
    )
    if maximum_speed < 1.0 and maximum_altitude_gain < 1.0:
        raise RuntimeError(
            f"{flight.name} contains a powered motor but never left the pad. "
            "The numerical integrator likely skipped the thrust curve; reduce "
            f"max_time_step below its current value of {max_time_step:g} s."
        )


def load_monte_carlo_output(path: str | Path) -> dict[str, Any]:
    """Load either a legacy or streamed Monte Carlo pickle.

    Streamed version-2 files store the metadata first and each flight as a
    subsequent pickle object. Loading reconstructs the legacy ``flights``
    list for analysis tools; simulation generation itself never holds that
    complete list in memory.
    """

    input_path = Path(path).expanduser().resolve()
    with input_path.open("rb") as pickle_file:
        payload = pickle.load(pickle_file)
        if not isinstance(payload, dict):
            raise TypeError("Monte Carlo pickle metadata must be a dictionary")
        if payload.get("storage") == "pickle_stream":
            flight_count = int(payload.get("flight_record_count", 0))
            payload["flights"] = [
                pickle.load(pickle_file) for _ in range(flight_count)
            ]
    return payload


def run_monte_carlo(
    rockets: Rocket | Sequence[Rocket],
    environment: Environment,
    *,
    launch_angle: float,
    launch_angle_std_dev: float,
    number_of_simulations: int,
    output_path: str | Path = "flight_data.pkl",
    rail_length: float = 6.0,
    heading: float = 0.0,
    heading_std_dev: float = 0.0,
    max_time: float = 600.0,
    max_time_step: float = 0.5,
    coast_period: float = 5.0,
    random_seed: int | None = None,
    parameter_variations: Mapping[str, Any] | None = None,
    workers: int = 1,
    native_threads_per_worker: int = 1,
    environment_variations: Sequence[Environment] | None = None,
    environment_profile_variations: Mapping[str, Any] | None = None,
    environment_sample_metadata: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run a launch-angle Monte Carlo sweep and write its data to a pickle.

    ``launch_angle`` is RocketPy's rail inclination in degrees above the
    horizontal: 90 degrees is vertical. One angle is sampled for each Monte
    Carlo iteration. A recognized ``full_stack`` and ``sustainer`` pair runs
    as one multistage flight; other rockets run independently.

    The returned dictionary contains the streamed file metadata. Use
    :func:`load_monte_carlo_output` when all full 14-state flight solutions
    need to be materialized for analysis.

    ``workers`` controls the number of concurrent simulation processes. Each
    process operates on private rocket copies, and results are stored in their
    original simulation-index order.
    """
    rocket_tuple = _as_rocket_tuple(rockets)
    multistage_pair = _multistage_pair(rocket_tuple)
    if not isinstance(environment, Environment):
        raise TypeError("environment must be a RocketPy Environment object")
    if isinstance(number_of_simulations, bool) or not isinstance(
        number_of_simulations, int
    ):
        raise TypeError("number_of_simulations must be an integer")
    if number_of_simulations <= 0:
        raise ValueError("number_of_simulations must be positive")
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise TypeError("workers must be an integer")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if isinstance(native_threads_per_worker, bool) or not isinstance(
        native_threads_per_worker, int
    ):
        raise TypeError("native_threads_per_worker must be an integer")
    if native_threads_per_worker <= 0:
        raise ValueError("native_threads_per_worker must be positive")
    worker_count = min(workers, number_of_simulations)

    if environment_variations is not None and environment_profile_variations is not None:
        raise ValueError(
            "use either environment_variations or environment_profile_variations, not both"
        )
    if environment_profile_variations is not None:
        wind_u_matrix = np.asarray(
            environment_profile_variations.get("wind_u_m_s"), dtype=float
        )
        wind_v_matrix = np.asarray(
            environment_profile_variations.get("wind_v_m_s"), dtype=float
        )
        height_vector = np.asarray(
            environment_profile_variations.get("height_m"), dtype=float
        )
        expected_shape = (number_of_simulations, len(height_vector))
        if wind_u_matrix.shape != expected_shape or wind_v_matrix.shape != expected_shape:
            raise ValueError(
                "environment_profile_variations wind matrices must have shape "
                f"{expected_shape}"
            )
        # Workers receive only the compact numeric plan and build their own
        # assigned Environment on demand. Do not capture even the base
        # RocketPy Environment in the serialized worker closure.
        case_environments: tuple[Environment, ...] = ()
    elif environment_variations is None:
        case_environments = (environment,) * number_of_simulations
    else:
        case_environments = tuple(environment_variations)
        if len(case_environments) != number_of_simulations:
            raise ValueError(
                "environment_variations must contain one Environment per simulation"
            )
        if not all(isinstance(item, Environment) for item in case_environments):
            raise TypeError("every environment variation must be an Environment")
    if environment_sample_metadata is None:
        case_environment_metadata: tuple[Mapping[str, Any] | None, ...] = (
            (None,) * number_of_simulations
        )
    else:
        case_environment_metadata = tuple(environment_sample_metadata)
        if len(case_environment_metadata) != number_of_simulations:
            raise ValueError(
                "environment_sample_metadata must contain one entry per simulation"
            )

    launch_angle = float(launch_angle)
    launch_angle_std_dev = float(launch_angle_std_dev)
    heading = float(heading)
    heading_std_dev = float(heading_std_dev)
    if launch_angle_std_dev < 0 or heading_std_dev < 0:
        raise ValueError("launch_angle_std_dev and heading_std_dev cannot be negative")

    parameter_definitions: dict[str, Any] = {
        "launch_angle": {
            "distribution": "gaussian",
            "mean": launch_angle,
            "std_dev": launch_angle_std_dev,
        },
        "heading": {
            "distribution": "gaussian",
            "mean": heading,
            "std_dev": heading_std_dev,
        },
        "rail_length": rail_length,
        "max_time": max_time,
        "max_time_step": max_time_step,
        "coast_period": coast_period,
        "rtol": SIMULATION_PARAMETER_DEFAULTS["rtol"],
        "atol": SIMULATION_PARAMETER_DEFAULTS["atol"],
        "sustainer_ignition_max_tilt_deg": SIMULATION_PARAMETER_DEFAULTS[
            "sustainer_ignition_max_tilt_deg"
        ],
    }
    if parameter_variations is not None:
        parameter_definitions.update(parameter_variations)
    sampled_parameter_sets = _parameter_plan(
        parameter_definitions,
        number_of_simulations,
        random_seed,
    )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.stem}_parts_", dir=output_path.parent
        )
    )

    def run_case(
        simulation_index: int, parameters: Mapping[str, float]
    ) -> tuple[Path, int]:
        simulation_start = time.perf_counter()
        print(
            f"Starting simulation {simulation_index + 1}/"
            f"{number_of_simulations}",
            flush=True,
        )
        sampled_angle = parameters["launch_angle"]
        sampled_heading = parameters["heading"]
        sampled_rail_length = parameters["rail_length"]
        sampled_max_time = parameters["max_time"]
        sampled_max_time_step = parameters["max_time_step"]
        sampled_coast_period = parameters["coast_period"]
        sampled_rtol = parameters["rtol"]
        sampled_atol = parameters["atol"]
        sampled_ignition_max_tilt = parameters[
            "sustainer_ignition_max_tilt_deg"
        ]
        case_environment = (
            _environment_from_wind_plan(
                environment_profile_variations, simulation_index
            )
            if environment_profile_variations is not None
            else case_environments[simulation_index]
        )
        weather_sample = case_environment_metadata[simulation_index]

        if multistage_pair is not None:
            # RocketPy builds interpolation caches during a Flight. Give each
            # integration isolated rocket objects so repeated cases within a
            # worker never retain mutable state from the previous case.
            full_stack, sustainer = (deepcopy(rocket) for rocket in multistage_pair)
            stage_properties = _apply_stage_ratio_variations(
                full_stack,
                sustainer,
                parameters["sustainer_booster_impulse_ratio_percent"],
                parameters["sustainer_booster_mass_ratio_percent"],
                parameters["booster_wet_mass_percent"],
                parameters["sustainer_wet_mass_percent"],
            )
            stage_impulses = {
                name: value
                for name, value in stage_properties.items()
                if "impulse" in name
            }
            stage_masses = {
                name: value
                for name, value in stage_properties.items()
                if "mass" in name
            }
            if sampled_angle <= 0:
                raise ValueError(
                    "a multistage launch angle must be greater than 0 degrees"
                )
            result = runfullstacksim(
                full_stack,
                sustainer,
                case_environment,
                sampled_max_time,
                coast_period=sampled_coast_period,
                rail_length=sampled_rail_length,
                rod_angle=90.0 - sampled_angle,
                heading=sampled_heading,
                max_time_step=sampled_max_time_step,
                rtol=sampled_rtol,
                atol=sampled_atol,
                sustainer_ignition_max_tilt_deg=sampled_ignition_max_tilt,
                return_details=True,
            )
            phase_flights = result.flights
            if len(phase_flights) < 2:
                raise RuntimeError(
                    f"multistage simulation {simulation_index + 1} did not "
                    "complete staging"
                )
            final_flight = phase_flights[-1]
            _validate_powered_flight_motion(final_flight, sampled_max_time_step)
            record = _flight_record(
                final_flight,
                simulation_index=simulation_index,
                rocket_index=0,
                launch_angle=sampled_angle,
                solution=_merge_phase_solutions(phase_flights),
                rocket_name=f"{full_stack.name} -> {sustainer.name}",
            )
            record.update(
                {
                    "flight_type": "multistage",
                    "heading": sampled_heading,
                    "simulation_parameters": dict(parameters),
                    "stage_impulses": stage_impulses,
                    "stage_masses": stage_masses,
                    "weather_sample": dict(weather_sample) if weather_sample else None,
                    "ignition_time_s": result.ignition_time,
                    "ignition_angle_deg_from_vertical": result.staging_tilt,
                    "sustainer_ignited": result.sustainer_ignited,
                    "tilt_lockout_triggered": result.tilt_lockout_triggered,
                    "tilt_lockout_threshold_deg": sampled_ignition_max_tilt,
                    "phases": [
                        {
                            "name": phase.name,
                            "start_time_s": float(phase.solution[0][0]),
                            "end_time_s": float(phase.solution[-1][0]),
                            "state_count": len(phase.solution),
                        }
                        for phase in phase_flights
                    ],
                }
            )
            case_records = [record]
            shard_path = shard_directory / f"simulation_{simulation_index:08d}.pkl"
            with shard_path.open("wb") as shard_file:
                pickle.dump(case_records, shard_file, protocol=pickle.HIGHEST_PROTOCOL)
            print(
                f"Completed simulation {simulation_index + 1}/"
                f"{number_of_simulations} in "
                f"{time.perf_counter() - simulation_start:.2f} s",
                flush=True,
            )
            return shard_path, len(case_records)

        case_records: list[dict[str, Any]] = []
        for rocket_index, rocket in enumerate(rocket_tuple):
            worker_rocket = deepcopy(rocket)
            flight = Flight(
                rocket=worker_rocket,
                environment=case_environment,
                rail_length=sampled_rail_length,
                inclination=sampled_angle,
                heading=sampled_heading,
                max_time=sampled_max_time,
                max_time_step=sampled_max_time_step,
                rtol=sampled_rtol,
                atol=sampled_atol,
                time_overshoot=True,
                name=(
                    f"Monte Carlo {simulation_index + 1}: "
                    f"{getattr(rocket, 'name', f'rocket_{rocket_index}')}"
                ),
            )
            _validate_powered_flight_motion(flight, sampled_max_time_step)
            record = _flight_record(
                flight,
                simulation_index=simulation_index,
                rocket_index=rocket_index,
                launch_angle=sampled_angle,
            )
            record.update(
                {
                    "flight_type": "independent",
                    "heading": sampled_heading,
                    "simulation_parameters": dict(parameters),
                    "weather_sample": dict(weather_sample) if weather_sample else None,
                }
            )
            case_records.append(record)

        shard_path = shard_directory / f"simulation_{simulation_index:08d}.pkl"
        with shard_path.open("wb") as shard_file:
            pickle.dump(case_records, shard_file, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"Completed simulation {simulation_index + 1}/"
            f"{number_of_simulations} in "
            f"{time.perf_counter() - simulation_start:.2f} s",
            flush=True,
        )
        return shard_path, len(case_records)

    indexed_parameters = list(enumerate(sampled_parameter_sets))
    if worker_count == 1:
        case_results = [
            run_case(index, parameters) for index, parameters in indexed_parameters
        ]
    else:
        if dill is None:
            raise RuntimeError(
                "workers greater than 1 require dill; install it with "
                "'python -m pip install dill'"
            )
        print(
            f"Running {number_of_simulations} simulations with "
            f"{worker_count} worker processes and "
            f"{native_threads_per_worker} native thread(s) per worker",
            flush=True,
        )
        logical_cpu_count = os.process_cpu_count() or os.cpu_count() or 1
        if worker_count > max(1, logical_cpu_count // 2):
            print(
                f"Note: {worker_count} workers are competing for "
                f"{logical_cpu_count} logical CPUs; per-simulation time may "
                "increase once physical cores are occupied.",
                flush=True,
            )
        # Write the large RocketPy/environment context once. Passing the blob
        # through ProcessPoolExecutor initargs makes Windows copy it to each
        # child serially, which can stagger worker startup by minutes.
        worker_context_path = shard_directory / "worker_context.dill"
        context_start = time.perf_counter()
        print("Serializing compact worker context...", flush=True)
        with worker_context_path.open("wb") as context_file:
            dill.dump(run_case, context_file, recurse=True)
        print(
            f"Worker context ready: {worker_context_path.stat().st_size / 1_048_576:.1f} MiB "
            f"in {time.perf_counter() - context_start:.2f} s. Starting workers...",
            flush=True,
        )
        process_context = get_context("spawn")
        startup_barrier = process_context.Barrier(worker_count)
        with _native_thread_limit(native_threads_per_worker):
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=process_context,
                initializer=_initialize_process_worker,
                initargs=(str(worker_context_path), startup_barrier),
            ) as executor:
                futures = [
                    executor.submit(_execute_process_case, index, parameters)
                    for index, parameters in indexed_parameters
                ]
                # Preserve simulation ordering in the output even when workers
                # complete in a different order.
                case_results = [future.result() for future in futures]
        worker_context_path.unlink()

    flight_record_count = sum(record_count for _, record_count in case_results)
    payload: dict[str, Any] = {
        "format": "projectblaze.rocketpy.monte_carlo",
        "format_version": 2,
        "storage": "pickle_stream",
        "flight_record_count": flight_record_count,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "state_columns": STATE_COLUMNS,
        "configuration": {
            "number_of_simulations": number_of_simulations,
            "workers": worker_count,
            "native_threads_per_worker": native_threads_per_worker,
            "number_of_rockets": len(rocket_tuple),
            "launch_angle": launch_angle,
            "launch_angle_std_dev": launch_angle_std_dev,
            "heading": heading,
            "heading_std_dev": heading_std_dev,
            "parameter_definitions": parameter_definitions,
            "sampled_parameters": sampled_parameter_sets,
            "sampled_launch_angles": [
                parameters["launch_angle"] for parameters in sampled_parameter_sets
            ],
            "sampled_headings": [
                parameters["heading"] for parameters in sampled_parameter_sets
            ],
            "simulation_mode": (
                "multistage" if multistage_pair is not None else "independent"
            ),
            "random_seed": random_seed,
            "weather_sampling_enabled": (
                environment_variations is not None
                or environment_profile_variations is not None
            ),
            "weather_samples": [
                dict(item) if item is not None else None
                for item in case_environment_metadata
            ],
        },
        "environment": {
            "latitude": _optional_float(environment, "latitude"),
            "longitude": _optional_float(environment, "longitude"),
            "elevation": _optional_float(environment, "elevation"),
            "timezone": str(getattr(environment, "timezone", "")),
            "date": str(getattr(environment, "date", "")),
        },
    }

    with output_path.open("wb") as pickle_file:
        pickle.dump(payload, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        # Merge one completed simulation at a time. This keeps memory bounded
        # by active workers plus one shard instead of all completed flights.
        for shard_path, _ in case_results:
            with shard_path.open("rb") as shard_file:
                records = pickle.load(shard_file)
            for record in records:
                pickle.dump(record, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
            shard_path.unlink()
    shard_directory.rmdir()

    return payload


def _daily_sample_hours(calls_per_day: int) -> tuple[int, ...]:
    if isinstance(calls_per_day, bool) or not isinstance(calls_per_day, int):
        raise ValueError("environment.wind_sampling.calls_per_day must be an integer")
    if not 1 <= calls_per_day <= 24:
        raise ValueError(
            "environment.wind_sampling.calls_per_day must be in the range [1, 24]"
        )
    return tuple((index * 24) // calls_per_day for index in range(calls_per_day))


def _build_openmeteo_wind_ensemble(
    *,
    create_environment: Any,
    latitude: float,
    longitude: float,
    launch_date: str,
    launch_time: str,
    timezone_name: str,
    elevation_m: float,
    model: str | None,
    endpoint: str,
    max_expected_height_m: float,
    extend_above_model_top: bool,
    calls_per_day: int,
    days_either_side: int,
    time_of_day_std_hours: float,
    number_of_simulations: int,
    random_seed: int | None,
    profile_source_label: str = "Open-Meteo",
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    """Fetch a time/day weather space and sample correlated Gaussian winds."""

    if (
        isinstance(days_either_side, bool)
        or not isinstance(days_either_side, int)
        or days_either_side < 0
    ):
        raise ValueError(
            "environment.wind_sampling.days_either_side must be a non-negative integer"
        )
    time_of_day_std_hours = float(time_of_day_std_hours)
    if not math.isfinite(time_of_day_std_hours) or time_of_day_std_hours < 0:
        raise ValueError(
            "environment.wind_sampling.time_of_day_std_hours must be non-negative"
        )

    sample_hours = _daily_sample_hours(calls_per_day)
    launch_hour = datetime.fromisoformat(f"{launch_date}T{launch_time}")
    dates = [
        launch_hour.date() + timedelta(days=offset)
        for offset in range(-days_either_side, days_either_side + 1)
    ]
    requests = [(date, hour) for date in dates for hour in sample_hours]
    total_requests = len(requests)
    profiles_by_hour: dict[int, list[Any]] = {hour: [] for hour in sample_hours}
    dated_results: list[tuple[datetime, Any]] = []

    print(
        f"Sampling {profile_source_label} wind space: {total_requests} profiles "
        f"({calls_per_day}/day across {len(dates)} days)",
        flush=True,
    )
    for request_index, (sample_date, sample_hour) in enumerate(requests, start=1):
        date_text = sample_date.isoformat()
        time_text = f"{sample_hour:02d}:00"
        print(
            f"{profile_source_label} profile {request_index}/{total_requests}: "
            f"{date_text} {time_text} {timezone_name}",
            flush=True,
        )
        result = create_environment(
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
        profiles_by_hour[sample_hour].append(result)
        dated_results.append(
            (datetime.combine(sample_date, datetime.min.time()).replace(hour=sample_hour), result)
        )
        print(
            f"Completed {profile_source_label} profile {request_index}/{total_requests}",
            flush=True,
        )

    def circular_hour_distance(left: float, right: float) -> float:
        difference = abs(left - right) % 24.0
        return min(difference, 24.0 - difference)

    requested_hour_value = launch_hour.hour + launch_hour.minute / 60.0
    baseline_datetime, baseline_result = min(
        dated_results,
        key=lambda item: (
            abs((item[0].date() - launch_hour.date()).days),
            circular_hour_distance(item[0].hour, requested_hour_value),
        ),
    )
    height_grid = np.union1d(
        np.asarray(baseline_result.wind_u, dtype=float)[:, 0],
        np.asarray(baseline_result.wind_v, dtype=float)[:, 0],
    )
    ensembles: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sample_hour, results in profiles_by_hour.items():
        rows = []
        for result in results:
            wind_u = np.asarray(result.wind_u, dtype=float)
            wind_v = np.asarray(result.wind_v, dtype=float)
            rows.append(
                np.concatenate(
                    (
                        np.interp(height_grid, wind_u[:, 0], wind_u[:, 1]),
                        np.interp(height_grid, wind_v[:, 0], wind_v[:, 1]),
                    )
                )
            )
        matrix = np.asarray(rows, dtype=float)
        ensembles[sample_hour] = (matrix.mean(axis=0), matrix - matrix.mean(axis=0))

    random_generator = random.Random(random_seed)
    sampled_wind_u: list[np.ndarray] = []
    sampled_wind_v: list[np.ndarray] = []
    sample_metadata: list[dict[str, Any]] = []
    date_tuple = launch_hour.timetuple()[:4]
    for simulation_index in range(number_of_simulations):
        sampled_time = (
            requested_hour_value
            if time_of_day_std_hours == 0
            else random_generator.gauss(requested_hour_value, time_of_day_std_hours)
        ) % 24.0
        condition_hour = min(
            sample_hours,
            key=lambda hour: circular_hour_distance(hour, sampled_time),
        )
        mean_vector, centered_profiles = ensembles[condition_hour]
        if len(centered_profiles) > 1:
            weights = np.asarray(
                [random_generator.gauss(0.0, 1.0) for _ in centered_profiles]
            )
            sampled_vector = mean_vector + (
                weights @ centered_profiles / math.sqrt(len(centered_profiles) - 1)
            )
        else:
            sampled_vector = mean_vector.copy()
        split = len(height_grid)
        sampled_wind_u.append(sampled_vector[:split])
        sampled_wind_v.append(sampled_vector[split:])
        sample_metadata.append(
            {
                "simulation_index": simulation_index,
                "sampled_time_of_day_hours": sampled_time,
                "condition_hour": condition_hour,
                "profiles_in_condition": len(centered_profiles),
                "calls_per_day": calls_per_day,
                "days_either_side": days_either_side,
                "time_of_day_std_hours": time_of_day_std_hours,
                "source_date_start": dates[0].isoformat(),
                "source_date_end": dates[-1].isoformat(),
                "baseline_date": baseline_datetime.date().isoformat(),
                "baseline_time": f"{baseline_datetime.hour:02d}:00",
            }
        )

    print(
        f"Built {number_of_simulations} Gaussian wind profiles from "
        f"{total_requests} completed {profile_source_label} profiles",
        flush=True,
    )
    wind_plan = {
        "latitude": latitude,
        "longitude": longitude,
        "elevation_m": float(baseline_result.environment.elevation),
        "timezone": timezone_name,
        "date_tuple": date_tuple,
        "max_expected_height_m": max_expected_height_m,
        "pressure": np.asarray(baseline_result.pressure, dtype=float),
        "temperature": np.asarray(baseline_result.temperature, dtype=float),
        "height_m": height_grid,
        "wind_u_m_s": np.asarray(sampled_wind_u, dtype=float),
        "wind_v_m_s": np.asarray(sampled_wind_v, dtype=float),
    }
    return wind_plan, sample_metadata, baseline_result


def _default_forecast_hour(timezone_name: str) -> tuple[str, str]:
    try:
        local_timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        if timezone_name != BLACK_ROCK_TIMEZONE:
            raise ValueError(
                f"cannot resolve IANA timezone {timezone_name!r}; supply "
                "--date and --time or install the tzdata package"
            ) from exc
        # Windows does not ship an IANA database. The project is normally run
        # in Pacific time, so preserve the notebook's host-local fallback for
        # the default Black Rock configuration.
        hour = datetime.now() + timedelta(hours=1)
    else:
        hour = datetime.now(local_timezone) + timedelta(hours=1)
    hour = hour.replace(minute=0, second=0, microsecond=0)
    return hour.strftime("%Y-%m-%d"), hour.strftime("%H:%M")


def load_monte_carlo_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read Monte Carlo config {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Monte Carlo configuration root must be a JSON object")
    if config.get("format") != "projectblaze.rocketpy.monte_carlo_config":
        raise ValueError("unsupported or missing Monte Carlo configuration format")
    if config.get("format_version") != 1:
        raise ValueError("only Monte Carlo configuration format_version 1 is supported")
    return config, config_path


def _config_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name, {})
    if not isinstance(section, Mapping):
        raise ValueError(f"configuration field {name!r} must be an object")
    return section


def _resolve_config_path(value: Any, base_directory: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"configuration field {field!r} must be a path string")
    path = Path(value).expanduser()
    return (base_directory / path).resolve() if not path.is_absolute() else path.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a RocketPy Monte Carlo simulation from one JSON config file."
    )
    parser.add_argument("config_file", type=Path, help="Monte Carlo JSON config")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the JSON, rockets, and parameter distributions without fetching weather or running flights",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from interpreter import RocketConfigurationError, load_rocket
    from openmeteo_environment import create_openmeteo_environment

    args = parse_args(argv)
    try:
        config, config_path = load_monte_carlo_config(args.config_file)
        base_directory = config_path.parent
        rocket_json = _resolve_config_path(
            config.get("rocket_json"), base_directory, "rocket_json"
        )
        rocket_keys = config.get("rockets", [None])
        if not isinstance(rocket_keys, list) or not rocket_keys:
            raise ValueError("configuration field 'rockets' must be a non-empty array")
        allow_incomplete = bool(config.get("allow_incomplete_rocket", False))
        rockets = [
            load_rocket(
                rocket_json,
                rocket_key=None if rocket_key is None else str(rocket_key),
                require_simulation_ready=not allow_incomplete,
            )
            for rocket_key in rocket_keys
        ]

        simulation = _config_section(config, "simulation")
        number_of_simulations = simulation.get("number_of_simulations", 100)
        if isinstance(number_of_simulations, bool) or not isinstance(
            number_of_simulations, int
        ):
            raise ValueError("simulation.number_of_simulations must be an integer")
        if number_of_simulations <= 0:
            raise ValueError("simulation.number_of_simulations must be positive")
        random_seed = simulation.get("random_seed")
        if random_seed is not None and not isinstance(random_seed, int):
            raise ValueError("simulation.random_seed must be an integer or null")
        workers = simulation.get("workers", 1)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
            raise ValueError("simulation.workers must be a positive integer")
        native_threads_per_worker = simulation.get("native_threads_per_worker", 1)
        if (
            isinstance(native_threads_per_worker, bool)
            or not isinstance(native_threads_per_worker, int)
            or native_threads_per_worker <= 0
        ):
            raise ValueError(
                "simulation.native_threads_per_worker must be a positive integer"
            )
        parameters = simulation.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("simulation.parameters must be an object")
        # Build the full plan here so --validate-only checks every distribution.
        _parameter_plan(parameters, number_of_simulations, random_seed)
        output_path = _resolve_config_path(
            simulation.get("output_path", "monte_carlo_flights.pkl"),
            base_directory,
            "simulation.output_path",
        )

        environment_config = _config_section(config, "environment")
        timezone_name = str(
            environment_config.get("timezone", BLACK_ROCK_TIMEZONE)
        )
        configured_date = environment_config.get("date")
        configured_time = environment_config.get("time")
        if (configured_date is None) != (configured_time is None):
            raise ValueError("environment.date and environment.time must be supplied together")
        forecast_date, forecast_time = (
            (str(configured_date), str(configured_time))
            if configured_date is not None
            else _default_forecast_hour(timezone_name)
        )
        wind_sampling = environment_config.get("wind_sampling", {})
        if not isinstance(wind_sampling, Mapping):
            raise ValueError("environment.wind_sampling must be an object")
        wind_sampling_enabled = bool(wind_sampling.get("enabled", False))
        calls_per_day = wind_sampling.get("calls_per_day", 8)
        _daily_sample_hours(calls_per_day)
        days_either_side = wind_sampling.get("days_either_side", 2)
        if (
            isinstance(days_either_side, bool)
            or not isinstance(days_either_side, int)
            or days_either_side < 0
        ):
            raise ValueError(
                "environment.wind_sampling.days_either_side must be a non-negative integer"
            )
        time_of_day_std_hours = float(
            wind_sampling.get("time_of_day_std_hours", 24.0 / calls_per_day)
        )
        if not math.isfinite(time_of_day_std_hours) or time_of_day_std_hours < 0:
            raise ValueError(
                "environment.wind_sampling.time_of_day_std_hours must be non-negative"
            )
        weather_call_count = calls_per_day * (2 * days_either_side + 1)
        configured_cache_file = wind_sampling.get("cache_file")
        configured_cache_path = (
            _resolve_config_path(
                configured_cache_file,
                base_directory,
                "environment.wind_sampling.cache_file",
            )
            if configured_cache_file is not None
            else None
        )

        if args.validate_only:
            mode = "multistage" if _multistage_pair(tuple(rockets)) else "independent"
            print(
                f"Valid Monte Carlo config: {config_path}\n"
                f"Rockets: {', '.join(str(rocket.name) for rocket in rockets)}\n"
                f"Mode: {mode}; simulations: {number_of_simulations}; "
                f"workers: {min(workers, number_of_simulations)}; "
                f"native threads/worker: {native_threads_per_worker}; "
                f"output: {output_path}\n"
                f"Wind sampling: {'enabled' if wind_sampling_enabled else 'disabled'}; "
                f"profiles required: {weather_call_count if wind_sampling_enabled else 1}; "
                f"source: {configured_cache_path if configured_cache_path else 'Open-Meteo'}"
            )
            return 0

        forecast_date, forecast_time = (
            str(forecast_date),
            str(forecast_time),
        )
        latitude = float(environment_config.get("latitude", BLACK_ROCK_LATITUDE))
        longitude = float(environment_config.get("longitude", BLACK_ROCK_LONGITUDE))
        elevation_m = float(
            environment_config.get("elevation_m", BLACK_ROCK_ELEVATION_M)
        )
        model = environment_config.get("model", "gfs_seamless")
        endpoint = str(environment_config.get("endpoint", "forecast"))
        max_expected_height_m = float(
            environment_config.get("max_expected_height_m", 80_000.0)
        )
        extend_above_model_top = bool(
            environment_config.get("extend_above_model_top", True)
        )
        wind_profile_plan = None
        weather_sample_metadata = None
        if wind_sampling_enabled:
            profile_provider = create_openmeteo_environment
            profile_source_label = "Open-Meteo"
            cache_file = wind_sampling.get("cache_file")
            if cache_file is not None:
                cache_path = _resolve_config_path(
                    cache_file,
                    base_directory,
                    "environment.wind_sampling.cache_file",
                )
                if not cache_path.is_file():
                    raise ValueError(
                        f"weather cache does not exist: {cache_path}. Run "
                        f"'python openmeteo_wind_cache.py {config_path}' first."
                    )
                from openmeteo_wind_cache import load_cached_environment_provider

                profile_provider = load_cached_environment_provider(cache_path)
                profile_source_label = f"weather cache {cache_path.name}"
                print(f"Loading wind profiles from {cache_path}", flush=True)
            (
                wind_profile_plan,
                weather_sample_metadata,
                openmeteo_result,
            ) = _build_openmeteo_wind_ensemble(
                create_environment=profile_provider,
                latitude=latitude,
                longitude=longitude,
                launch_date=forecast_date,
                launch_time=forecast_time,
                timezone_name=timezone_name,
                elevation_m=elevation_m,
                model=model,
                endpoint=endpoint,
                max_expected_height_m=max_expected_height_m,
                extend_above_model_top=extend_above_model_top,
                calls_per_day=calls_per_day,
                days_either_side=days_either_side,
                time_of_day_std_hours=time_of_day_std_hours,
                number_of_simulations=number_of_simulations,
                random_seed=random_seed,
                profile_source_label=profile_source_label,
            )
        else:
            print("Open-Meteo call 1/1", flush=True)
            openmeteo_result = create_openmeteo_environment(
                latitude=latitude,
                longitude=longitude,
                date=forecast_date,
                time=forecast_time,
                timezone_name=timezone_name,
                elevation_m=elevation_m,
                model=model,
                endpoint=endpoint,
                max_expected_height_m=max_expected_height_m,
                extend_above_model_top=extend_above_model_top,
            )
            print("Completed Open-Meteo call 1/1", flush=True)
        start_time = time.perf_counter()
        simulation_environment = (
            _environment_from_wind_plan(wind_profile_plan, 0)
            if wind_profile_plan is not None
            else openmeteo_result.environment
        )
        payload = run_monte_carlo(
            rockets,
            simulation_environment,
            launch_angle=SIMULATION_PARAMETER_DEFAULTS["launch_angle"],
            launch_angle_std_dev=0.0,
            heading=SIMULATION_PARAMETER_DEFAULTS["heading"],
            heading_std_dev=0.0,
            number_of_simulations=number_of_simulations,
            output_path=output_path,
            random_seed=random_seed,
            parameter_variations=parameters,
            workers=workers,
            native_threads_per_worker=native_threads_per_worker,
            environment_profile_variations=wind_profile_plan,
            environment_sample_metadata=weather_sample_metadata,
        )
    except (OSError, RocketConfigurationError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Saved {payload['flight_record_count']} flights from "
        f"{number_of_simulations} simulations to {output_path} in {time.perf_counter() - start_time:.2f} s"
    )
    print(
        f"Open-Meteo profile: {forecast_date} {forecast_time} {timezone_name}; "
        f"model top {openmeteo_result.model_top_height_m:.0f} m"
    )
    return 0


__all__ = [
    "SIMULATION_PARAMETER_DEFAULTS",
    "STATE_COLUMNS",
    "load_monte_carlo_output",
    "load_monte_carlo_config",
    "parse_args",
    "run_monte_carlo",
]


if __name__ == "__main__":
    raise SystemExit(main())
