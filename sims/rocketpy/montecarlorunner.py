"""Monte Carlo helpers for RocketPy flights.

The pickle produced here stores RocketPy's complete numerical flight solution
for every run. Live ``Flight`` objects are intentionally not pickled because
RocketPy environments contain interpolation callables that standard
``pickle`` cannot serialize reliably.

Standalone usage::

    python montecarlorunner.py montecarlo_config.json

Simulation parameters in the JSON may be fixed numbers, Gaussian objects with
``mean`` and ``std_dev``, or deterministic sweep objects with ``min`` and
``max``. Set ``simulation.workers`` to the number of launches to run
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
    "coast_period": 5.0,
    "sustainer_booster_impulse_ratio_percent": 100.0,
    "sustainer_booster_mass_ratio_percent": 100.0,
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
        "sustainer_booster_impulse_ratio_percent",
        "sustainer_booster_mass_ratio_percent",
    } and value <= 0:
        raise ValueError(f"{name} must be positive")
    if name == "coast_period" and value < 0:
        raise ValueError("coast_period cannot be negative")
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

    distribution = str(definition.get("distribution", "fixed")).lower()
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

    if distribution == "gaussian":
        if "mean" not in definition or "std_dev" not in definition:
            raise ValueError(
                f"gaussian parameter {name!r} requires 'mean' and 'std_dev'"
            )
        mean = float(definition["mean"])
        standard_deviation = float(definition["std_dev"])
        if not math.isfinite(mean) or not math.isfinite(standard_deviation):
            raise ValueError(f"gaussian settings for {name!r} must be finite")
        if standard_deviation < 0:
            raise ValueError(f"gaussian std_dev for {name!r} cannot be negative")
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
            f"could not sample a valid value for gaussian parameter {name!r}"
        )

    raise ValueError(
        f"parameter {name!r} has unsupported distribution {distribution!r}; "
        "use fixed, gaussian, or sweep"
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
        raise ValueError("impulse-ratio variation requires tabulated thrust curves")
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
) -> dict[str, float]:
    """Redistribute fixed totals to obtain requested sustainer/booster ratios."""

    booster_motor = full_stack.motor
    sustainer_motor = sustainer.motor
    nominal_booster_impulse = _tabulated_motor_impulse(booster_motor)
    nominal_sustainer_impulse = _tabulated_motor_impulse(sustainer_motor)

    if impulse_ratio_percent != 100.0:
        booster_impulse = nominal_booster_impulse
        sustainer_impulse = nominal_sustainer_impulse
        if booster_impulse <= 0 or sustainer_impulse <= 0:
            raise ValueError("impulse-ratio variation requires two powered stages")
        impulse_ratio = (
            sustainer_impulse / booster_impulse * impulse_ratio_percent / 100.0
        )
        total_impulse = booster_impulse + sustainer_impulse
        varied_booster_impulse = total_impulse / (1.0 + impulse_ratio)
        varied_sustainer_impulse = total_impulse - varied_booster_impulse
        _scale_motor_thrust(booster_motor, varied_booster_impulse / booster_impulse)
        _scale_motor_thrust(
            sustainer_motor, varied_sustainer_impulse / sustainer_impulse
        )

    if mass_ratio_percent != 100.0:
        booster_motor_mass = _motor_initial_mass(booster_motor)
        sustainer_motor_mass = _motor_initial_mass(sustainer_motor)
        sustainer_wet_mass = float(sustainer.mass) + sustainer_motor_mass
        full_stack_wet_mass = float(full_stack.mass) + booster_motor_mass
        booster_wet_mass = full_stack_wet_mass - sustainer_wet_mass
        if booster_wet_mass <= 0:
            raise ValueError("full-stack mass must exceed sustainer wet mass")

        mass_ratio = (
            sustainer_wet_mass / booster_wet_mass * mass_ratio_percent / 100.0
        )
        varied_booster_wet_mass = full_stack_wet_mass / (1.0 + mass_ratio)
        varied_sustainer_wet_mass = full_stack_wet_mass - varied_booster_wet_mass
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
    return {
        "booster_impulse_n_s": realized_booster_impulse,
        "sustainer_impulse_n_s": realized_sustainer_impulse,
        "total_impulse_n_s": realized_booster_impulse + realized_sustainer_impulse,
        "nominal_total_impulse_n_s": (
            nominal_booster_impulse + nominal_sustainer_impulse
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

        if multistage_pair is not None:
            # RocketPy builds interpolation caches during a Flight. Give each
            # integration isolated rocket objects so repeated cases within a
            # worker never retain mutable state from the previous case.
            full_stack, sustainer = (deepcopy(rocket) for rocket in multistage_pair)
            stage_impulses = _apply_stage_ratio_variations(
                full_stack,
                sustainer,
                parameters["sustainer_booster_impulse_ratio_percent"],
                parameters["sustainer_booster_mass_ratio_percent"],
            )
            if sampled_angle <= 0:
                raise ValueError(
                    "a multistage launch angle must be greater than 0 degrees"
                )
            result = runfullstacksim(
                full_stack,
                sustainer,
                environment,
                sampled_max_time,
                coast_period=sampled_coast_period,
                rail_length=sampled_rail_length,
                rod_angle=90.0 - sampled_angle,
                heading=sampled_heading,
                max_time_step=sampled_max_time_step,
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
                    "ignition_time_s": result.ignition_time,
                    "ignition_angle_deg_from_vertical": result.staging_tilt,
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
                environment=environment,
                rail_length=sampled_rail_length,
                inclination=sampled_angle,
                heading=sampled_heading,
                max_time=sampled_max_time,
                max_time_step=sampled_max_time_step,
                rtol=1e-4,
                atol=1e-6,
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
        with worker_context_path.open("wb") as context_file:
            dill.dump(run_case, context_file, recurse=True)
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

        if args.validate_only:
            mode = "multistage" if _multistage_pair(tuple(rockets)) else "independent"
            print(
                f"Valid Monte Carlo config: {config_path}\n"
                f"Rockets: {', '.join(str(rocket.name) for rocket in rockets)}\n"
                f"Mode: {mode}; simulations: {number_of_simulations}; "
                f"workers: {min(workers, number_of_simulations)}; "
                f"native threads/worker: {native_threads_per_worker}; "
                f"output: {output_path}"
            )
            return 0

        forecast_date, forecast_time = (
            str(forecast_date),
            str(forecast_time),
        )
        openmeteo_result = create_openmeteo_environment(
            latitude=float(environment_config.get("latitude", BLACK_ROCK_LATITUDE)),
            longitude=float(environment_config.get("longitude", BLACK_ROCK_LONGITUDE)),
            date=forecast_date,
            time=forecast_time,
            timezone_name=timezone_name,
            elevation_m=float(
                environment_config.get("elevation_m", BLACK_ROCK_ELEVATION_M)
            ),
            model=environment_config.get("model", "gfs_seamless"),
            endpoint=str(environment_config.get("endpoint", "forecast")),
            max_expected_height_m=float(
                environment_config.get("max_expected_height_m", 80_000.0)
            ),
            extend_above_model_top=bool(
                environment_config.get("extend_above_model_top", True)
            ),
        )
        start_time = time.perf_counter()
        payload = run_monte_carlo(
            rockets,
            openmeteo_result.environment,
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
