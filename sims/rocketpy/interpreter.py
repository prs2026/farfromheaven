"""Build RocketPy objects from the ``rocketpy-cdx1`` JSON format.

The CDX1 converter can recover a rocket's external geometry, but a CDX1 file
does not contain enough information for a physically meaningful RocketPy
flight.  In particular, the JSON must be supplemented with dry mass and
inertia, drag coefficients/curves, a motor, rail buttons, and (for recovery)
parachute ``cd_s`` values and triggers.  :func:`simulation_readiness_issues`
lists these omissions without preventing geometry-only inspection.

All dimensions used by this module are SI.  Paths in the JSON are resolved
relative to the JSON file, not to the process's current working directory.
Version 2 files may contain ``full_stack``, ``sustainer``, and ``booster``
entries; pass ``rocket_key`` to select one.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from rocketpy import GenericMotor, Motor, Rocket, SolidMotor
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "RocketPy is required. Install it with `python -m pip install rocketpy`."
    ) from exc


class RocketConfigurationError(ValueError):
    """Raised when JSON cannot be translated safely into RocketPy objects."""


_INERTIA_KEYS = ("I11", "I22", "I33", "I12", "I13", "I23")
_PATH_SUFFIXES = {".csv", ".eng", ".txt"}
POUND_PER_CUBIC_INCH_TO_KG_PER_CUBIC_METER = 0.45359237 / 0.0254**3
DEFAULT_PROPELLANT_DENSITY = 0.065 * POUND_PER_CUBIC_INCH_TO_KG_PER_CUBIC_METER


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    try:
        value = mapping[key]
    except KeyError as exc:
        raise RocketConfigurationError(f"Missing required field: {context}.{key}") from exc
    if value is None:
        raise RocketConfigurationError(f"Required field is null: {context}.{key}")
    return value


def _finite_number(value: Any, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise RocketConfigurationError(f"{context} must be a number, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RocketConfigurationError(f"{context} must be a number") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "a positive finite number" if positive else "a finite number"
        raise RocketConfigurationError(f"{context} must be {qualifier}")
    return result


def _inertia(value: Any, context: str) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        return tuple(_finite_number(value.get(key, 0), f"{context}.{key}") for key in _INERTIA_KEYS)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) in (3, 6):
        return tuple(_finite_number(item, context) for item in value)
    raise RocketConfigurationError(
        f"{context} must be an object with I11...I23 or an array of 3 or 6 numbers"
    )


def _source(value: Any, base_dir: Path, context: str) -> Any:
    """Resolve a Function-compatible file source while preserving arrays/scalars."""
    if isinstance(value, str):
        path = Path(value)
        if path.suffix.lower() in _PATH_SUFFIXES:
            path = path if path.is_absolute() else base_dir / path
            if not path.is_file():
                raise RocketConfigurationError(f"{context} file does not exist: {path}")
            return str(path.resolve())
    return value


def _normalized_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _aero_sources_from_csv(
    value: Any,
    base_dir: Path,
    *,
    alpha: float = 0.0,
) -> tuple[list[list[float]], list[list[float]], Path]:
    """Extract Mach/CD power-off and power-on sources from one aero CSV."""
    if not isinstance(value, str) or not value.strip():
        raise RocketConfigurationError("rocket.aero_curves must be a CSV file path")
    path = Path(value)
    path = path if path.is_absolute() else base_dir / path
    if not path.is_file():
        raise RocketConfigurationError(f"rocket.aero_curves file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise RocketConfigurationError(f"Aerodynamic CSV has no header: {path}")
            headers = {_normalized_header(header): header for header in reader.fieldnames}
            required = {
                "mach": "Mach",
                "cdpoweroff": "CD Power-Off",
                "cdpoweron": "CD Power-On",
            }
            missing = [label for key, label in required.items() if key not in headers]
            if missing:
                raise RocketConfigurationError(
                    f"Aerodynamic CSV {path} is missing header(s): {', '.join(missing)}"
                )

            mach_header = headers["mach"]
            off_header = headers["cdpoweroff"]
            on_header = headers["cdpoweron"]
            alpha_header = headers.get("alpha")
            points: dict[float, tuple[float, float]] = {}
            for line_number, row in enumerate(reader, start=2):
                try:
                    row_alpha = float(row[alpha_header]) if alpha_header else alpha
                    if not math.isclose(row_alpha, alpha, rel_tol=0, abs_tol=1e-9):
                        continue
                    mach = float(row[mach_header])
                    power_off = float(row[off_header])
                    power_on = float(row[on_header])
                except (TypeError, ValueError) as exc:
                    raise RocketConfigurationError(
                        f"Invalid aerodynamic number in {path} at line {line_number}"
                    ) from exc
                if not all(math.isfinite(item) for item in (mach, power_off, power_on)):
                    raise RocketConfigurationError(
                        f"Non-finite aerodynamic value in {path} at line {line_number}"
                    )
                if mach < 0 or power_off < 0 or power_on < 0:
                    raise RocketConfigurationError(
                        f"Mach and drag coefficients must be non-negative in {path} at line {line_number}"
                    )
                if mach in points and points[mach] != (power_off, power_on):
                    raise RocketConfigurationError(
                        f"Duplicate Mach {mach} with conflicting coefficients in {path}"
                    )
                points[mach] = (power_off, power_on)
    except OSError as exc:
        raise RocketConfigurationError(f"Could not read aerodynamic CSV {path}: {exc}") from exc

    if len(points) < 2:
        alpha_note = f" at Alpha={alpha:g}" if alpha_header else ""
        raise RocketConfigurationError(
            f"Aerodynamic CSV {path} needs at least two Mach rows{alpha_note}"
        )
    ordered = sorted(points.items())
    power_off_source = [[mach, coefficients[0]] for mach, coefficients in ordered]
    power_on_source = [[mach, coefficients[1]] for mach, coefficients in ordered]
    return power_off_source, power_on_source, path.resolve()


def _total_length(stages: Sequence[Mapping[str, Any]]) -> float:
    ends = []
    for index, stage in enumerate(stages):
        location = _finite_number(stage.get("location", 0), f"stages[{index}].location")
        length = _finite_number(stage.get("length", 0), f"stages[{index}].length")
        ends.append(location + length)
    return max(ends, default=0.0)


def _rocket_position(nose_position: float, total_length: float, orientation: str) -> float:
    """Convert a CDX1 nose-tip datum to the configured RocketPy coordinate."""
    return total_length - nose_position if orientation == "tail_to_nose" else nose_position


def available_rockets(data: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the selectable rocket keys in decoded converter output."""
    rockets = data.get("rockets")
    if rockets is None:
        return ()
    if not isinstance(rockets, Mapping):
        raise RocketConfigurationError("rockets must be a JSON object")
    return tuple(str(key) for key in rockets)


def _select_rocket_data(
    data: Mapping[str, Any], rocket_key: str | None
) -> dict[str, Any]:
    """Overlay one version 2 rocket configuration onto shared root data."""
    keys = available_rockets(data)
    if not keys:
        if rocket_key is not None:
            raise RocketConfigurationError(
                f"rocket_key={rocket_key!r} was supplied, but this file contains only one rocket"
            )
        return dict(data)

    selected_key = rocket_key or data.get("default_rocket")
    if selected_key is None:
        raise RocketConfigurationError(
            "Multiple rockets are available; specify rocket_key as one of: "
            + ", ".join(keys)
        )
    if selected_key not in keys:
        raise RocketConfigurationError(
            f"Unknown rocket_key {selected_key!r}; available rockets: {', '.join(keys)}"
        )

    variant = data["rockets"][selected_key]
    if not isinstance(variant, Mapping):
        raise RocketConfigurationError(f"rockets.{selected_key} must be a JSON object")
    selected = dict(data)
    selected["rocket"] = _require(variant, "rocket", f"rockets.{selected_key}")
    selected["stages"] = _require(variant, "stages", f"rockets.{selected_key}")
    selected["selected_rocket"] = selected_key
    selected["rocket_format"] = variant.get("format")
    selected["configuration_type"] = variant.get("configuration_type")
    selected["selected_simulation"] = variant.get("simulation", {})
    return selected


def _motor_from_eng(
    value: Any,
    base_dir: Path,
    geometry: Mapping[str, Any] | None = None,
) -> tuple[SolidMotor, float, Path]:
    """Build a grain-resolved SolidMotor using dimensions from a RASP header."""
    source = _source(value, base_dir, "rocket.thrust_curve")
    if not isinstance(source, str) or Path(source).suffix.lower() != ".eng":
        raise RocketConfigurationError(
            "rocket.thrust_curve must point to a RASP .eng file so motor mass can be read"
        )
    path = Path(source)
    try:
        _, raw_description, thrust_points = Motor.import_eng(str(path))
        description = [item for item in raw_description if item]
        if len(description) < 6:
            raise ValueError("header needs designation, diameter, length, delays, propellant mass, and total mass")
        diameter = float(description[1]) / 1000
        length = float(description[2]) / 1000
        propellant_mass = float(description[4])
        loaded_mass = float(description[5])
    except (OSError, TypeError, ValueError) as exc:
        raise RocketConfigurationError(f"Invalid RASP motor header in {path}: {exc}") from exc
    if not all(math.isfinite(item) and item > 0 for item in (diameter, length, propellant_mass, loaded_mass)):
        raise RocketConfigurationError(f"Motor dimensions and masses must be positive in {path}")
    if loaded_mass < propellant_mass:
        raise RocketConfigurationError(
            f"Loaded motor mass cannot be less than propellant mass in {path}"
        )

    geometry = geometry or {}
    motor_radius = diameter / 2
    grain_outer_radius = _finite_number(
        geometry.get("grain_outer_radius", motor_radius),
        "rocket.motor.grain_outer_radius",
        positive=True,
    )
    if grain_outer_radius > motor_radius:
        raise RocketConfigurationError(
            f"rocket.motor.grain_outer_radius ({grain_outer_radius:g} m) cannot exceed "
            f"the motor radius ({motor_radius:g} m) from {path}"
        )
    grain_density = _finite_number(
        geometry.get("grain_density", DEFAULT_PROPELLANT_DENSITY),
        "rocket.motor.grain_density",
        positive=True,
    )
    grain_number = int(geometry.get("grain_number", 1))
    if grain_number <= 0:
        raise RocketConfigurationError("rocket.motor.grain_number must be positive")
    grain_separation = _finite_number(
        geometry.get("grain_separation", 0), "rocket.motor.grain_separation"
    )
    if grain_separation < 0:
        raise RocketConfigurationError("rocket.motor.grain_separation cannot be negative")
    propellant_length = length - (grain_number - 1) * grain_separation
    if propellant_length <= 0:
        raise RocketConfigurationError(
            "Motor length from the RASP header must exceed total grain separation"
        )
    grain_height = propellant_length / grain_number

    dry_motor_mass = loaded_mass - propellant_mass
    # Some research .eng files set propellant mass equal to loaded mass. A
    # near-zero casing mass and inertia makes the flight solver effectively
    # singular at burnout. Preserve the header's loaded mass while reserving a
    # small, configurable portion of it as casing mass.
    minimum_dry_mass = _finite_number(
        geometry.get("minimum_dry_mass", min(0.01, loaded_mass * 0.01)),
        "rocket.motor.minimum_dry_mass",
        positive=True,
    )
    if minimum_dry_mass >= loaded_mass:
        raise RocketConfigurationError(
            "rocket.motor.minimum_dry_mass must be less than loaded motor mass"
        )
    modeled_dry_mass = max(dry_motor_mass, minimum_dry_mass)
    modeled_propellant_mass = loaded_mass - modeled_dry_mass
    propellant_volume = modeled_propellant_mass / grain_density
    inner_radius_squared = grain_outer_radius**2 - propellant_volume / (
        math.pi * propellant_length
    )
    if inner_radius_squared <= 0:
        raise RocketConfigurationError(
            f"The propellant mass in {path} cannot fit inside a {grain_outer_radius:g} m "
            f"outer radius and {propellant_length:g} m length at {grain_density:g} kg/m^3"
        )
    grain_inner_radius = math.sqrt(inner_radius_squared)
    transverse_inertia = modeled_dry_mass * (3 * motor_radius**2 + length**2) / 12
    axial_inertia = modeled_dry_mass * motor_radius**2 / 2
    # RocketPy prepends [0, 0] while importing RASP files. Some curve files
    # already contain a time-zero sample, so collapse duplicate times here.
    thrust_by_time = {float(time): float(thrust) for time, thrust in thrust_points}
    clean_thrust_points = [[time, thrust_by_time[time]] for time in sorted(thrust_by_time)]
    throat_radius = _finite_number(
        geometry.get("throat_radius", min(0.01, motor_radius / 2)),
        "rocket.motor.throat_radius",
        positive=True,
    )
    if throat_radius >= motor_radius:
        raise RocketConfigurationError(
            "rocket.motor.throat_radius must be smaller than the nozzle radius"
        )
    # Match RocketPy's plotted nozzle geometry: a 15-degree divergent section
    # followed by a 45-degree convergent section. The nozzle exit is the zero
    # (aft) datum, so the grain stack begins at the nozzle's forward face.
    nozzle_length = (motor_radius - throat_radius) * (
        1 / math.tan(math.radians(15)) + 1 / math.tan(math.radians(45))
    )
    grains_center = nozzle_length + propellant_length / 2
    motor = SolidMotor(
        thrust_source=clean_thrust_points,
        burn_time=(clean_thrust_points[0][0], clean_thrust_points[-1][0]),
        nozzle_radius=motor_radius,
        dry_mass=modeled_dry_mass,
        center_of_dry_mass_position=grains_center,
        dry_inertia=(transverse_inertia, transverse_inertia, axial_inertia),
        nozzle_position=0,
        grain_number=grain_number,
        grain_density=grain_density,
        grain_outer_radius=grain_outer_radius,
        grain_initial_inner_radius=grain_inner_radius,
        grain_initial_height=grain_height,
        grain_separation=grain_separation,
        grains_center_of_mass_position=grains_center,
        throat_radius=throat_radius,
    )
    motor.nozzle_length = nozzle_length
    motor.grain_aft_position = nozzle_length
    motor.motor_length = nozzle_length + propellant_length
    motor.propellant_length = propellant_length
    return motor, loaded_mass, path.resolve()


def _motor_from_json(config: Mapping[str, Any], base_dir: Path):
    motor_type = str(config.get("type", "generic")).strip().lower()
    common = {
        "thrust_source": _source(_require(config, "thrust_source", "motor"), base_dir, "motor.thrust_source"),
        "burn_time": config.get("burn_time"),
        "dry_mass": _finite_number(_require(config, "dry_mass", "motor"), "motor.dry_mass"),
        "dry_inertia": _inertia(_require(config, "dry_inertia", "motor"), "motor.dry_inertia"),
        "nozzle_radius": _finite_number(_require(config, "nozzle_radius", "motor"), "motor.nozzle_radius", positive=True),
        "center_of_dry_mass_position": _finite_number(
            _require(config, "center_of_dry_mass_position", "motor"),
            "motor.center_of_dry_mass_position",
        ),
        "nozzle_position": _finite_number(config.get("nozzle_position", 0), "motor.nozzle_position"),
        "coordinate_system_orientation": config.get(
            "coordinate_system_orientation", "nozzle_to_combustion_chamber"
        ),
        "interpolation_method": config.get("interpolation_method", "linear"),
        "reference_pressure": config.get("reference_pressure"),
    }
    # Let RocketPy infer burn time from point/file sources when it is omitted.
    if common["burn_time"] is None:
        common.pop("burn_time")

    if motor_type == "generic":
        return GenericMotor(
            **common,
            chamber_radius=_finite_number(
                _require(config, "chamber_radius", "motor"), "motor.chamber_radius", positive=True
            ),
            chamber_height=_finite_number(
                _require(config, "chamber_height", "motor"), "motor.chamber_height", positive=True
            ),
            chamber_position=_finite_number(
                _require(config, "chamber_position", "motor"), "motor.chamber_position"
            ),
            propellant_initial_mass=_finite_number(
                _require(config, "propellant_initial_mass", "motor"),
                "motor.propellant_initial_mass",
                positive=True,
            ),
            reshape_thrust_curve=config.get("reshape_thrust_curve", False),
        )

    if motor_type == "solid":
        solid_fields = {
            "grain_number": int(_require(config, "grain_number", "motor")),
            "grain_density": _finite_number(_require(config, "grain_density", "motor"), "motor.grain_density", positive=True),
            "grain_outer_radius": _finite_number(_require(config, "grain_outer_radius", "motor"), "motor.grain_outer_radius", positive=True),
            "grain_initial_inner_radius": _finite_number(_require(config, "grain_initial_inner_radius", "motor"), "motor.grain_initial_inner_radius", positive=True),
            "grain_initial_height": _finite_number(_require(config, "grain_initial_height", "motor"), "motor.grain_initial_height", positive=True),
            "grain_separation": _finite_number(_require(config, "grain_separation", "motor"), "motor.grain_separation"),
            "grains_center_of_mass_position": _finite_number(
                _require(config, "grains_center_of_mass_position", "motor"),
                "motor.grains_center_of_mass_position",
            ),
            "throat_radius": _finite_number(config.get("throat_radius", 0.01), "motor.throat_radius", positive=True),
            "reshape_thrust_curve": config.get("reshape_thrust_curve", False),
            "only_radial_burn": bool(config.get("only_radial_burn", False)),
        }
        return SolidMotor(**common, **solid_fields)

    raise RocketConfigurationError("motor.type must be 'generic' or 'solid'")


def _parachute_is_enabled(parachute: Mapping[str, Any]) -> bool:
    """Use the editable JSON switch as the authority for deployment."""
    return bool(parachute.get("enabled", True))


def simulation_readiness_issues(data: Mapping[str, Any]) -> list[str]:
    """Return fields that need attention before trusting a flight simulation."""
    issues: list[str] = []
    rocket = data.get("rocket", {})
    if not isinstance(rocket, Mapping):
        return ["rocket must be a JSON object"]

    if not rocket.get("aero_curves"):
        issues.append(
            "rocket.aero_curves requires a CSV containing Mach, Alpha, "
            "CD Power-Off, and CD Power-On"
        )
    inertia = rocket.get("inertia", {})
    try:
        diagonal = _inertia(inertia, "rocket.inertia")[:3]
        if any(value <= 0 for value in diagonal):
            issues.append("rocket.inertia I11, I22 and I33 must be measured positive dry inertias")
    except RocketConfigurationError as exc:
        issues.append(str(exc))
    if rocket.get("launch_mass") is None and rocket.get("mass") is None:
        issues.append("rocket.launch_mass is required")
    parachutes = rocket.get("parachutes", [])
    if not isinstance(parachutes, list):
        issues.append("rocket.parachutes must be an array")
    else:
        enabled_parachutes = [
            parachute
            for parachute in parachutes
            if isinstance(parachute, Mapping) and _parachute_is_enabled(parachute)
        ]
        for index, parachute in enumerate(enabled_parachutes):
            if parachute.get("diameter") is None or parachute.get("cd") is None:
                issues.append(
                    f"enabled parachute {index} requires diameter in meters and cd"
                )
            if parachute.get("trigger") is None:
                issues.append(f"enabled parachute {index} requires a deployment trigger")
    for index, stage in enumerate(data.get("stages", [])):
        if str(stage.get("part_type", "")).lower() == "nosecone" and not stage.get("shape"):
            issues.append(f"stages[{index}].shape is required; using 'conical' only as a fallback")
    return issues


def build_rocket(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
    require_simulation_ready: bool = False,
    rocket_key: str | None = None,
) -> Rocket:
    """Construct and return a RocketPy :class:`Rocket` from decoded JSON.

    Set ``require_simulation_ready=True`` to reject the incomplete placeholders
    emitted by ``cdx1tojson.py``.  The default still creates a useful geometric
    Rocket with RocketPy's EmptyMotor so converted files can be inspected.
    """
    if data.get("format") != "rocketpy-cdx1":
        raise RocketConfigurationError("format must be 'rocketpy-cdx1'")
    if data.get("format_version") not in (1, 2):
        raise RocketConfigurationError("Only format_version 1 and 2 are supported")
    data = _select_rocket_data(data, rocket_key)
    units = data.get("units", {})
    expected_units = {"length": "m", "mass": "kg", "time": "s"}
    for quantity, expected in expected_units.items():
        if units.get(quantity) != expected:
            raise RocketConfigurationError(f"units.{quantity} must be {expected!r}")

    config = _require(data, "rocket", "root")
    if not isinstance(config, Mapping):
        raise RocketConfigurationError("rocket must be a JSON object")
    stages = data.get("stages", [])
    if not isinstance(stages, list):
        raise RocketConfigurationError("stages must be an array")
    base_dir = Path(base_dir)

    issues = simulation_readiness_issues(data)
    if require_simulation_ready and issues:
        raise RocketConfigurationError(
            "JSON is not simulation-ready:\n- " + "\n- ".join(issues)
        )

    orientation = config.get("coordinate_system_orientation", "tail_to_nose")
    if orientation not in {"tail_to_nose", "nose_to_tail"}:
        raise RocketConfigurationError(
            "rocket.coordinate_system_orientation must be 'tail_to_nose' or 'nose_to_tail'"
        )
    total_length = _total_length(stages)

    automatic_motor = None
    motor_curve_path = None
    loaded_motor_mass = None
    if config.get("thrust_curve"):
        motor_geometry = config.get("motor", {})
        if not isinstance(motor_geometry, Mapping):
            raise RocketConfigurationError("rocket.motor must be a JSON object")
        automatic_motor, loaded_motor_mass, motor_curve_path = _motor_from_eng(
            config["thrust_curve"], base_dir, motor_geometry
        )
        launch_mass = _finite_number(
            _require(config, "launch_mass", "rocket"), "rocket.launch_mass", positive=True
        )
        dry_rocket_mass = launch_mass - loaded_motor_mass
        if dry_rocket_mass <= 0:
            raise RocketConfigurationError(
                f"rocket.launch_mass ({launch_mass:g} kg) must exceed loaded motor mass "
                f"({loaded_motor_mass:g} kg) from {motor_curve_path}"
            )
    else:
        # Motorless configurations are valid ballistic rockets. Treat an
        # explicitly supplied dry mass as authoritative; accept launch_mass as
        # the dry mass for older motorless JSON files.
        launch_mass = config.get("launch_mass")
        dry_rocket_mass = _finite_number(
            config.get("mass", launch_mass), "rocket.mass", positive=True
        )

    # CDX1 CG and component locations are measured aft from the nose tip.
    center_of_mass = _finite_number(
        _require(config, "center_of_mass_without_motor", "rocket"),
        "rocket.center_of_mass_without_motor",
    )
    center_of_mass = _rocket_position(center_of_mass, total_length, orientation)

    aero_curve_path = None
    if config.get("aero_curves"):
        alpha = _finite_number(config.get("aero_curve_alpha", 0), "rocket.aero_curve_alpha")
        power_off, power_on, aero_curve_path = _aero_sources_from_csv(
            config["aero_curves"], base_dir, alpha=alpha
        )
    else:
        # Zero is deliberately used only for non-strict geometry inspection.
        # Strict mode has already rejected the missing combined aero CSV.
        power_off = 0.0
        power_on = 0.0

    rocket = Rocket(
        radius=_finite_number(_require(config, "radius", "rocket"), "rocket.radius", positive=True),
        mass=dry_rocket_mass,
        inertia=_inertia(_require(config, "inertia", "rocket"), "rocket.inertia"),
        power_off_drag=power_off,
        power_on_drag=power_on,
        center_of_mass_without_motor=center_of_mass,
        coordinate_system_orientation=orientation,
    )
    if isinstance(power_off, list):
        # RocketPy wraps these tables in callables for its 7-DOF drag model.
        # Its visualizer cannot recover the original domain from that wrapper
        # and otherwise falls back to a hard-coded Mach 0-2 plot. Restore the
        # requested CSV domain on the 1-D functions used by plots and lookup.
        plot_max_mach = _finite_number(
            config.get("aero_plot_max_mach", 8.0),
            "rocket.aero_plot_max_mach",
            positive=True,
        )
        plotted_power_off = [point for point in power_off if point[0] <= plot_max_mach]
        plotted_power_on = [point for point in power_on if point[0] <= plot_max_mach]
        if len(plotted_power_off) < 2 or len(plotted_power_on) < 2:
            raise RocketConfigurationError(
                "rocket.aero_plot_max_mach must include at least two aerodynamic rows"
            )
        rocket.power_off_drag_by_mach.set_source(plotted_power_off)
        rocket.power_off_drag_by_mach.set_interpolation("linear")
        rocket.power_on_drag_by_mach.set_source(plotted_power_on)
        rocket.power_on_drag_by_mach.set_interpolation("linear")
        rocket.aero_plot_max_mach = min(
            plotted_power_off[-1][0], plotted_power_on[-1][0]
        )

        def plot_stability_margin_to_configured_mach() -> None:
            rocket.stability_margin.plot_2d(
                lower=0,
                upper=[rocket.aero_plot_max_mach, rocket.motor.burn_out_time],
                samples=[80, 20],
                disp_type="surface",
                alpha=1,
            )

        # RocketPy currently hard-codes Mach 2 in this visualizer method.
        rocket.plots.stability_margin = plot_stability_margin_to_configured_mach
    rocket.name = str(config.get("name", "Rocket"))
    rocket.aero_curve_file = aero_curve_path
    rocket.aero_curve_mach_range = (
        (power_off[0][0], power_off[-1][0]) if isinstance(power_off, list) else None
    )
    rocket.drag_curves_are_identical = (
        power_off == power_on if isinstance(power_off, list) else None
    )
    rocket.launch_mass_from_json = launch_mass
    rocket.loaded_motor_mass_from_curve = loaded_motor_mass
    rocket.thrust_curve_file = motor_curve_path

    for index, stage in enumerate(stages):
        part_type = str(stage.get("part_type", "")).lower()
        stage_location = _finite_number(stage.get("location", 0), f"stages[{index}].location")
        stage_length = _finite_number(stage.get("length", 0), f"stages[{index}].length")
        stage_radius = _finite_number(stage.get("diameter", 2 * rocket.radius), f"stages[{index}].diameter") / 2

        if part_type == "nosecone":
            rocket.add_nose(
                length=stage_length,
                kind=str(stage.get("shape", "conical")).lower(),
                position=_rocket_position(stage_location, total_length, orientation),
                bluffness=_finite_number(stage.get("bluffness", 0), f"stages[{index}].bluffness"),
                base_radius=stage_radius,
                name=str(stage.get("name", "Nose Cone")),
            )

        shoulder_length = _finite_number(
            stage.get("shoulder_length", 0), f"stages[{index}].shoulder_length"
        )
        inside_diameter = _finite_number(
            stage.get("inside_diameter", 0), f"stages[{index}].inside_diameter"
        )
        if part_type == "fincan":
            if stage_length <= 0 or stage_radius <= 0 or inside_diameter <= 0:
                raise RocketConfigurationError(
                    f"stages[{index}] FinCan requires positive length, diameter, "
                    "and inside_diameter"
                )
            if inside_diameter > 2 * stage_radius:
                raise RocketConfigurationError(
                    f"stages[{index}].inside_diameter cannot exceed its diameter"
                )
            # A CDX1 FinCan is an external sleeve over the preceding airframe.
            # RocketPy has no combined fin-can surface, so represent the sleeve
            # as a Tail transition and add its nested Fin entries below.
            rocket.add_tail(
                top_radius=inside_diameter / 2,
                bottom_radius=stage_radius,
                length=stage_length,
                position=_rocket_position(stage_location, total_length, orientation),
                radius=stage_radius,
                name=str(stage.get("name", "Fin Can Tail")),
            )

        if (
            part_type == "booster"
            and shoulder_length > 0
            and inside_diameter > 0
            and not math.isclose(inside_diameter / 2, stage_radius)
        ):
            # RASAero models the forward interstage transition through the
            # booster's shoulder and inside diameter. The top plane mates to
            # the narrower sustainer; the bottom plane is the booster body.
            rocket.add_tail(
                top_radius=inside_diameter / 2,
                bottom_radius=stage_radius,
                length=shoulder_length,
                position=_rocket_position(stage_location, total_length, orientation),
                radius=stage_radius,
                name=f"{stage.get('part_type', 'Booster')} Transition",
            )

        boattail_length = _finite_number(stage.get("boattail_length", 0), f"stages[{index}].boattail_length")
        rear_diameter = _finite_number(
            stage.get("boattail_rear_diameter", 0), f"stages[{index}].boattail_rear_diameter"
        )
        if boattail_length > 0 and rear_diameter > 0:
            # The boattail starts at the stage's aft end and extends aft.
            forward = stage_location + stage_length - boattail_length
            rocket.add_tail(
                top_radius=stage_radius,
                bottom_radius=rear_diameter / 2,
                length=boattail_length,
                position=_rocket_position(forward, total_length, orientation),
                radius=stage_radius,
                name=f"{stage.get('part_type', 'Stage')} Boattail",
            )

        for fin_index, fin in enumerate(stage.get("fins", [])):
            # RASAero's fin Location is the distance forward from the component
            # aft end to the root leading edge.
            leading_edge_from_nose = stage_location + stage_length - _finite_number(
                fin.get("location", fin.get("root_chord", 0)),
                f"stages[{index}].fins[{fin_index}].location",
            )
            rocket.add_trapezoidal_fins(
                n=int(_require(fin, "count", f"stages[{index}].fins[{fin_index}]")),
                root_chord=_finite_number(_require(fin, "root_chord", "fin"), "fin.root_chord", positive=True),
                tip_chord=_finite_number(_require(fin, "tip_chord", "fin"), "fin.tip_chord", positive=True),
                span=_finite_number(_require(fin, "span", "fin"), "fin.span", positive=True),
                position=_rocket_position(leading_edge_from_nose, total_length, orientation),
                sweep_length=_finite_number(fin.get("sweep_distance", 0), "fin.sweep_distance"),
                cant_angle=_finite_number(fin.get("cant_angle", 0), "fin.cant_angle"),
                radius=stage_radius,
                name=str(fin.get("name", f"{stage.get('part_type', 'Stage')} Fins")),
            )

    motor_config = config.get("motor")
    if automatic_motor is not None:
        motor_position = _finite_number(config.get("motor_position", 0), "rocket.motor_position")
        rocket.add_motor(automatic_motor, position=motor_position)
    elif motor_config:
        if not isinstance(motor_config, Mapping):
            raise RocketConfigurationError("rocket.motor must be a JSON object")
        motor_position = _finite_number(
            _require(motor_config, "position", "rocket.motor"), "rocket.motor.position"
        )
        if motor_config.get("position_reference", "rocket") == "nose_tip":
            motor_position = _rocket_position(motor_position, total_length, orientation)
        rocket.add_motor(_motor_from_json(motor_config, base_dir), position=motor_position)

    rail_buttons = config.get("rail_buttons")
    if (
        isinstance(rail_buttons, Mapping)
        and rail_buttons.get("upper_position") is not None
        and rail_buttons.get("lower_position") is not None
    ):
        upper = _finite_number(_require(rail_buttons, "upper_position", "rocket.rail_buttons"), "rail_buttons.upper_position")
        lower = _finite_number(_require(rail_buttons, "lower_position", "rocket.rail_buttons"), "rail_buttons.lower_position")
        if rail_buttons.get("position_reference", "rocket") == "nose_tip":
            upper = _rocket_position(upper, total_length, orientation)
            lower = _rocket_position(lower, total_length, orientation)
        rocket.set_rail_buttons(
            upper_button_position=upper,
            lower_button_position=lower,
            angular_position=_finite_number(rail_buttons.get("angular_position", 45), "rail_buttons.angular_position"),
            radius=rail_buttons.get("radius"),
        )

    for index, parachute in enumerate(config.get("parachutes", [])):
        if not isinstance(parachute, Mapping):
            raise RocketConfigurationError(f"rocket.parachutes[{index}] must be an object")
        if not _parachute_is_enabled(parachute):
            continue
        trigger = _require(parachute, "trigger", f"rocket.parachutes[{index}]")
        if isinstance(trigger, str) and trigger.lower() != "apogee":
            raise RocketConfigurationError(
                f"rocket.parachutes[{index}].trigger must be 'apogee' or an AGL altitude"
            )
        diameter = _finite_number(
            _require(parachute, "diameter", f"rocket.parachutes[{index}]"),
            f"rocket.parachutes[{index}].diameter",
            positive=True,
        )
        drag_coefficient = _finite_number(
            _require(parachute, "cd", f"rocket.parachutes[{index}]"),
            f"rocket.parachutes[{index}].cd",
            positive=True,
        )
        cd_a = drag_coefficient * math.pi * (diameter / 2) ** 2
        rocket.add_parachute(
            name=str(_require(parachute, "name", f"rocket.parachutes[{index}]")),
            cd_s=cd_a,
            trigger=trigger.lower() if isinstance(trigger, str) else _finite_number(trigger, "parachute.trigger"),
            sampling_rate=_finite_number(parachute.get("sampling_rate", 100), "parachute.sampling_rate", positive=True),
            lag=_finite_number(parachute.get("lag", 0), "parachute.lag"),
            noise=tuple(parachute.get("noise", (0, 0, 0))),
        )

    # Useful to callers and harmless to RocketPy: launch_site is Environment /
    # Flight input and therefore is intentionally not consumed by Rocket.
    rocket.source_configuration = data
    rocket.simulation_readiness_issues = issues
    return rocket


def load_rocket(
    path: str | Path,
    *,
    require_simulation_ready: bool = False,
    rocket_key: str | None = None,
) -> Rocket:
    """Read *path* and return one corresponding RocketPy Rocket.

    For multi-rocket version 2 JSON, choose ``full_stack``, ``sustainer``, or
    ``booster`` with ``rocket_key``. If omitted, the file's ``default_rocket``
    is used.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RocketConfigurationError(f"Could not read {path}: {exc}") from exc
    return build_rocket(
        data,
        base_dir=path.resolve().parent,
        require_simulation_ready=require_simulation_ready,
        rocket_key=rocket_key,
    )


# Friendly aliases for consumers that prefer constructor-like names.
rocket_from_json = load_rocket
interpret = load_rocket


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_file", type=Path, nargs="?", default=Path(__file__).with_name("test.json"))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail unless all fields needed for a meaningful flight are present",
    )
    parser.add_argument(
        "--rocket",
        dest="rocket_key",
        help="rocket configuration to load (full_stack, sustainer, or booster)",
    )
    args = parser.parse_args(argv)
    try:
        rocket = load_rocket(
            args.json_file,
            require_simulation_ready=args.strict,
            rocket_key=args.rocket_key,
        )
    except RocketConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Built RocketPy Rocket: {rocket.name}")
    if rocket.simulation_readiness_issues:
        print("Additional JSON fields/corrections required for simulation:")
        for issue in rocket.simulation_readiness_issues:
            print(f"  - {issue}")
    else:
        print("Configuration is simulation-ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
