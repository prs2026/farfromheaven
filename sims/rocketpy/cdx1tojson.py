"""Convert a RASAero II .CDX1 file to a RocketPy-oriented JSON file.

RASAero stores dimensions in inches, launch mass in pounds, altitude in feet,
pressure in inches of mercury, wind speed in miles per hour, and launch-site
temperature in degrees Fahrenheit. The generated file uses SI units and
contains the fields needed to construct a RocketPy ``Rocket``. CDX1 files
do not contain thrust curves, drag curves, or mass moments of inertia, so
those values are represented by zero/empty placeholders rather than omitted.
When a booster is present, the output contains selectable ``full_stack``,
``sustainer``, and ``booster`` rocket configurations in the same file.

Usage:
	python cdx1tojson.py input.CDX1 output.json --full-stack-aero full.csv --sustainer-aero sustainer.csv --full-stack-thrust booster.eng --sustainer-thrust sustainer.eng
	python cdx1tojson.py input.CDX1
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


INCH_TO_M = 0.0254
FOOT_TO_M = 0.3048
POUND_TO_KG = 0.45359237
INHG_TO_PA = 3386.389
MPH_TO_MPS = 0.44704
FAHRENHEIT_TO_KELVIN_OFFSET = 459.67
POUND_PER_CUBIC_INCH_TO_KG_PER_CUBIC_METER = POUND_TO_KG / INCH_TO_M**3
DEFAULT_PROPELLANT_DENSITY = 0.065 * POUND_PER_CUBIC_INCH_TO_KG_PER_CUBIC_METER


def _number(value: str | None, default: float = 0.0) -> float:
	"""Return a finite float from a CDX1 text value."""
	if value is None or not value.strip():
		return default
	try:
		result = float(value)
	except ValueError:
		return default
	return result if math.isfinite(result) else default


def _boolean(value: str | None) -> bool:
	return (value or "").strip().lower() in {"true", "1", "yes"}


def _text(element: ET.Element, name: str, default: str = "") -> str:
	child = element.find(name)
	return (child.text or default).strip() if child is not None else default


def _float(element: ET.Element, name: str, scale: float = 1.0) -> float:
	return _number(_text(element, name)) * scale


def _fin(fin: ET.Element) -> dict[str, Any]:
	return {
		"count": int(_number(_text(fin, "Count"))),
		"root_chord": _float(fin, "Chord", INCH_TO_M),
		"span": _float(fin, "Span", INCH_TO_M),
		"sweep_distance": _float(fin, "SweepDistance", INCH_TO_M),
		"tip_chord": _float(fin, "TipChord", INCH_TO_M),
		"thickness": _float(fin, "Thickness", INCH_TO_M),
		"leading_edge_radius": _float(fin, "LERadius", INCH_TO_M),
		"location": _float(fin, "Location", INCH_TO_M),
		"airfoil_section": _text(fin, "AirfoilSection"),
	}


def _stage(element: ET.Element) -> dict[str, Any]:
	stage = {
		"part_type": _text(element, "PartType"),
		"length": _float(element, "Length", INCH_TO_M),
		"diameter": _float(element, "Diameter", INCH_TO_M),
		"inside_diameter": _float(element, "InsideDiameter", INCH_TO_M),
		"location": _float(element, "Location", INCH_TO_M),
		"shoulder_length": _float(element, "ShoulderLength", INCH_TO_M),
		"boattail_length": _float(element, "BoattailLength", INCH_TO_M),
		"boattail_rear_diameter": _float(element, "BoattailRearDiameter", INCH_TO_M),
		"color": _text(element, "Color"),
		"fins": [_fin(fin) for fin in element.findall("Fin")],
	}
	if element.tag == "NoseCone":
		stage["shape"] = _text(element, "Shape", "Conical")
	return stage


def _zero_inertia() -> dict[str, float]:
	return {"I11": 0.0, "I22": 0.0, "I33": 0.0, "I12": 0.0, "I13": 0.0, "I23": 0.0}


def _motor_geometry(
	thrust_curve: str | Path | None, base_dir: str | Path = "."
) -> dict[str, Any] | None:
	"""Read motor diameter and length from a RASP header for grain modeling."""
	if thrust_curve is None:
		return None
	path = Path(thrust_curve)
	if not path.is_absolute():
		path = Path(base_dir) / path
	try:
		with path.open(encoding="utf-8-sig") as stream:
			for line in stream:
				line = line.strip()
				if line and not line.startswith(";"):
					fields = line.split()
					break
			else:
				raise ValueError("file has no RASP motor header")
		if len(fields) < 6:
			raise ValueError("header needs designation, diameter, length, delays, propellant mass, and total mass")
		diameter = float(fields[1]) / 1000
		length = float(fields[2]) / 1000
	except (OSError, ValueError) as exc:
		raise ValueError(f"Invalid RASP motor header in {path}: {exc}") from exc
	if not math.isfinite(diameter) or diameter <= 0 or not math.isfinite(length) or length <= 0:
		raise ValueError(f"Motor diameter and length must be positive in {path}")
	return {
		"type": "solid",
		"grain_outer_radius": diameter / 2,
		"grain_density": DEFAULT_PROPELLANT_DENSITY,
		"grain_number": 1,
		"grain_separation": 0.0,
		"minimum_dry_mass": 0.01,
		"propellant_length": length,
		"geometry_source": "RASP motor diameter and length; density assumed as 0.065 lb/in^3",
	}


def _rebase_stages(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Copy stages and move their forward-most point to the zero datum."""
	if not stages:
		return []
	origin = min(stage["location"] for stage in stages)
	result = copy.deepcopy(stages)
	for stage in result:
		stage["location"] -= origin
	return result


def _recovery_events(recovery: ET.Element) -> list[dict[str, Any]]:
	"""Return both RASAero recovery events with SI parachute dimensions."""
	events = []
	for index in (1, 2):
		event_type = _text(recovery, f"EventType{index}", "None")
		if event_type.strip().lower() == "apogee":
			trigger: str | float | None = "apogee"
		elif event_type.strip().lower() == "altitude":
			trigger = _float(recovery, f"Altitude{index}", FOOT_TO_M)
		else:
			trigger = None
		events.append(
			{
				"name": f"recovery_{index}",
				"enabled": _boolean(_text(recovery, f"Event{index}")),
				"device_type": _text(recovery, f"DeviceType{index}", "None"),
				"event_type": event_type,
				"trigger": trigger,
				"deployment_altitude": _float(recovery, f"Altitude{index}", FOOT_TO_M),
				"diameter": _float(recovery, f"Size{index}", INCH_TO_M),
				"cd": _float(recovery, f"CD{index}"),
				"sampling_rate": 100,
				"lag": 0.0,
				"noise": [0.0, 0.0, 0.0],
			}
		)
	return events


def _rocket_definition(
	*,
	name: str,
	stages: list[dict[str, Any]],
	mass: float,
	center_of_mass: float,
	engine_name: str,
	aero_curves: str | Path | None,
	thrust_curve: str | Path | None,
	mass_source: str,
	parachutes: list[dict[str, Any]],
	asset_base_dir: str | Path = ".",
) -> dict[str, Any]:
	if not stages:
		raise ValueError(f"Cannot create rocket configuration {name!r} without stages")
	radius = max(stage["diameter"] for stage in stages) / 2
	return {
		"name": name,
		"radius": radius,
		"launch_mass": mass,
		"mass_source": mass_source,
		"inertia": _zero_inertia(),
		"aero_curves": str(Path(aero_curves)) if aero_curves else None,
		"aero_curve_alpha": 0.0,
		"aero_plot_max_mach": 8.0,
		"thrust_curve": str(Path(thrust_curve)) if thrust_curve else None,
		"motor": _motor_geometry(thrust_curve, asset_base_dir),
		"center_of_mass_without_motor": center_of_mass,
		"coordinate_system_orientation": "tail_to_nose",
		"engine_name": engine_name,
		"rail_buttons": {
			"upper_position": None,
			"lower_position": None,
			"position_reference": "nose_tip",
			"angular_position": 45.0,
			"radius": radius,
		},
		"parachutes": copy.deepcopy(parachutes),
	}


def _simulation(simulation: ET.Element | None) -> dict[str, Any]:
	if simulation is None:
		simulation = ET.Element("Simulation")
	return {
		"sustainer": {
			"engine": _text(simulation, "SustainerEngine"),
			"launch_mass": _float(simulation, "SustainerLaunchWt", POUND_TO_KG),
			"center_of_mass": _float(simulation, "SustainerCG", INCH_TO_M),
			"ignition_delay": _float(simulation, "SustainerIgnitionDelay"),
		},
		"boosters": [
			{
				"engine": _text(simulation, f"Booster{i}Engine"),
				"included": _boolean(_text(simulation, f"IncludeBooster{i}")),
				"launch_mass": _float(simulation, f"Booster{i}LaunchWt", POUND_TO_KG),
				"center_of_mass": _float(simulation, f"Booster{i}CG", INCH_TO_M),
				"ignition_delay": _float(simulation, f"Booster{i}IgnitionDelay"),
				"separation_delay": _float(simulation, f"Booster{i}SeparationDelay"),
			}
			for i in (1, 2)
		],
		"results": {
			"flight_time": _float(simulation, "FlightTime"),
			"time_to_apogee": _float(simulation, "TimetoApogee"),
			"max_altitude": _float(simulation, "MaxAltitude", FOOT_TO_M),
			"max_velocity": _float(simulation, "MaxVelocity", FOOT_TO_M),
		},
	}


def convert_cdx1(
	input_path: str | Path,
	*,
	aero_curves: str | Path | None = None,
	full_stack_aero: str | Path | None = None,
	sustainer_aero: str | Path | None = None,
	booster_aero: str | Path | None = None,
	thrust_curve: str | Path | None = None,
	full_stack_thrust: str | Path | None = None,
	sustainer_thrust: str | Path | None = None,
	booster_thrust: str | Path | None = None,
	asset_base_dir: str | Path = ".",
) -> dict[str, Any]:
	"""Parse *input_path* and return the generated JSON-compatible object."""
	input_path = Path(input_path)
	root = ET.parse(input_path).getroot()
	design = root.find("RocketDesign")
	if design is None:
		raise ValueError("The CDX1 file does not contain RocketDesign")

	stages = [
		_stage(part)
		for part in design
		if part.tag in {"NoseCone", "BodyTube", "Booster"}
	]
	simulation_list = root.find("SimulationList")
	simulation = simulation_list.find("Simulation") if simulation_list is not None else None
	sim = _simulation(simulation)
	sustainer = sim["sustainer"]

	launch_site = root.find("LaunchSite")
	if launch_site is None:
		launch_site = ET.Element("LaunchSite")
	recovery = root.find("Recovery")
	if recovery is None:
		recovery = ET.Element("Recovery")
	recovery_events = _recovery_events(recovery)

	sustainer_stages = [stage for stage in stages if stage["part_type"].lower() != "booster"]
	booster_stages = [stage for stage in stages if stage["part_type"].lower() == "booster"]
	sustainer_rocket = _rocket_definition(
		name=f"{input_path.stem} - Sustainer" if booster_stages else input_path.stem,
		stages=sustainer_stages,
		mass=sustainer["launch_mass"],
		center_of_mass=sustainer["center_of_mass"],
		engine_name=sustainer["engine"],
		aero_curves=sustainer_aero or aero_curves,
		thrust_curve=sustainer_thrust or thrust_curve,
		mass_source="RASAero SustainerLaunchWt (lb converted to kg)",
		parachutes=recovery_events,
		asset_base_dir=asset_base_dir,
	)

	result = {
		"format": "rocketpy-cdx1",
		"format_version": 1,
		"units": {"length": "m", "mass": "kg", "time": "s", "temperature": "K", "pressure": "Pa"},
		"source": {"file": input_path.name, "format": "RASAero II CDX1"},
		"rocket": sustainer_rocket,
		"stages": _rebase_stages(sustainer_stages),
		"launch_site": {
			"altitude": _float(launch_site, "Altitude", FOOT_TO_M),
			"pressure": _float(launch_site, "Pressure", INHG_TO_PA),
			"temperature": (_float(launch_site, "Temperature") + FAHRENHEIT_TO_KELVIN_OFFSET) * 5 / 9,
			"wind_speed": _float(launch_site, "WindSpeed", MPH_TO_MPS),
			"rod_angle": _float(launch_site, "RodAngle"),
			"rod_length": _float(launch_site, "RodLength", FOOT_TO_M),
		},
		"recovery": {"events": copy.deepcopy(recovery_events)},
	}

	if not booster_stages:
		return result

	# RASAero's BoosterNLaunchWt/CG describe the complete vehicle while that
	# booster is attached. Prefer an explicitly included booster, then the first
	# booster entry containing mass data.
	booster_sim = next(
		(booster for booster in sim["boosters"] if booster["included"]),
		next(
			(booster for booster in sim["boosters"] if booster["launch_mass"] > 0),
			sim["boosters"][0],
		),
	)
	full_mass = booster_sim["launch_mass"]
	full_cg = booster_sim["center_of_mass"]
	booster_mass = full_mass - sustainer["launch_mass"]
	booster_origin = min(stage["location"] for stage in booster_stages)
	if booster_mass > 0:
		booster_cg_global = (
			full_mass * full_cg
			- sustainer["launch_mass"] * sustainer["center_of_mass"]
		) / booster_mass
	else:
		# No standalone booster mass properties can be inferred. Preserve the
		# CDX1 value so the zero mass remains an obvious, correctable placeholder.
		booster_cg_global = full_cg

	full_stack_rocket = _rocket_definition(
		name=f"{input_path.stem} - Full Stack",
		stages=stages,
		mass=full_mass,
		center_of_mass=full_cg,
		engine_name=booster_sim["engine"],
		aero_curves=full_stack_aero or aero_curves,
		thrust_curve=full_stack_thrust or thrust_curve,
		mass_source="RASAero BoosterLaunchWt (lb converted to kg; complete vehicle with booster attached)",
		parachutes=recovery_events,
		asset_base_dir=asset_base_dir,
	)
	booster_rocket = _rocket_definition(
		name=f"{input_path.stem} - Booster",
		stages=booster_stages,
		mass=booster_mass,
		center_of_mass=booster_cg_global - booster_origin,
		engine_name=booster_sim["engine"],
		aero_curves=booster_aero,
		thrust_curve=booster_thrust,
		mass_source="Derived as full-stack launch mass minus sustainer launch mass",
		parachutes=recovery_events,
		asset_base_dir=asset_base_dir,
	)

	result["format_version"] = 2
	result["default_rocket"] = "full_stack"
	result["rockets"] = {
		"full_stack": {
			"format": "rocketpy-cdx1-full-stack",
			"configuration_type": "sustainer_with_booster",
			"rocket": full_stack_rocket,
			"stages": copy.deepcopy(stages),
			"simulation": copy.deepcopy(booster_sim),
		},
		"sustainer": {
			"format": "rocketpy-cdx1-sustainer",
			"configuration_type": "sustainer",
			"rocket": sustainer_rocket,
			"stages": _rebase_stages(sustainer_stages),
			"simulation": copy.deepcopy(sustainer),
		},
		"booster": {
			"format": "rocketpy-cdx1-booster",
			"configuration_type": "booster",
			"rocket": booster_rocket,
			"stages": _rebase_stages(booster_stages),
			"simulation": copy.deepcopy(booster_sim),
		},
	}
	# Keep the default configuration at the top level so existing consumers of
	# version 1 files continue to construct the complete launch vehicle.
	result["rocket"] = copy.deepcopy(full_stack_rocket)
	result["stages"] = copy.deepcopy(stages)
	return result


def _existing_file(value: str) -> Path:
	"""Argparse type which accepts paths to existing files only."""
	path = Path(value)
	if not path.is_file():
		raise argparse.ArgumentTypeError(f"file does not exist: {path}")
	return path


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("input", type=_existing_file, help="RASAero II .CDX1 file")
	parser.add_argument("output", type=Path, nargs="?", help="output JSON path")
	aero_group = parser.add_argument_group("aerodynamic curve inputs")
	aero_group.add_argument(
		"--aero-curves",
		type=_existing_file,
		help="combined aerodynamic CSV for a single-stage rocket or fallback for all configurations",
	)
	aero_group.add_argument(
		"--full-stack-aero", "--full-stack-aero-curves",
		dest="full_stack_aero", type=_existing_file,
		help="combined aerodynamic CSV for the attached stack",
	)
	aero_group.add_argument(
		"--sustainer-aero", "--sustainer-aero-curves",
		dest="sustainer_aero", type=_existing_file,
		help="combined aerodynamic CSV for the sustainer",
	)
	aero_group.add_argument(
		"--booster-aero", "--booster-aero-curves",
		dest="booster_aero", type=_existing_file,
		help="combined aerodynamic CSV for the booster alone",
	)
	thrust_group = parser.add_argument_group("motor thrust curve inputs")
	thrust_group.add_argument(
		"--thrust-curve",
		type=_existing_file,
		help="RASP .eng thrust curve for a single-stage rocket or fallback for all configurations",
	)
	thrust_group.add_argument(
		"--full-stack-thrust", "--full-stack-thrust-curve",
		dest="full_stack_thrust", type=_existing_file,
		help="RASP .eng curve for the booster motor used by the attached stack",
	)
	thrust_group.add_argument(
		"--sustainer-thrust", "--sustainer-thrust-curve",
		dest="sustainer_thrust", type=_existing_file,
		help="RASP .eng curve for the sustainer motor",
	)
	thrust_group.add_argument(
		"--booster-thrust", "--booster-thrust-curve",
		dest="booster_thrust", type=_existing_file,
		help="RASP .eng curve for the booster alone",
	)
	args = parser.parse_args()
	output = args.output or args.input.with_suffix(".json")

	def output_relative(path: Path | None) -> Path | None:
		if path is None:
			return None
		return Path(os.path.relpath(path.resolve(), output.resolve().parent))

	try:
		data = convert_cdx1(
			args.input,
			aero_curves=output_relative(args.aero_curves),
			full_stack_aero=output_relative(args.full_stack_aero),
			sustainer_aero=output_relative(args.sustainer_aero),
			booster_aero=output_relative(args.booster_aero),
			thrust_curve=output_relative(args.thrust_curve),
			full_stack_thrust=output_relative(args.full_stack_thrust),
			sustainer_thrust=output_relative(args.sustainer_thrust),
			booster_thrust=output_relative(args.booster_thrust),
			asset_base_dir=output.resolve().parent,
		)
		output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
	except (OSError, ET.ParseError, ValueError) as error:
		print(f"error: {error}", file=sys.stderr)
		return 1
	print(f"Wrote {output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
