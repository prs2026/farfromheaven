"""Reusable RocketPy simulation helpers."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

from rocketpy import EmptyMotor, Environment, Flight, GenericMotor, Rocket


@dataclass(frozen=True)
class FullStackSimulationResult:
    """Per-call multistage results safe to consume from concurrent workers."""

    solution: list[list[float]]
    flights: tuple[Flight, ...]
    staging_tilt: float | None
    ignition_time: float
    sustainer_ignited: bool
    tilt_lockout_triggered: bool


def run_single_simulation(
    rocket: Rocket,
    environment: Environment,
    *,
    rail_length: float = 5.2,
    inclination: float = 90.0,
    heading: float = 0.0,
    max_time_step: float = math.inf,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> Flight:
    """Run one simulation and return its RocketPy ``Flight`` object."""
    return Flight(
        rocket=rocket,
        environment=environment,
        rail_length=rail_length,
        inclination=inclination,
        heading=heading,
        max_time_step=max_time_step,
        rtol=rtol,
        atol=atol,
    )


def runfullstacksim(
    full_stack: Rocket,
    sustainer: Rocket,
    environment: Environment,
    time_limit: float,
    *,
    coast_period: float = 0.0,
    rail_length: float = 6.0,
    rod_angle: float = 0.0,
    heading: float = 0.0,
    max_time_step: float = math.inf,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    sustainer_ignition_max_tilt_deg: float | None = None,
    return_details: bool = False,
) -> list[list[float]] | FullStackSimulationResult:
    """Run the boost and sustainer phases and return the original solution format.

    The booster separates at burnout. The sustainer then coasts without a
    motor for ``coast_period`` seconds before ignition. ``rod_angle`` is
    measured away from vertical. Returned times are absolute seconds since
    launch, matching the original ``runfullstacksim`` function. When
    ``sustainer_ignition_max_tilt_deg`` is set and the ignition tilt exceeds
    it, the sustainer remains unpowered and continues through recovery.
    ``rtol`` and ``atol`` control the RocketPy integrator accuracy.
    """
    time_limit = float(time_limit)
    coast_period = float(coast_period)
    rail_length = float(rail_length)
    rod_angle = float(rod_angle)
    heading = float(heading)
    max_time_step = float(max_time_step)
    rtol = float(rtol)
    atol = float(atol)
    if sustainer_ignition_max_tilt_deg is not None:
        sustainer_ignition_max_tilt_deg = float(sustainer_ignition_max_tilt_deg)
    if not all(math.isfinite(value) for value in (time_limit, coast_period, rail_length, rod_angle, heading)):
        raise ValueError("time_limit, coast_period, rail_length, rod_angle, and heading must be finite")
    if coast_period < 0:
        raise ValueError("coast_period cannot be negative")
    if rail_length <= 0:
        raise ValueError("rail_length must be positive")
    if not 0 <= rod_angle < 90:
        raise ValueError("rod_angle must be in the range [0, 90) degrees from vertical")
    if not 0 <= heading < 360:
        raise ValueError("heading must be in the range [0, 360) degrees")
    if math.isnan(max_time_step) or max_time_step <= 0:
        raise ValueError("max_time_step must be positive")
    if not math.isfinite(rtol) or rtol <= 0 or not math.isfinite(atol) or atol <= 0:
        raise ValueError("rtol and atol must be positive finite numbers")
    if sustainer_ignition_max_tilt_deg is not None and not (
        math.isfinite(sustainer_ignition_max_tilt_deg)
        and 0 <= sustainer_ignition_max_tilt_deg <= 90
    ):
        raise ValueError("sustainer_ignition_max_tilt_deg must be in [0, 90]")

    BoosterMotor = full_stack.motor
    booster_burnout_time = float(BoosterMotor.burn_out_time)
    if not math.isfinite(booster_burnout_time) or booster_burnout_time <= 0:
        raise ValueError(
            "full_stack must contain the booster motor; regenerate the JSON "
            "with --booster-thrust"
        )
    ignitiontime = booster_burnout_time + coast_period
    if time_limit <= ignitiontime:
        raise ValueError(
            "time_limit must be greater than booster burnout plus coast_period"
        )

    StackFlight2 = Flight(
        rocket=full_stack,
        environment=environment,
        rail_length=rail_length,
        inclination=90.0 - rod_angle,
        heading=heading,
        max_time=booster_burnout_time,
        max_time_step=max_time_step,
        rtol=rtol,
        atol=atol,
        time_overshoot=True,
        name=f"{getattr(full_stack, 'name', 'Full Stack')} boost",
    )

    stagingtime = StackFlight2.solution[-1][0]
    staging_altitude = StackFlight2.solution[-1][3]
    if (
        stagingtime < booster_burnout_time - 1e-6
        or staging_altitude <= environment.elevation
    ):
        print("Staging aborted: full stack impacted before booster burnout")
        result = FullStackSimulationResult(
            StackFlight2.solution,
            (StackFlight2,),
            None,
            ignitiontime,
            False,
            False,
        )
        runfullstacksim.last_flights = result.flights
        runfullstacksim.staging_tilt = result.staging_tilt
        runfullstacksim.ignition_time = result.ignition_time
        runfullstacksim.sustainer_ignited = result.sustainer_ignited
        runfullstacksim.tilt_lockout_triggered = result.tilt_lockout_triggered
        return result if return_details else result.solution
    sustainer_motor = sustainer.motor
    coast_motor = GenericMotor(
        thrust_source=0,
        burn_time=max(time_limit, 1.0),
        chamber_radius=float(sustainer_motor.grain_outer_radius),
        chamber_height=float(sustainer_motor.propellant_length),
        chamber_position=float(sustainer_motor.grains_center_of_mass_position),
        propellant_initial_mass=float(sustainer_motor.propellant_initial_mass),
        nozzle_radius=float(sustainer_motor.nozzle_radius),
        dry_mass=float(sustainer_motor.dry_mass),
        center_of_dry_mass_position=float(
            sustainer_motor.center_of_dry_mass_position
        ),
        dry_inertia=(
            float(sustainer_motor.dry_I_11),
            float(sustainer_motor.dry_I_22),
            float(sustainer_motor.dry_I_33),
            float(sustainer_motor.dry_I_12),
            float(sustainer_motor.dry_I_13),
            float(sustainer_motor.dry_I_23),
        ),
        nozzle_position=float(sustainer_motor.nozzle_position),
        coordinate_system_orientation=(
            sustainer_motor.coordinate_system_orientation
        ),
    )
    SustainerNOMOTOR = deepcopy(sustainer)
    SustainerNOMOTOR.motor = EmptyMotor()
    SustainerNOMOTOR.add_motor(
        coast_motor, position=getattr(sustainer, "motor_position", 0.0)
    )
    SustainerNOMOTOR.parachutes = []

    if coast_period > 0:
        SustainerNOMOTORFlight2 = Flight(
            rocket=SustainerNOMOTOR,
            environment=environment,
            initial_solution=StackFlight2,
            rail_length=0.01,
            inclination=StackFlight2.attitude_angle(stagingtime),
            heading=heading,
            max_time=ignitiontime,
            max_time_step=max_time_step,
            rtol=rtol,
            atol=atol,
            time_overshoot=True,
            name=f"{getattr(sustainer, 'name', 'Sustainer')} coast",
        )
        coast_end_time = SustainerNOMOTORFlight2.solution[-1][0]
        coast_end_altitude = SustainerNOMOTORFlight2.solution[-1][3]
        if (
            coast_end_time < ignitiontime - 1e-6
            or coast_end_altitude <= environment.elevation
        ):
            print("Sustainer ignition aborted: sustainer impacted during coast")
            result = FullStackSimulationResult(
                SustainerNOMOTORFlight2.solution,
                (StackFlight2, SustainerNOMOTORFlight2),
                None,
                ignitiontime,
                False,
                False,
            )
            runfullstacksim.last_flights = result.flights
            runfullstacksim.staging_tilt = result.staging_tilt
            runfullstacksim.ignition_time = result.ignition_time
            runfullstacksim.sustainer_ignited = result.sustainer_ignited
            runfullstacksim.tilt_lockout_triggered = result.tilt_lockout_triggered
            return result if return_details else result.solution
        ignition_state = SustainerNOMOTORFlight2.solution[-1][:]
        comparison_flights = (StackFlight2, SustainerNOMOTORFlight2)
    else:
        SustainerNOMOTORFlight2 = None
        ignition_state = StackFlight2.solution[-1][:]
        comparison_flights = (StackFlight2,)

    stagingtilt = abs(
        90.0
        - (
            SustainerNOMOTORFlight2.attitude_angle(ignitiontime)
            if SustainerNOMOTORFlight2 is not None
            else StackFlight2.attitude_angle(stagingtime)
        )
    )
    print(f"Sustainer ignition tilt: {stagingtilt:.2f} degrees from vertical")

    if (
        sustainer_ignition_max_tilt_deg is not None
        and stagingtilt > sustainer_ignition_max_tilt_deg
    ):
        print(
            f"Sustainer ignition locked out: {stagingtilt:.2f} deg exceeds "
            f"{sustainer_ignition_max_tilt_deg:.2f} deg",
            flush=True,
        )
        lockout_sustainer = deepcopy(SustainerNOMOTOR)
        lockout_sustainer.parachutes = deepcopy(sustainer.parachutes)
        lockout_flight = Flight(
            rocket=lockout_sustainer,
            environment=environment,
            initial_solution=comparison_flights[-1],
            rail_length=0.01,
            inclination=comparison_flights[-1].attitude_angle(ignitiontime),
            heading=heading,
            max_time=time_limit,
            max_time_step=max_time_step,
            rtol=rtol,
            atol=atol,
            time_overshoot=True,
            name=f"{getattr(sustainer, 'name', 'Sustainer')} tilt lockout",
        )
        result = FullStackSimulationResult(
            lockout_flight.solution,
            (*comparison_flights, lockout_flight),
            stagingtilt,
            ignitiontime,
            False,
            True,
        )
        runfullstacksim.last_flights = result.flights
        runfullstacksim.staging_tilt = result.staging_tilt
        runfullstacksim.ignition_time = result.ignition_time
        runfullstacksim.sustainer_ignited = result.sustainer_ignited
        runfullstacksim.tilt_lockout_triggered = result.tilt_lockout_triggered
        return result if return_details else result.solution

    sustainerstartcondition = ignition_state
    sustainerstartcondition[0] = 0

    SustainerFlight2 = Flight(
        rocket=sustainer,
        environment=environment,
        initial_solution=sustainerstartcondition,
        rail_length=0.01,
        inclination=StackFlight2.attitude_angle(stagingtime),
        heading=heading,
        max_time=time_limit - ignitiontime,
        max_time_step=max_time_step,
        rtol=rtol,
        atol=atol,
        time_overshoot=True,
        name=f"{getattr(sustainer, 'name', 'Sustainer')} powered",
    )

    # The motor must simulate on a zero-based clock. Shift the completed
    # powered flight back to its absolute ignition time so CompareFlights uses
    # the same launch-time axis for boost, coast, and sustainer traces.
    for entry in SustainerFlight2.solution:
        entry[0] += ignitiontime
    SustainerFlight2.t_initial += ignitiontime
    SustainerFlight2.t += ignitiontime
    SustainerFlight2.t_final += ignitiontime
    SustainerFlight2.max_time += ignitiontime
    SustainerFlight2.out_of_rail_time += ignitiontime
    apogee_time = getattr(SustainerFlight2, "apogee_time", None)
    if apogee_time is not None and apogee_time > 0:
        SustainerFlight2.apogee_time += ignitiontime
    impact_time = getattr(SustainerFlight2, "impact_time", None)
    if impact_time is not None:
        SustainerFlight2.impact_time += ignitiontime

    result = FullStackSimulationResult(
        SustainerFlight2.solution,
        (*comparison_flights, SustainerFlight2),
        stagingtilt,
        ignitiontime,
        True,
        False,
    )
    runfullstacksim.last_flights = result.flights
    runfullstacksim.staging_tilt = result.staging_tilt
    runfullstacksim.ignition_time = result.ignition_time
    runfullstacksim.sustainer_ignited = result.sustainer_ignited
    runfullstacksim.tilt_lockout_triggered = result.tilt_lockout_triggered
    return result if return_details else result.solution


run_full_stack_simulation = runfullstacksim
runfullstacksim.last_flights = ()
runfullstacksim.staging_tilt = None
runfullstacksim.ignition_time = None
runfullstacksim.sustainer_ignited = False
runfullstacksim.tilt_lockout_triggered = False

__all__ = [
    "FullStackSimulationResult",
    "run_full_stack_simulation",
    "run_single_simulation",
    "runfullstacksim",
]
