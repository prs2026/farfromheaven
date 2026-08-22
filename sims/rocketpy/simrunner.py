"""Reusable RocketPy simulation helpers."""

from __future__ import annotations

import math
from copy import deepcopy

from rocketpy import EmptyMotor, Environment, Flight, GenericMotor, Rocket


def run_single_simulation(
    rocket: Rocket,
    environment: Environment,
    *,
    rail_length: float = 5.2,
    inclination: float = 90.0,
    heading: float = 0.0,
) -> Flight:
    """Run one simulation and return its RocketPy ``Flight`` object."""
    return Flight(
        rocket=rocket,
        environment=environment,
        rail_length=rail_length,
        inclination=inclination,
        heading=heading,
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
) -> list[list[float]]:
    """Run the boost and sustainer phases and return the original solution format.

    The booster separates at burnout. The sustainer then coasts without a
    motor for ``coast_period`` seconds before ignition. ``rod_angle`` is
    measured away from vertical. Returned times are absolute seconds since
    launch, matching the original ``runfullstacksim`` function.
    """
    time_limit = float(time_limit)
    coast_period = float(coast_period)
    rail_length = float(rail_length)
    rod_angle = float(rod_angle)
    heading = float(heading)
    max_time_step = float(max_time_step)
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
        rtol=1e-4,
        atol=1e-6,
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
        runfullstacksim.last_flights = (StackFlight2,)
        return StackFlight2.solution
    sustainer_motor = sustainer.motor
    coast_motor = GenericMotor(
        thrust_source=0,
        burn_time=max(ignitiontime, 1.0),
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
            rtol=1e-4,
            atol=1e-6,
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
            runfullstacksim.last_flights = (StackFlight2, SustainerNOMOTORFlight2)
            return SustainerNOMOTORFlight2.solution
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
        rtol=1e-4,
        atol=1e-6,
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

    runfullstacksim.last_flights = (*comparison_flights, SustainerFlight2)
    runfullstacksim.staging_tilt = stagingtilt
    runfullstacksim.ignition_time = ignitiontime
    return SustainerFlight2.solution


run_full_stack_simulation = runfullstacksim
runfullstacksim.last_flights = ()
runfullstacksim.staging_tilt = None
runfullstacksim.ignition_time = None
