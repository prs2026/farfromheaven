"""Utilities for running a single RocketPy simulation."""

from __future__ import annotations

from rocketpy import Environment, Flight, Rocket


def run_single_simulation(
    rocket: Rocket,
    environment: Environment,
    *,
    rail_length: float = 5.2,
    inclination: float = 90.0,
    heading: float = 0.0,
) -> Flight:
    """Run one simulation and return its RocketPy ``Flight`` object.

    Parameters
    ----------
    rocket
        A fully configured RocketPy rocket.
    environment
        The RocketPy environment in which the rocket will fly.
    rail_length
        Launch rail length in metres.
    inclination
        Launch rail inclination in degrees above the horizontal.
    heading
        Launch heading in degrees clockwise from north.
    """
    return Flight(
        rocket=rocket,
        environment=environment,
        rail_length=rail_length,
        inclination=inclination,
        heading=heading,
    )
