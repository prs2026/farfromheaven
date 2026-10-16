"""Preliminary, unvalidated rocket vibration screening from a trajectory CSV.

This is not flight data, a qualification spectrum, or a motor vibration model.
Run ``python preliminary_vibration.py --help`` for inputs and options.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid


TRAJECTORY_COLUMNS = (
    "time_s", "altitude_m", "mach", "velocity_mps", "air_density_kgm3",
    "thrust_N", "exhaust_velocity_mps",
)
REGIMES = (
    "ignition_rail_release", "powered_ascent", "maximum_dynamic_pressure",
    "mach_6_5_segment", "motor_cutoff", "coast",
)
REFERENCES = {
    "MIL-STD-810H Method 514.8 Annex D Category 19":
        "https://milstd.net/pdf/download/MILSTD_810H.pdf",
    "NASA-HDBK-7005, Dynamic Environmental Criteria":
        "https://www.vibrationdata.com/tutorials2/NASA7005.pdf",
    "MSFC-STD-3676B, Section 5.2":
        "https://standards.nasa.gov/sites/default/files/standards/MSFC/B/0/msfc-std-3676b.pdf",
    "NASA TN D-1836": "https://ntrs.nasa.gov/citations/19630012009",
}


def _positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def _nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return number


def _read_numeric_csv(path: Path, columns: tuple[str, ...],
                      min_rows: int = 2) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
    if len(frame) < min_rows:
        raise ValueError(f"{path}: at least {min_rows} row(s) required")
    result = frame.loc[:, list(columns)].copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: values must be finite and not missing")
    return result


def _read_curve(path: Path, value_column: str,
                allow_single: bool = False) -> tuple[np.ndarray, np.ndarray]:
    frame = _read_numeric_csv(path, ("frequency_Hz", value_column),
                              min_rows=1 if allow_single else 2)
    frequency = frame["frequency_Hz"].to_numpy(dtype=float)
    values = frame[value_column].to_numpy(dtype=float)
    if np.any(frequency <= 0) or np.any(np.diff(frequency) <= 0):
        raise ValueError(f"{path}: frequencies must be positive and strictly increasing")
    if np.any(values < 0):
        raise ValueError(f"{path}: {value_column} cannot be negative")
    return frequency, values


def _interpolate_curve(curve: tuple[np.ndarray, np.ndarray], frequency: np.ndarray,
                       name: str) -> np.ndarray:
    source_frequency, values = curve
    if frequency[0] < source_frequency[0] or frequency[-1] > source_frequency[-1]:
        raise ValueError(
            f"{name} covers {source_frequency[0]:g}–{source_frequency[-1]:g} Hz; "
            f"it must cover the requested {frequency[0]:g}–{frequency[-1]:g} Hz band"
        )
    # Linear PSD/magnitude interpolation is explicit; no extrapolation is allowed.
    return np.interp(frequency, source_frequency, values)


def _read_trajectory(path: Path) -> pd.DataFrame:
    frame = _read_numeric_csv(path, TRAJECTORY_COLUMNS)
    t = frame["time_s"].to_numpy()
    if np.any(np.diff(t) <= 0):
        raise ValueError("trajectory time_s must be strictly increasing")
    for column in ("altitude_m", "mach", "velocity_mps", "air_density_kgm3",
                   "thrust_N", "exhaust_velocity_mps"):
        if np.any(frame[column].to_numpy() < 0):
            raise ValueError(f"trajectory {column} cannot be negative")
    if np.any((frame["thrust_N"] > 0) & (frame["exhaust_velocity_mps"] <= 0)):
        raise ValueError("positive thrust requires positive exhaust_velocity_mps")
    return frame


def _pressure_psd(frequency: np.ndarray, density: np.ndarray,
                  velocity: np.ndarray, local_length: float,
                  viscosity: float, cq: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Lorentzian screening shape normalized over 0..infinity.

    This assumed shape is not a digitization of NASA-HDBK-7005 Figure 4.3.
    Its integral is p_rms**2. For zero-speed samples the PSD is exactly zero.
    """
    # q = rho*U^2/2; p_rms = Cq*q. At a particular station x, the supplied
    # flat-plate estimates are Re_x=rho*U*x/mu, delta=.37*x*Re_x^(-.2),
    # delta*=delta/8, and characteristic f0=.1*U/delta*.
    dynamic_pressure = 0.5 * density * velocity**2
    pressure_rms = cq * dynamic_pressure
    reynolds = density * velocity * local_length / viscosity
    active = (reynolds > 0) & (velocity > 0)
    f0 = np.ones_like(velocity)
    delta = 0.37 * local_length * reynolds[active] ** (-0.2)
    delta_star = delta / 8.0
    f0[active] = 0.1 * velocity[active] / delta_star
    # phi(f)=2/[pi*f0*(1+(f/f0)^2)] for f>=0 integrates to one.
    # Sp(f)=p_rms^2*phi(f) has units Pa^2/Hz.
    spectrum = np.zeros((len(velocity), len(frequency)), dtype=float)
    spectrum[active] = (
        pressure_rms[active, None] ** 2
        * 2.0 / (np.pi * f0[active, None])
        / (1.0 + (frequency[None, :] / f0[active, None]) ** 2)
    )
    return dynamic_pressure, spectrum


def _motor_psd(thrust: np.ndarray, exhaust_velocity: np.ndarray,
               motor_mass: float, reference: np.ndarray | None,
               reference_thrust: float | None, reference_ve: float | None,
               reference_mass: float | None, frequency_count: int) -> np.ndarray:
    result = np.zeros((len(thrust), frequency_count), dtype=float)
    active = thrust > 0
    if not np.any(active):
        return result
    if reference is None:
        result[active] = np.nan  # Powered vibration is unknown, not zero.
        return result
    assert reference_thrust is not None and reference_ve is not None
    assert reference_mass is not None
    # MSFC-STD-3676B scales acceleration amplitude as (T*Ve/W).
    # W=m*g and g cancels between the new and reference ratios, so masses
    # may be used here. PSD scales as the square of acceleration amplitude.
    scale = ((thrust[active] * exhaust_velocity[active] / motor_mass)
             / (reference_thrust * reference_ve / reference_mass))
    result[active] = reference[None, :] * scale[:, None] ** 2
    return result


def _window_spectra(times: np.ndarray, trajectory: pd.DataFrame,
                    frequency: np.ndarray, args: argparse.Namespace,
                    motor_reference: np.ndarray | None,
                    transfer: np.ndarray | None) -> dict[str, np.ndarray | float]:
    source_time = trajectory["time_s"].to_numpy(dtype=float)
    values = {
        name: np.interp(times, source_time, trajectory[name].to_numpy(dtype=float))
        for name in TRAJECTORY_COLUMNS if name != "time_s"
    }
    q, pressure = _pressure_psd(
        frequency, values["air_density_kgm3"], values["velocity_mps"],
        args.station_from_nose_m, args.dynamic_viscosity_pa_s, args.cq,
    )
    # H_ap is acceleration amplitude in g per Pa; squaring gives g^2/Pa^2.
    # Without an H_ap measurement/model, acceleration from pressure is unknown.
    aero = (pressure * transfer[None, :] ** 2 if transfer is not None
            else np.where(pressure == 0, 0.0, np.nan))
    motor = _motor_psd(
        values["thrust_N"], values["exhaust_velocity_mps"], args.motor_mass_kg,
        motor_reference, args.reference_thrust_n, args.reference_exhaust_velocity_mps,
        args.reference_motor_mass_kg, len(frequency),
    )
    # Independent broadband sources combine in power, not acceleration amplitude.
    # NaN propagates whenever a required source is unknown.
    combined = motor + aero
    duration = times[-1] - times[0]
    return {
        "motor": trapezoid(motor, times, axis=0) / duration,
        "aero": trapezoid(aero, times, axis=0) / duration,
        "combined": trapezoid(combined, times, axis=0) / duration,
        "pressure": trapezoid(pressure, times, axis=0) / duration,
        "max_q_pa": float(np.max(q)),
        "max_mach": float(np.max(values["mach"])),
    }


def _regime_masks(grid: np.ndarray, trajectory: pd.DataFrame,
                  args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, float | None]]:
    t = trajectory["time_s"].to_numpy(dtype=float)
    thrust = np.interp(grid, t, trajectory["thrust_N"])
    mach = np.interp(grid, t, trajectory["mach"])
    density = np.interp(grid, t, trajectory["air_density_kgm3"])
    velocity = np.interp(grid, t, trajectory["velocity_mps"])
    q = 0.5 * density * velocity**2
    powered = thrust > args.thrust_threshold_n
    if np.any(powered):
        ignition = float(grid[np.flatnonzero(powered)[0]])
        last_powered = int(np.flatnonzero(powered)[-1])
        cutoff = float(grid[last_powered + 1]) if last_powered + 1 < len(grid) else None
    else:
        ignition = None
        cutoff = None
    max_q_time = float(grid[int(np.argmax(q))])

    def near(center: float | None, width: float) -> np.ndarray:
        if center is None:
            return np.zeros_like(grid, dtype=bool)
        return (grid >= center - width / 2) & (grid < center + width / 2)

    masks = {
        "ignition_rail_release": (
            (grid >= ignition) & (grid < ignition + args.ignition_window_s)
            if ignition is not None else np.zeros_like(grid, dtype=bool)
        ),
        "powered_ascent": powered,
        "maximum_dynamic_pressure": near(max_q_time, args.event_window_s),
        "mach_6_5_segment": np.abs(mach - args.mach_target) <= args.mach_tolerance,
        "motor_cutoff": near(cutoff, args.event_window_s),
        "coast": ((~powered) & (grid > ignition)
                  if ignition is not None else np.zeros_like(grid, dtype=bool)),
    }
    return masks, {"ignition_s": ignition, "cutoff_s": cutoff,
                   "max_q_s": max_q_time}


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return list(zip(starts.tolist(), ends.tolist()))


def _rms(psd: np.ndarray, frequency: np.ndarray) -> float | None:
    if not np.isfinite(psd).all():
        return None
    # For a one-sided acceleration PSD, variance in g^2 is integral Sa(f) df.
    return float(np.sqrt(max(0.0, trapezoid(psd, frequency))))


def _save_plot(path: Path, regime: str, frequency: np.ndarray,
               envelope: dict[str, np.ndarray],
               tones: tuple[np.ndarray, np.ndarray] | None, powered: bool) -> None:
    rows = 3 if tones is not None and powered else 2
    fig, axes = plt.subplots(rows, 1, figsize=(10, 8 if rows == 2 else 10),
                             constrained_layout=True)
    fig.suptitle(f"PRELIMINARY SCREENING ESTIMATE — {regime.replace('_', ' ')}")
    ax = axes[0]
    for key, color, label in (
        ("motor", "#d97706", "Motor: scaled reference"),
        ("aero", "#0284c7", "Aerodynamic: transfer estimate"),
        ("combined", "#111827", "Combined independent broadband"),
    ):
        values = envelope[key]
        positive = np.where(np.isfinite(values) & (values > 0), values, np.nan)
        if np.isfinite(positive).any():
            ax.loglog(frequency, positive, color=color, label=label,
                      linewidth=2 if key == "combined" else 1.4)
    ax.set(xlabel="Frequency (Hz)", ylabel="Acceleration PSD (g²/Hz)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(frequency[0], frequency[-1])
    ax.grid(True, which="both", alpha=0.25)
    if ax.lines:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No acceleration PSD available for this regime",
                transform=ax.transAxes, ha="center")
        ax.set_ylim(1e-9, 1)
    ax = axes[1]
    pressure = np.where(envelope["pressure"] > 0, envelope["pressure"], np.nan)
    if np.isfinite(pressure).any():
        ax.loglog(frequency, pressure, color="#7c3aed")
        positive_pressure = pressure[np.isfinite(pressure)]
        ax.set_ylim(positive_pressure.min() / 10, positive_pressure.max() * 10)
    else:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-9, 1)
        ax.text(0.5, 0.5, "No aerodynamic pressure excitation",
                transform=ax.transAxes, ha="center")
    ax.set(xlabel="Frequency (Hz)", ylabel="Pressure PSD (Pa²/Hz)")
    ax.set_xlim(frequency[0], frequency[-1])
    ax.grid(True, which="both", alpha=0.25)
    if rows == 3:
        ax = axes[2]
        ax.stem(tones[0], tones[1], basefmt=" ", linefmt="#b91c1c", markerfmt="o")
        ax.set_xscale("log")
        ax.set(xlabel="Frequency (Hz)", ylabel="Separate motor tones (g RMS)")
        ax.grid(True, which="both", alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _format(value: float | None, unit: str = "") -> str:
    return "unavailable" if value is None else f"{value:.5g}{unit}"


def run(args: argparse.Namespace) -> Path:
    trajectory = _read_trajectory(args.trajectory)
    if args.station_from_nose_m > args.vehicle_length_m:
        raise ValueError("station-from-nose-m must not exceed vehicle-length-m")
    if not 0 < args.structural_damping_ratio < 1:
        raise ValueError("structural-damping-ratio must be between 0 and 1")
    if args.frequency_max_hz <= args.frequency_min_hz:
        raise ValueError("frequency-max-hz must exceed frequency-min-hz")
    if args.frequency_points < 2:
        raise ValueError("frequency-points must be at least 2")
    if args.mach_tolerance <= 0:
        raise ValueError("mach-tolerance must be positive")
    if args.motor_psd is None and any(x is not None for x in (
        args.reference_thrust_n, args.reference_exhaust_velocity_mps,
        args.reference_motor_mass_kg,
    )):
        raise ValueError("reference motor parameters require --motor-psd")
    if args.motor_psd is not None and any(x is None for x in (
        args.reference_thrust_n, args.reference_exhaust_velocity_mps,
        args.reference_motor_mass_kg,
    )):
        raise ValueError("--motor-psd requires reference thrust, exhaust velocity, and motor mass")

    frequency = np.geomspace(args.frequency_min_hz, args.frequency_max_hz,
                             args.frequency_points)
    reference = (
        _interpolate_curve(_read_curve(args.motor_psd, "acceleration_psd_g2_per_Hz"),
                           frequency, "motor reference PSD")
        if args.motor_psd is not None else None
    )
    transfer = (
        _interpolate_curve(_read_curve(args.transfer_function,
                                       "acceleration_per_pressure_g_per_Pa"),
                           frequency, "pressure-to-acceleration transfer function")
        if args.transfer_function is not None else None
    )
    tones = None
    if args.motor_tones is not None:
        tone_data = _read_curve(args.motor_tones, "acceleration_rms_g",
                                allow_single=True)
        tones = tone_data

    t = trajectory["time_s"].to_numpy(dtype=float)
    grid = np.arange(t[0], t[-1], args.sample_step_s)
    grid = np.unique(np.r_[grid, t, t[-1]])
    if len(grid) > 100_000:
        raise ValueError("time grid exceeds 100,000 samples; increase --sample-step-s")
    masks, events = _regime_masks(grid, trajectory, args)
    grid_q = (0.5 * np.interp(grid, t, trajectory["air_density_kgm3"])
              * np.interp(grid, t, trajectory["velocity_mps"]) ** 2)
    warnings = ["PRELIMINARY SCREENING ESTIMATES ONLY; not validated flight data or qualification levels."]
    if args.cq > 0.02:
        warnings.append("Cq > 0.02 assumes separated flow, shock interaction, or local protuberances; location-specific data are needed.")
    if args.cq != 0.02:
        warnings.append(f"Cq={args.cq:g} is user-selected; 0.02 is the attached-flow default.")
    if transfer is None:
        warnings.append("No pressure-to-acceleration transfer function: aerodynamic acceleration PSD and powered/coast total PSD with nonzero aerodynamic excitation are unavailable. A structural model or measurement is required.")
    if reference is None and np.any(trajectory["thrust_N"] > 0):
        warnings.append("No reference motor PSD: powered motor vibration and powered total PSD cannot be reliably estimated from thrust alone.")
    if reference is not None:
        warnings.append("Motor scaling assumes the reference PSD is measured at the same location, mounting, and axis; differences require a validated transfer model.")
        warnings.append("Motor mass is only a proxy for the effective vibrating motor weight in MSFC-STD-3676B; this scaling is low-fidelity for dissimilar motor cases or structural paths.")
    warnings.append("Avionics mass and structural damping are recorded only; their effects must be embodied in the supplied transfer function. No structural modes are invented.")
    warnings.append("The pressure spectrum uses an assumed normalized Lorentzian attached-flow shape; it is not a measured or digitized NASA spectrum and excludes local shock/buffet effects.")
    warnings.append("The flat-plate boundary-layer equations are low-fidelity at high Mach, near separation, shocks, fins, or protuberances; Mach 6.5 predictions require local validation.")
    warnings.append("One-second trajectory windows are model averages, not PSD estimates from measured acceleration time histories. Separate regime envelopes do not represent statistical flight-to-flight variation.")
    warnings.append("The ignition/rail-release regime is approximated as the first configured interval after positive thrust; actual rail-release time is not available in the trajectory CSV.")
    if tones is not None:
        warnings.append("Supplied motor tones are separate fixed RMS lines at the avionics location; they are not added to the broadband PSD or g RMS.")

    output = args.output_dir or args.trajectory.with_name(
        f"{args.trajectory.stem}_vibration_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    if tones is not None:
        pd.DataFrame({"frequency_Hz": tones[0], "acceleration_rms_g": tones[1],
                      "estimate_classification": "USER-SUPPLIED SEPARATE TONES; NOT BROADBAND PSD"}).to_csv(
            output / "motor_tones_separate.csv", index=False
        )

    summary = {
        "classification": "PRELIMINARY SCREENING ESTIMATES; NOT VALIDATED FLIGHT DATA",
        "inputs": {
            "trajectory": str(args.trajectory.resolve()),
            "trajectory_provenance": args.trajectory_provenance,
            "motor_psd": str(args.motor_psd.resolve()) if args.motor_psd else None,
            "motor_psd_provenance": args.reference_psd_provenance if args.motor_psd else None,
            "transfer_function": str(args.transfer_function.resolve()) if args.transfer_function else None,
            "transfer_provenance": args.transfer_provenance if args.transfer_function else None,
            "motor_tones": str(args.motor_tones.resolve()) if args.motor_tones else None,
            "vehicle_length_m": args.vehicle_length_m,
            "station_from_nose_m": args.station_from_nose_m,
            "motor_mass_kg": args.motor_mass_kg,
            "avionics_mass_kg": args.avionics_mass_kg,
            "structural_damping_ratio": args.structural_damping_ratio,
            "dynamic_viscosity_pa_s_assumed": args.dynamic_viscosity_pa_s,
            "cq": args.cq,
            "frequency_band_hz": [args.frequency_min_hz, args.frequency_max_hz],
            "window_s": args.window_s,
            "mach_target": args.mach_target,
            "mach_tolerance": args.mach_tolerance,
        },
        "maximum_mach": float(trajectory["mach"].max()),
        "maximum_dynamic_pressure_pa": float(grid_q.max()),
        "event_times_s": events,
        "warnings": warnings,
        "regimes": {},
        "references": REFERENCES,
    }
    for regime in REGIMES:
        mask = masks[regime]
        windows = []
        exposure_s = 0.0
        for start_index, end_index in _segments(mask):
            start = float(grid[start_index])
            end = float(grid[end_index])
            if end_index < len(grid) - 1:
                end = min(float(grid[end_index + 1]), t[-1])
            if end <= start:
                continue
            exposure_s += end - start
            for left in np.arange(start, end, args.window_s):
                right = min(float(left + args.window_s), end)
                sample_times = np.linspace(left, right,
                    max(2, int(np.ceil((right - left) / args.sample_step_s)) + 1))
                spectra = _window_spectra(sample_times, trajectory, frequency,
                                           args, reference, transfer)
                windows.append((float(left), right, spectra))

        envelope = {}
        for key in ("motor", "aero", "combined", "pressure"):
            if windows:
                stack = np.stack([w[2][key] for w in windows])
                # A source is available only if every contributing window is known.
                envelope[key] = np.max(stack, axis=0) if np.isfinite(stack).all() else np.full_like(frequency, np.nan)
            else:
                envelope[key] = np.full_like(frequency, np.nan)
        pd.DataFrame({
            "estimate_classification": "PRELIMINARY SCREENING ESTIMATE; NOT VALIDATED FLIGHT DATA",
            "frequency_Hz": frequency,
            "motor_psd_g2_per_Hz": envelope["motor"],
            "aerodynamic_psd_g2_per_Hz": envelope["aero"],
            "combined_psd_g2_per_Hz": envelope["combined"],
            "aerodynamic_pressure_psd_Pa2_per_Hz": envelope["pressure"],
        }).to_csv(output / f"{regime}_envelope.csv", index=False, na_rep="")
        pd.DataFrame([
            {
                "estimate_classification": "PRELIMINARY SCREENING ESTIMATE; NOT VALIDATED FLIGHT DATA",
                "window_start_s": left, "window_end_s": right,
                "window_duration_s": right - left,
                "maximum_q_pa": data["max_q_pa"],
                "maximum_mach": data["max_mach"],
                "motor_rms_g": _rms(data["motor"], frequency),
                "aerodynamic_rms_g": _rms(data["aero"], frequency),
                "combined_rms_g": _rms(data["combined"], frequency),
            }
            for left, right, data in windows
        ], columns=("estimate_classification", "window_start_s", "window_end_s", "window_duration_s",
                   "maximum_q_pa", "maximum_mach", "motor_rms_g",
                   "aerodynamic_rms_g", "combined_rms_g")
        ).to_csv(output / f"{regime}_windows.csv", index=False, na_rep="")
        has_power = bool(np.any(mask & (np.interp(grid, t, trajectory["thrust_N"]) > args.thrust_threshold_n)))
        _save_plot(output / f"{regime}_psd.png", regime, frequency, envelope,
                   tones, has_power)
        summary["regimes"][regime] = {
            "duration_s": exposure_s,
            "one_second_windows": len(windows),
            "motor_rms_g": _rms(envelope["motor"], frequency),
            "aerodynamic_rms_g": _rms(envelope["aero"], frequency),
            "combined_rms_g": _rms(envelope["combined"], frequency),
            "complete_acceleration_psd": bool(np.isfinite(envelope["combined"]).all()),
        }
    if summary["regimes"]["mach_6_5_segment"]["duration_s"] == 0:
        warnings.append(f"No trajectory samples fell within Mach {args.mach_target:g} ± {args.mach_tolerance:g}.")
    if events["cutoff_s"] is None:
        warnings.append("Motor cutoff could not be located because the trajectory has no off-thrust samples after powered flight.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# PRELIMINARY SCREENING ESTIMATES — NOT VALIDATED FLIGHT DATA", "",
        f"Trajectory: `{args.trajectory.resolve()}` ({args.trajectory_provenance})", "",
        f"Maximum Mach: **{summary['maximum_mach']:.4g}**  ",
        f"Maximum dynamic pressure: **{summary['maximum_dynamic_pressure_pa']:.5g} Pa**", "",
        "| Regime | Duration (s) | ~1 s windows | Motor (g RMS) | Aero (g RMS) | Combined (g RMS) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for regime, result in summary["regimes"].items():
        lines.append(
            f"| {regime.replace('_', ' ')} | {result['duration_s']:.3f} | "
            f"{result['one_second_windows']} | {_format(result['motor_rms_g'])} | "
            f"{_format(result['aerodynamic_rms_g'])} | {_format(result['combined_rms_g'])} |"
        )
    lines += ["", "## Provenance, assumptions, and missing inputs", "",
              f"- Trajectory: {args.trajectory_provenance} input CSV; not independently validated.",
              f"- Motor reference PSD: {args.reference_psd_provenance if reference is not None else 'missing'}.",
              f"- Structural transfer function: {args.transfer_provenance if transfer is not None else 'missing'}.",
              f"- Cq = {args.cq:g}; dynamic viscosity = {args.dynamic_viscosity_pa_s:g} Pa·s (constant assumption).",
              f"- Motor mass = {args.motor_mass_kg:g} kg, avionics mass = {args.avionics_mass_kg:g} kg, structural damping = {args.structural_damping_ratio:g}.",
              "- Motor scaling holds motor mass fixed and uses T·Ve/m ratio; transfer and motor reference must match the avionics location/axis.",
              "- Regimes may overlap by design. Envelopes are the frequency-by-frequency maximum of window-averaged PSDs, not a single simultaneous time history.",
              "- Blank CSV PSD cells mean unavailable, not zero. g RMS integrates only the requested frequency band.",
              "", "## Warnings", ""]
    lines += [f"- {message}" for message in warnings]
    lines += ["", "## Method references", ""]
    lines += [f"- [{title}]({url})" for title, url in REFERENCES.items()]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("trajectory", type=Path, help="CSV with required trajectory columns")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--trajectory-provenance", choices=("measured", "simulated", "unspecified"),
                   default="unspecified")
    p.add_argument("--vehicle-length-m", type=_positive, required=True)
    p.add_argument("--station-from-nose-m", type=_positive, required=True)
    p.add_argument("--motor-mass-kg", type=_positive, required=True)
    p.add_argument("--avionics-mass-kg", type=_positive, required=True)
    p.add_argument("--structural-damping-ratio", type=_positive, required=True)
    p.add_argument("--dynamic-viscosity-pa-s", type=_positive, default=1.8e-5)
    p.add_argument("--cq", type=_positive, choices=(0.006, 0.02, 0.05, 0.1), default=0.02)
    p.add_argument("--motor-psd", type=Path)
    p.add_argument("--reference-thrust-n", type=_positive)
    p.add_argument("--reference-exhaust-velocity-mps", type=_positive)
    p.add_argument("--reference-motor-mass-kg", type=_positive)
    p.add_argument("--reference-psd-provenance", choices=("measured", "estimated", "unknown"),
                   default="unknown")
    p.add_argument("--transfer-function", type=Path)
    p.add_argument("--transfer-provenance", choices=("measured", "modeled", "unknown"),
                   default="unknown")
    p.add_argument("--motor-tones", type=Path,
                   help="optional separate tones CSV: frequency_Hz,acceleration_rms_g")
    p.add_argument("--frequency-min-hz", type=_positive, default=20.0)
    p.add_argument("--frequency-max-hz", type=_positive, default=2000.0)
    p.add_argument("--frequency-points", type=int, default=256)
    p.add_argument("--window-s", type=_positive, default=1.0)
    p.add_argument("--sample-step-s", type=_positive, default=0.1)
    p.add_argument("--ignition-window-s", type=_positive, default=1.0)
    p.add_argument("--event-window-s", type=_positive, default=1.0)
    p.add_argument("--mach-target", type=_positive, default=6.5)
    p.add_argument("--mach-tolerance", type=_positive, default=0.2)
    p.add_argument("--thrust-threshold-n", type=_nonnegative, default=0.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        output = run(args)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"PRELIMINARY SCREENING ESTIMATES written to {output}")
    print(f"Read {output / 'summary.md'} before using the PSD files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
