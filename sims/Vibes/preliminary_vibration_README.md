# Preliminary rocket vibration screening

`preliminary_vibration.py` turns a trajectory into separate regime-by-regime
pressure and acceleration PSD screening estimates. It is **not flight data, a
qualified test spectrum, or a substitute for measured vibration and structural
response**. Its example trajectory is synthetic and is only for exercising the
tool.

Run a pressure-only example from the `rocketpy` directory:

```powershell
python .\preliminary_vibration.py .\vibration_example_trajectory.csv `
  --vehicle-length-m 3.3 --station-from-nose-m 1.2 `
  --motor-mass-kg 8 --avionics-mass-kg 1 `
  --structural-damping-ratio 0.03 `
  --trajectory-provenance simulated
```

The default output is a timestamped folder next to the trajectory. It contains
`summary.md`, `summary.json`, and an envelope CSV, window-metrics CSV, and PNG
plot for each of the six regimes. Missing acceleration components are blank in
CSV and marked unavailable in the report. Aerodynamic pressure PSD remains
available without a structural transfer function.

To estimate acceleration at the avionics mount, supply a transfer function CSV
with `frequency_Hz,acceleration_per_pressure_g_per_Pa`. The file must describe
the relevant mounting location, direction, avionics mass, and damping; the
script does not synthesize structural modes. If you also have a reference motor
PSD at a comparable location and axis, supply a CSV with
`frequency_Hz,acceleration_psd_g2_per_Hz` and all three reference motor
parameters:

```powershell
python .\preliminary_vibration.py .\vibration_example_trajectory.csv `
  --vehicle-length-m 3.3 --station-from-nose-m 1.2 `
  --motor-mass-kg 8 --avionics-mass-kg 1 `
  --structural-damping-ratio 0.03 `
  --transfer-function path\to\mount_transfer.csv `
  --motor-psd path\to\reference_motor_psd.csv `
  --reference-thrust-n 7000 --reference-exhaust-velocity-mps 2200 `
  --reference-motor-mass-kg 8 --reference-psd-provenance measured
```

Both spectrum files must span the requested frequency band (default 20–2000
Hz). The optional `--motor-tones` CSV has
`frequency_Hz,acceleration_rms_g`. These tones appear separately in plots and
`motor_tones_separate.csv`, and are excluded from broadband PSD and RMS.
Use `--help` for window, frequency, Mach, Cq, and viscosity options.

## Screening equations and interpretation

- Dynamic pressure is `q = 0.5 rho U²`, and assumed wall-pressure RMS is
  `p_rms = Cq q`. `Cq=0.02` is the default. Higher choices indicate possible
  separation, shock interaction, or local protuberances and need local data.
- At local station `x`, the script uses `Re_x = rho U x/mu`,
  `delta = 0.37 x Re_x^-0.2`, `delta_star = delta/8`, and
  `f0 = 0.1 U/delta_star`. A *user-independent assumed* one-sided Lorentzian
  pressure-spectrum shape is normalized so its integral over all positive
  frequencies equals `p_rms²`. It is **not** a digitization of a NASA curve.
- With a supplied acceleration-per-pressure magnitude `H_ap`, the aerodynamic
  contribution is `Sa_aero = |H_ap|² Sp`. Without it, this contribution is
  unavailable whenever aerodynamic pressure is nonzero.
- Given reference motor PSD and matching reference data, the amplitude ratio is
  `(T_new Ve_new/m_new)/(T_ref Ve_ref/m_ref)`, and motor PSD is multiplied by
  the ratio squared. Constant motor mass is assumed; gravity cancels from the
  weight ratio. No reference PSD means powered motor PSD is unavailable.
- Independent broadband PSDs add as `Sa_total = Sa_motor + Sa_aero` only when
  both are known. Each regime is divided into approximately one-second windows.
  Spectra are averaged within each window and enveloped frequency by frequency
  within that regime. Reported `g RMS` is the square root of the integral of
  the envelope over the requested band. Event regimes may overlap powered or
  coast regimes. These are screening envelopes, not simultaneous histories or
  statistically qualified environments.

The approach follows the emphasis on measured flight data in
[MIL-STD-810H Method 514.8 Category 19](https://milstd.net/pdf/download/MILSTD_810H.pdf),
the boundary-layer scales and PSD treatment in
[NASA-HDBK-7005](https://www.vibrationdata.com/tutorials2/NASA7005.pdf),
the motor scaling guidance in
[MSFC-STD-3676B](https://standards.nasa.gov/sites/default/files/standards/MSFC/B/0/msfc-std-3676b.pdf),
and the source-identification approach in
[NASA TN D-1836](https://ntrs.nasa.gov/citations/19630012009).
