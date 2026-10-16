# Preliminary rocket vibration screening

`preliminary_vibration.py` turns a trajectory into separate regime-by-regime
pressure and acceleration PSD screening estimates. It is **not flight data, a
qualified test spectrum, or a substitute for measured vibration and structural
response**. Its example trajectory is synthetic and is only for exercising the
tool.

## Blue Raven + Open-Meteo desktop workflow

From the workspace root, launch the GUI:

```powershell
python .\Vibes\vibration_gui.py
```

The tabs guide you through a Blue Raven HR/LR pair (or existing standard
trajectory), launch coordinates and local hour, rocket geometry/mass, and
optional motor and structural inputs. Choose **Fetch weather only** to save a
vertical atmosphere CSV, or **Run complete screening** to convert telemetry,
fetch weather, prepare the vibration input, and run the PSD model. Each action
gets a new timestamped results directory. Network fetching runs in a background
thread so the window remains responsive.

The **All inputs / Export** tab updates as you edit the form and displays every
entered value, effective model defaults, and warnings about missing inputs.
Use **Export JSON** for a machine-readable snapshot or **Export Markdown** for
a readable input report, including before running the model. Each new run also
saves `run_inputs.json` and `run_inputs.md` in its results folder. These record
the settings captured when you clicked Run, then add the resolved file paths
and fetched weather metadata (including the selected hour). Changing the form
during a run does not change that run's settings. Exports contain settings and
file paths, not copies of the source CSVs; they are records, not GUI import files.

### Save and load a simulation (.RSF)

Use **Save RSF...** and **Load RSF...** at the top of the Tkinter window to save
and restore a setup. `.RSF` (Rocket Simulation File) is UTF-8 JSON, not a binary
file. It contains all form settings, CSV references, and the remaining PSD
model options/defaults. A draft with blank fields can also be saved. Loading
replaces the form after confirmation; it does not fetch weather or run a model.

Set **CSV inputs folder** on the Flight data tab to the folder holding your
CSVs. For example, keep `my_flight.RSF` beside a `csv_inputs` folder and select
that folder. File entries can be just `trajectory.csv`, `motor.csv`, etc.
Files in subfolders, outside that folder, or on another drive are also supported.
The GUI defaults to the existing `Vibes` folder so existing inputs still work.
Nothing is moved, copied, or embedded when saving.

RSF version 1 has these fields:

| Field | Meaning |
| --- | --- |
| `schema` | Must be `projectblaze-vibration-rsf`. |
| `version` | Integer `1`; unsupported versions are rejected. |
| `saved_at_utc` | Informational save timestamp. |
| `csv_inputs_folder` | Folder path relative to the RSF location, or an absolute path. |
| `csv_files` | `trajectory`, `hr`, `lr`, `motor_curve`, `transfer`, `motor_psd`; filenames relative to the CSV folder, or absolute paths. Blank means unset. |
| `settings` | All non-file form fields, stored as strings so blank/unfinished inputs can be restored. Units are those shown in the GUI (SI). `output_dir` is relative to the RSF location or absolute. |
| `model_options` | Typed JSON values for all other model options: frequency minimum/points, averaging and sampling intervals, event windows, Mach target/tolerance, thrust threshold, viscosity, reference/transfer provenance, and optional `motor_tones` CSV. The tones filename also uses the CSV folder. |

Every setting and model option is written explicitly, so loading does not
silently adopt new model defaults. The All inputs page shows the restored
effective values; advanced values in `model_options` can be edited in the JSON
and will be applied on the next run. Invalid schemas, missing keys, or malformed
option values are rejected before changing the form. Missing CSVs produce a
warning and can be fixed using the folder/filename controls.

Relative paths are resolved against the RSF file and its CSV folder, **not**
the directory from which Python was launched. Moving the RSF and its CSV folder
together preserves those references. The file stores weather request settings,
not the fetched atmosphere or simulation outputs: running still fetches
Open-Meteo data. Existing `run_inputs.json`/Markdown exports remain audit reports;
use `.RSF` when you want a loadable setup.

The Open-Meteo profile can also be fetched on the command line. It writes a
CSV, a vertical-profile PNG, and a metadata JSON:

```powershell
python .\Vibes\openmeteo_profile.py `
  --latitude 40.88 --longitude -119.08 `
  --local-time 2026-09-26T14:43 --timezone America/Los_Angeles `
  --mode historical --output .\Vibes\weather_profile.csv
```

Use `--mode forecast` with a future time, or `--mode auto` to choose based on
whether the requested local time is past or future. Open-Meteo is hourly, so
the script selects the nearest hour and records it in the metadata. Coordinates above are only
an example; enter the actual launch location. Historical mode uses Open-Meteo's
archived forecast service, not an observed radiosonde. The profile CSV contains
ASL altitude, pressure, temperature, dry-air density, sound speed, and east/north
wind. No out-of-range altitude extrapolation is performed.

To prepare the specific input required by the PSD script:

```powershell
python .\Vibes\prepare_vibration_trajectory.py `
  .\Vibes\blue_raven_output\flight_trajectory.csv `
  .\Vibes\weather_profile.csv `
  --end-time-s 23.3 `
  --output .\Vibes\vibration_input.csv
```

If a motor thrust curve is available, add `--motor-curve motor.csv
--exhaust-velocity-mps VALUE`; `motor.csv` must contain `time_s,thrust_N`, with
time relative to ignition. Use `--ignition-time-s` for a delayed start. Without
a curve, thrust remains zero and powered regimes cannot be identified. The
BOOSKA flight's inertial data need special caution: its gyro nears the sensor
limit at 1.77 s. The 23.3 s end-time above merely trims roughly at apogee; it
does **not** repair later inertial velocities. Validate against another source
before trusting Mach or dynamic pressure.

The bridge uses the trajectory's **ground speed** for Mach and pressure, since
Blue Raven downrange/crossrange axes are not georeferenced and the weather wind
cannot safely be subtracted. It assumes trajectory `z` is meters above the
launch point, as produced by `blue_raven_to_trajectory.py`. The weather profile
is held constant for the flight. These approximations are recorded in a
metadata JSON alongside the prepared CSV.

Then run the screening script with your actual geometry and mass values:

```powershell
python .\Vibes\preliminary_vibration.py .\Vibes\vibration_input.csv `
  --vehicle-length-m 3.3 --station-from-nose-m 1.2 `
  --motor-mass-kg 8 --avionics-mass-kg 1 `
  --structural-damping-ratio 0.03 `
  --trajectory-provenance measured
```

The numeric rocket values above are examples, not values from this flight.
Supply a suitable `--transfer-function` to obtain *aerodynamic acceleration*
PSD, and a comparable `--motor-psd` plus reference motor thrust, exhaust
velocity and mass to estimate *motor acceleration* PSD. Otherwise those
components remain unavailable; the tool never substitutes arbitrary levels.

## Direct example

Run a pressure-only example from the workspace root:

```powershell
python .\Vibes\preliminary_vibration.py .\Vibes\vibration_example_trajectory.csv `
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
python .\Vibes\preliminary_vibration.py .\Vibes\vibration_example_trajectory.csv `
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
