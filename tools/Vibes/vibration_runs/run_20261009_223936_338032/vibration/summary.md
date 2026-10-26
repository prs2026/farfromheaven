# PRELIMINARY SCREENING ESTIMATES — NOT VALIDATED FLIGHT DATA

Trajectory: `C:\Users\Ejfso\Documents\GitHub\projectblaze\sims\Vibes\vibration_runs\run_20261009_223936_338032\vibration_input.csv` (measured)

Maximum Mach: **4.868**  
Maximum dynamic pressure: **1.2591e+06 Pa**

| Regime | Duration (s) | ~1 s windows | Motor (g RMS) | Aero (g RMS) | Combined (g RMS) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ignition rail release | 0.000 | 0 | unavailable | unavailable | unavailable |
| powered ascent | 0.000 | 0 | unavailable | unavailable | unavailable |
| maximum dynamic pressure | 0.900 | 1 | 0 | unavailable | unavailable |
| mach 6 5 segment | 0.000 | 0 | unavailable | unavailable | unavailable |
| motor cutoff | 0.000 | 0 | unavailable | unavailable | unavailable |
| coast | 0.000 | 0 | unavailable | unavailable | unavailable |

## Provenance, assumptions, and missing inputs

- Trajectory: measured input CSV; not independently validated.
- Motor reference PSD: missing.
- Structural transfer function: missing.
- Cq = 0.02; dynamic viscosity = 1.8e-05 Pa·s (constant assumption).
- Motor mass = 2 kg, avionics mass = 0.1 kg, structural damping = 0.99.
- Motor scaling holds motor mass fixed and uses T·Ve/m ratio; transfer and motor reference must match the avionics location/axis.
- Regimes may overlap by design. Envelopes are the frequency-by-frequency maximum of window-averaged PSDs, not a single simultaneous time history.
- Blank CSV PSD cells mean unavailable, not zero. g RMS integrates only the requested frequency band.

## Warnings

- PRELIMINARY SCREENING ESTIMATES ONLY; not validated flight data or qualification levels.
- No pressure-to-acceleration transfer function: aerodynamic acceleration PSD and powered/coast total PSD with nonzero aerodynamic excitation are unavailable. A structural model or measurement is required.
- Avionics mass and structural damping are recorded only; their effects must be embodied in the supplied transfer function. No structural modes are invented.
- The pressure spectrum uses an assumed normalized Lorentzian attached-flow shape; it is not a measured or digitized NASA spectrum and excludes local shock/buffet effects.
- The flat-plate boundary-layer equations are low-fidelity at high Mach, near separation, shocks, fins, or protuberances; Mach 6.5 predictions require local validation.
- One-second trajectory windows are model averages, not PSD estimates from measured acceleration time histories. Separate regime envelopes do not represent statistical flight-to-flight variation.
- The ignition/rail-release regime is approximated as the first configured interval after positive thrust; actual rail-release time is not available in the trajectory CSV.
- No trajectory samples fell within Mach 6.5 ± 0.2.
- Motor cutoff could not be located because the trajectory has no off-thrust samples after powered flight.

## Method references

- [MIL-STD-810H Method 514.8 Annex D Category 19](https://milstd.net/pdf/download/MILSTD_810H.pdf)
- [NASA-HDBK-7005, Dynamic Environmental Criteria](https://www.vibrationdata.com/tutorials2/NASA7005.pdf)
- [MSFC-STD-3676B, Section 5.2](https://standards.nasa.gov/sites/default/files/standards/MSFC/B/0/msfc-std-3676b.pdf)
- [NASA TN D-1836](https://ntrs.nasa.gov/citations/19630012009)
