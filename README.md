# Far From Heaven

Vehicle (flight) sims and motor sims are kept in separate trees:

- **`Vehicle Sims/`**: RASAero `.CDX1` and OpenRocket `.ork` designs, one folder
  per vehicle. The **current design sits at the top of each vehicle folder**, and
  superseded designs are in its `old/` subfolder.
  Each vehicle folder also holds the `.eng` file its current RASAero sim loads
  (`AMA.eng`, `EAYF.eng`).
- **`Motor Sims/`**: all other thrust curves (`.eng`) and openMotor files (`.ric`).
  Vehicle-specific motors are in a folder named after the vehicle; general-purpose
  curves are in `Library/`.
- **`tools/`**: simulation and analysis code.

## Current configs

| Vehicle | Current design | Engine file for the sim |
| --- | --- | --- |
| AMA | [`Vehicle Sims/AMA/AMA_CONFIG_15.CDX1`](Vehicle%20Sims/AMA/AMA_CONFIG_15.CDX1)<br>OpenRocket: [`AMA CONFIG 16 OR.ork`](Vehicle%20Sims/AMA/AMA%20CONFIG%2016%20OR.ork) | [`Vehicle Sims/AMA/AMA.eng`](Vehicle%20Sims/AMA/AMA.eng) |
| EYOF | [`Vehicle Sims/EYOF/EAYF_CONFIG_1.CDX1`](Vehicle%20Sims/EYOF/EAYF_CONFIG_1.CDX1) | [`Vehicle Sims/EYOF/EAYF.eng`](Vehicle%20Sims/EYOF/EAYF.eng) |

Earlier EYOF iterations live in `Vehicle Sims/EYOF/old/`.

## Layout

```text
farfromheaven/
├── Vehicle Sims/
│   ├── AMA/
│   │   ├── AMA_CONFIG_15.CDX1          current RASAero design
│   │   ├── AMA CONFIG 16 OR.ork        current OpenRocket design
│   │   ├── AMA.eng                     engine file the RASAero sim loads (N3364, O8220)
│   │   └── old/                        earlier CONFIG 15, AMA 4-3 Current Booster
│   ├── EYOF/
│   │   ├── EAYF_CONFIG_1.CDX1          current RASAero design
│   │   ├── EAYF.eng                    engine file the RASAero sim loads (N3051, M3140)
│   │   └── old/                        EAYF 3-3 and "attempt 2" iterations
│   │       └── summaries/              cdx1_summary PNGs of the old iterations
│   └── Other Vehicles/
│       ├── FFH 4-3/                    ABJ / Chase / RCS booster variants, FFH_2to4.ork
│       ├── Ember/
│       └── Test Vehicles/              30 Wildman, 31 Ejarok, 32 Hybrid
├── Motor Sims/
│   ├── EYOF/                           individual EAYF booster/sustainer curves
│   ├── Library/                        general-purpose .eng curves
│   ├── openMotor/                      openMotor .ric source files
│   └── old/
└── tools/
    ├── rocketpy/                       CDX1 → JSON → RocketPy sims and Monte Carlo
    ├── cdx1_summary/                   CDX1 dimension/mass summary PNG tool
    └── Vibes/                          vibration screening, Blue Raven trajectory export
```

## Adding a new config

1. Save the new design at the top of the vehicle folder (e.g. `Vehicle Sims/AMA/AMA_CONFIG_16.CDX1`).
2. Move the design it replaces into that vehicle's `old/`.
3. Keep the `.eng` file the sim loads beside it; put other motor curves in `Motor Sims/`.
4. Update the table above.

## Tools

See [`tools/rocketpy/README.md`](tools/rocketpy/README.md) for the simulation
workflow. The RocketPy rocket JSONs, aero CSVs and Monte Carlo outputs (which
include AMA CONFIG 13–15 inputs) stay inside `tools/rocketpy/` because their
configs reference each other by relative path.

The CDX1 summary launcher expects the virtual environment at `.venv` in the repo
root. Install dependencies with `pip install -r requirements.txt`.

## RASAero engine file

RASAero II remembers the absolute path of the last engine file loaded. If a
sim says it can't find a motor (or `Debugfile.txt`) after files move, reload the
engine file from its new location in RASAero, e.g. `Vehicle Sims/AMA/AMA.eng`.
