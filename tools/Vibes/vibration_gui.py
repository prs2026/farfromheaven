"""Desktop GUI for Blue Raven -> Open-Meteo -> preliminary vibration screening.

Run with: python Vibes/vibration_gui.py
Tkinter is part of the standard Python installation on most desktop systems.
"""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

if __package__:
    from .vibration_input_manifest import build_manifest, render_manifest, save_manifest
    from .vibration_workflow import run_workflow
    from .vibration_rsf import load_rsf, model_options, save_rsf
else:
    from vibration_input_manifest import build_manifest, render_manifest, save_manifest
    from vibration_workflow import run_workflow
    from vibration_rsf import load_rsf, model_options, save_rsf


HERE = Path(__file__).resolve().parent


class VibrationApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Preliminary Rocket Vibration Workflow")
        self.geometry("960x800")
        self.minsize(790, 650)
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.busy = False
        self.latest_output: Path | None = None
        self.vars: dict[str, tk.StringVar] = {}
        self.model_options = model_options()
        self.rsf_path: Path | None = None
        self.review_refresh_id: str | None = None
        self._build()
        self.after(100, self._drain_events)

    def _var(self, name: str, default: str = "") -> tk.StringVar:
        value = tk.StringVar(value=default)
        self.vars[name] = value
        return value

    def _entry(self, parent: ttk.Frame, row: int, label: str, name: str,
               default: str = "", browse: bool = False) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=5)
        entry = ttk.Entry(parent, textvariable=self._var(name, default))
        entry.grid(row=row, column=1, sticky="ew", padx=8, pady=5)
        if browse:
            ttk.Button(parent, text="Browse…", command=lambda: self._browse_file(name)).grid(
                row=row, column=2, padx=8, pady=5)

    def _browse_file(self, name: str) -> None:
        folder = Path(self.vars["csv_inputs_folder"].get() or ".").expanduser().resolve()
        path = filedialog.askopenfilename(initialdir=str(folder),
                                         filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            try:
                path = os.path.relpath(path, folder)
            except ValueError:  # A different Windows drive requires an absolute path.
                pass
            self.vars[name].set(path)

    def _build(self) -> None:
        file_controls = ttk.Frame(self, padding=(12, 6))
        file_controls.pack(fill="x")
        ttk.Button(file_controls, text="Load RSF...", command=self._load_rsf).pack(side="left")
        ttk.Button(file_controls, text="Save RSF...", command=self._save_rsf).pack(side="left", padx=8)
        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True, padx=12, pady=(12, 6))
        flight = ttk.Frame(tabs, padding=12)
        weather = ttk.Frame(tabs, padding=12)
        vibration = ttk.Frame(tabs, padding=12)
        optional = ttk.Frame(tabs, padding=12)
        review = ttk.Frame(tabs, padding=12)
        tabs.add(flight, text="1  Flight data")
        tabs.add(weather, text="2  Atmosphere")
        tabs.add(vibration, text="3  Rocket & motor")
        tabs.add(optional, text="4  PSD options")
        tabs.add(review, text="5  All inputs / Export")
        for tab in (flight, weather, vibration, optional, review):
            tab.columnconfigure(1, weight=1)

        ttk.Label(flight, text="Use an existing standard trajectory, or leave it empty to convert the Blue Raven pair.",
                  wraplength=820).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        self._entry(flight, 1, "Existing trajectory CSV", "trajectory", browse=True)
        default_hr = HERE / ".BOOSKA_RAVEN HR_09-26-2026_14_43_26.csv"
        default_lr = HERE / ".BOOSKA_RAVEN LR_09-26-2026_14_43_26.csv"
        self._entry(flight, 2, "Blue Raven HR CSV", "hr", str(default_hr) if default_hr.exists() else "", True)
        self._entry(flight, 3, "Blue Raven LR CSV", "lr", str(default_lr) if default_lr.exists() else "", True)
        self._entry(flight, 4, "Analysis end time (s)", "end_time")
        ttk.Label(flight, text="Trajectory provenance").grid(row=5, column=0, sticky="w", padx=8, pady=5)
        ttk.Combobox(flight, textvariable=self._var("provenance", "measured"),
                     values=("measured", "simulated", "unspecified"), state="readonly").grid(
                         row=5, column=1, sticky="ew", padx=8, pady=5)
        ttk.Label(flight, text="Leave blank for the full file. Trim segments with invalid inertial altitude; "
                  "the supplied BOOSKA flight reaches apogee near 23.3 s, but its gyro nears its limit at 1.77 s.",
                  wraplength=820, foreground="#9c4c00").grid(
                      row=6, column=0, columnspan=3, sticky="w", padx=8, pady=8)
        self._entry(flight, 7, "CSV inputs folder", "csv_inputs_folder", str(HERE))
        ttk.Button(flight, text="Browse...", command=self._browse_inputs).grid(row=7, column=2, padx=8)
        ttk.Label(flight, text="All relative CSV filenames use this folder (including motor and PSD inputs). "
                  "You can keep your files here or select a dedicated csv_inputs folder.",
                  wraplength=820).grid(row=8, column=0, columnspan=3, sticky="w", padx=8, pady=5)

        ttk.Label(weather, text="Enter the launch location and local hour. Historical uses archived forecasts; "
                  "forecast uses the current forecast. Weather is modeled, not measured at the rocket.",
                  wraplength=820).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        self._entry(weather, 1, "Latitude (degrees)", "latitude")
        self._entry(weather, 2, "Longitude (degrees)", "longitude")
        self._entry(weather, 3, "Local launch time (YYYY-MM-DDTHH:MM)", "local_time", "2026-09-26T14:43")
        self._entry(weather, 4, "IANA timezone", "timezone", "America/Los_Angeles")
        ttk.Label(weather, text="Weather mode").grid(row=5, column=0, sticky="w", padx=8, pady=5)
        ttk.Combobox(weather, textvariable=self._var("mode", "auto"),
                     values=("auto", "historical", "forecast"), state="readonly").grid(
                         row=5, column=1, sticky="ew", padx=8, pady=5)
        self._entry(weather, 6, "Site elevation override (m ASL)", "elevation")
        ttk.Label(weather, text="If elevation is blank, Open-Meteo's grid elevation is used. "
                  "The nearest available hour is selected and held fixed during the flight.",
                  wraplength=820).grid(row=7, column=0, columnspan=3, sticky="w", padx=8, pady=8)

        ttk.Label(vibration, text="Required geometry and mass inputs are not inferred from telemetry. "
                  "An optional motor curve is time_s,thrust_N CSV; its time is relative to ignition.",
                  wraplength=820).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        fields = [
            (1, "Vehicle length (m)", "length"),
            (2, "Avionics station from nose (m)", "station"),
            (3, "Motor mass (kg)", "motor_mass"),
            (4, "Avionics mass (kg)", "avionics_mass"),
            (5, "Structural damping ratio", "damping"),
        ]
        for row, label, name in fields:
            self._entry(vibration, row, label, name)
        ttk.Separator(vibration).grid(row=6, column=0, columnspan=3, sticky="ew", pady=8)
        self._entry(vibration, 7, "Motor thrust curve CSV (optional)", "motor_curve", browse=True)
        self._entry(vibration, 8, "Assumed exhaust velocity (m/s)", "exhaust_velocity")
        self._entry(vibration, 9, "Motor ignition time (s)", "ignition_time", "0")
        ttk.Label(optional, text="Without a structural transfer function, the model outputs aerodynamic pressure, "
                  "not aerodynamic acceleration. Without a comparable motor PSD, motor vibration remains unavailable.",
                  wraplength=820).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        self._entry(optional, 1, "Structural transfer CSV (optional)", "transfer", browse=True)
        self._entry(optional, 2, "Reference motor PSD CSV (optional)", "motor_psd", browse=True)
        self._entry(optional, 3, "Reference thrust (N)", "ref_thrust")
        self._entry(optional, 4, "Reference exhaust velocity (m/s)", "ref_ve")
        self._entry(optional, 5, "Reference motor mass (kg)", "ref_mass")
        ttk.Label(optional, text="Pressure coefficient Cq").grid(row=6, column=0, sticky="w", padx=8, pady=5)
        ttk.Combobox(optional, textvariable=self._var("cq", "0.02"),
                     values=("0.006", "0.02", "0.05", "0.1"), state="readonly").grid(
                         row=6, column=1, sticky="ew", padx=8, pady=5)
        self._entry(optional, 7, "Maximum PSD frequency (Hz)", "frequency_max", "2000")

        ttk.Label(review, text="Live snapshot of every entered value and effective model default. "
                  "Export it before running, or find run_inputs.json and run_inputs.md in every results folder.",
                  wraplength=820).pack(fill="x", pady=(0, 8))
        review_buttons = ttk.Frame(review)
        review_buttons.pack(fill="x", pady=(0, 8))
        ttk.Button(review_buttons, text="Refresh", command=self._refresh_review).pack(side="left")
        ttk.Button(review_buttons, text="Export JSON…",
                   command=lambda: self._export_inputs(".json")).pack(side="left", padx=8)
        ttk.Button(review_buttons, text="Export Markdown…",
                   command=lambda: self._export_inputs(".md")).pack(side="left")
        self.review_text = scrolledtext.ScrolledText(review, wrap="word", state="disabled")
        self.review_text.pack(fill="both", expand=True)

        controls = ttk.Frame(self, padding=(12, 4))
        controls.pack(fill="x")
        ttk.Label(controls, text="Output parent folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self._var("output_dir", str(HERE / "vibration_runs"))).grid(
            row=0, column=1, sticky="ew", padx=8)
        ttk.Button(controls, text="Browse…", command=self._browse_output).grid(row=0, column=2)
        controls.columnconfigure(1, weight=1)
        buttons = ttk.Frame(self, padding=(12, 2))
        buttons.pack(fill="x")
        self.weather_button = ttk.Button(buttons, text="Fetch weather only",
                                         command=lambda: self._start("weather"))
        self.weather_button.pack(side="left", padx=(0, 8))
        self.run_button = ttk.Button(buttons, text="Run complete screening",
                                     command=lambda: self._start("complete"))
        self.run_button.pack(side="left")
        ttk.Button(buttons, text="Open results folder", command=self._open_results).pack(side="right")
        self.log = scrolledtext.ScrolledText(self, height=12, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=False, padx=12, pady=(4, 12))
        self._write_log("Results are preliminary screening estimates. No motor or structural vibration level is invented when inputs are missing.\n")
        for variable in self.vars.values():
            variable.trace_add("write", self._schedule_review_refresh)
        self._refresh_review()

    def _current_values(self) -> dict[str, str]:
        return {name: variable.get().strip() for name, variable in self.vars.items()}

    def _schedule_review_refresh(self, *_: object) -> None:
        if self.review_refresh_id is not None:
            self.after_cancel(self.review_refresh_id)
        self.review_refresh_id = self.after(150, self._refresh_review)

    def _refresh_review(self) -> None:
        if self.review_refresh_id is not None:
            self.after_cancel(self.review_refresh_id)
        self.review_refresh_id = None
        try:
            manifest = build_manifest(self._current_values(), options=self.model_options)
            preview = render_manifest(manifest)
        except (OSError, ValueError, RuntimeError) as exc:
            preview = f"Cannot resolve the current inputs yet: {exc}"
        self.review_text.configure(state="normal")
        self.review_text.delete("1.0", "end")
        self.review_text.insert("1.0", preview)
        self.review_text.configure(state="disabled")

    def _export_inputs(self, suffix: str) -> None:
        filetype = "JSON files" if suffix == ".json" else "Markdown files"
        filename = filedialog.asksaveasfilename(
            title="Export all simulation inputs",
            defaultextension=suffix,
            initialfile="vibration_inputs" + suffix,
            filetypes=[(filetype, "*" + suffix)],
        )
        if not filename:
            return
        path = Path(filename)
        if path.exists() and not messagebox.askyesno(
            "Replace existing file?", f"Replace {path}?"
        ):
            return
        try:
            saved = save_manifest(build_manifest(self._current_values(), options=self.model_options),
                                  path, overwrite=path.exists())
        except (OSError, ValueError) as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self._write_log(f"Exported simulation inputs: {saved}\n")

    def _save_rsf(self) -> None:
        filename = filedialog.asksaveasfilename(
            title="Save simulation settings", defaultextension=".RSF",
            initialfile=self.rsf_path.name if self.rsf_path else "vibration_simulation.RSF",
            initialdir=str(self.rsf_path.parent) if self.rsf_path else str(HERE),
            filetypes=[("Rocket Simulation Files", "*.RSF *.rsf")],
            confirmoverwrite=False,
        )
        if not filename:
            return
        path = Path(filename)
        if path.exists() and not messagebox.askyesno("Replace RSF?", f"Replace {path}?"):
            return
        try:
            saved = save_rsf(self._current_values(), path, options=self.model_options,
                             overwrite=path.exists())
        except (OSError, ValueError) as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.rsf_path = saved
        self.title(f"Preliminary Rocket Vibration Workflow - {saved.name}")
        self._write_log(f"Saved simulation settings: {saved}\n")

    def _load_rsf(self) -> None:
        if self.busy:
            messagebox.showinfo("Simulation running", "Wait for the current run before loading another setup.")
            return
        filename = filedialog.askopenfilename(
            title="Load simulation settings",
            filetypes=[("Rocket Simulation Files", "*.RSF *.rsf"), ("All files", "*.*")],
        )
        if not filename:
            return
        try:
            values, options, warnings = load_rsf(Path(filename))
        except (OSError, ValueError) as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        if not messagebox.askyesno("Load simulation?", "Replace the current form with this RSF setup?"):
            return
        # Apply only after the entire file has validated and replacement is approved.
        self.model_options = options
        for name, variable in self.vars.items():
            variable.set(values[name])
        self.rsf_path = Path(filename).resolve()
        self.title(f"Preliminary Rocket Vibration Workflow - {self.rsf_path.name}")
        self._refresh_review()
        self._write_log(f"Loaded simulation settings: {self.rsf_path}\n")
        if warnings:
            messagebox.showwarning("Missing input files", "\n".join(warnings) +
                                   "\n\nUpdate the CSV inputs folder or filenames before running.")

    def _browse_inputs(self) -> None:
        path = filedialog.askdirectory(initialdir=self.vars["csv_inputs_folder"].get() or str(HERE))
        if path:
            self.vars["csv_inputs_folder"].set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.vars["output_dir"].set(path)

    def _write_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start(self, action: str) -> None:
        if self.busy:
            return
        values = self._current_values()
        self.busy = True
        self.weather_button.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self._write_log(f"\nStarting {action}...\n")
        threading.Thread(target=self._worker, args=(action, values, dict(self.model_options)), daemon=True).start()

    def _worker(self, action: str, values: dict[str, str], options: dict) -> None:
        try:
            output = run_workflow(
                action, values,
                lambda message: self.events.put(("log", message + "\n")),
                options=options,
            )
            self.events.put(("done", str(output)))
        except (Exception, SystemExit) as exc:
            self.events.put(("error", str(exc)))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, message = self.events.get_nowait()
                if kind == "log":
                    self._write_log(message)
                elif kind == "done":
                    self.busy = False
                    self.latest_output = Path(message)
                    self.weather_button.configure(state="normal")
                    self.run_button.configure(state="normal")
                    self._write_log(f"Complete. Results: {message}\n")
                elif kind == "error":
                    self.busy = False
                    self.weather_button.configure(state="normal")
                    self.run_button.configure(state="normal")
                    self._write_log(f"ERROR: {message}\n")
                    messagebox.showerror("Workflow error", message)
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def _open_results(self) -> None:
        if self.latest_output is None:
            messagebox.showinfo("No results", "Run a weather fetch or complete screening first.")
        elif os.name == "nt":
            os.startfile(self.latest_output)  # type: ignore[attr-defined]
        else:
            messagebox.showinfo("Results folder", str(self.latest_output))


if __name__ == "__main__":
    VibrationApp().mainloop()
