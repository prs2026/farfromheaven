"""Render a RASAero II CDX1 design and its key dimensions to one PNG.

The script can be called normally or through ``cdx1_summary_png.bat`` for
Windows Explorer drag-and-drop usage. Dimensions are shown in inches and the
remaining values use RASAero's native imperial units.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import textwrap
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# Standard-atmosphere sea-level sound speed at 59 °F (15 °C), in ft/s.
# https://ntrs.nasa.gov/api/citations/19880002266/downloads/19880002266.pdf
SEA_LEVEL_SOUND_SPEED_FPS = 1116.45
PART_TAGS = {"NoseCone", "BodyTube", "FinCan", "Booster", "Transition", "BoatTail"}
PART_COLORS = {
    "nosecone": "#64748b",
    "bodytube": "#38bdf8",
    "fincan": "#22c55e",
    "transition": "#f59e0b",
    "boattail": "#fb7185",
    "booster": "#8b5cf6",
}
# Dark headers support white text; pale stripes keep values easy to read.
SECTION_COLORS = {
    "components": ("#1e40af", "#eff6ff"),
    "fins": ("#166534", "#f0fdf4"),
    "motors": ("#9a3412", "#fff7ed"),
    "results": ("#6b21a8", "#faf5ff"),
    "launch": ("#155e75", "#ecfeff"),
    "recovery": ("#9f1239", "#fff1f2"),
    "additional": ("#475569", "#f1f5f9"),
}


def _text(element: ET.Element | None, name: str, default: str = "") -> str:
    if element is None:
        return default
    child = element.find(name)
    return (child.text or default).strip() if child is not None else default


def _number(value: str | None, default: float = 0.0) -> float:
    try:
        result = float(value) if value is not None and value.strip() else default
    except (AttributeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _float(element: ET.Element | None, name: str, default: float = 0.0) -> float:
    return _number(_text(element, name), default)


def _optional_float(element: ET.Element | None, name: str) -> float | None:
    """Return a saved numeric field, preserving the difference from a missing tag."""

    value = _text(element, name)
    return _number(value) if value else None


def _boolean(element: ET.Element | None, name: str) -> bool:
    return _text(element, name).lower() in {"true", "1", "yes"}


def _parse_fin(element: ET.Element) -> dict[str, Any]:
    return {
        "count": int(_float(element, "Count")),
        "root_chord": _float(element, "Chord"),
        "span": _float(element, "Span"),
        "sweep": _float(element, "SweepDistance"),
        "tip_chord": _float(element, "TipChord"),
        "thickness": _float(element, "Thickness"),
        "le_radius": _optional_float(element, "LERadius"),
        "location": _float(element, "Location"),
        "airfoil": _text(element, "AirfoilSection", "Not specified"),
    }


def _parse_part(element: ET.Element, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "tag": element.tag,
        "part_type": _text(element, "PartType", element.tag),
        "location": _float(element, "Location"),
        "length": _float(element, "Length"),
        "diameter": _float(element, "Diameter"),
        "inside_diameter": _float(element, "InsideDiameter"),
        "rear_diameter": _float(element, "RearDiameter"),
        "shoulder_length": _float(element, "ShoulderLength"),
        "boattail_length": _float(element, "BoattailLength"),
        "boattail_rear_diameter": _float(element, "BoattailRearDiameter"),
        "shape": _text(element, "Shape"),
        "tip_radius": _optional_float(element, "BluntRadius"),
        "color": _text(element, "Color"),
        "fins": [_parse_fin(fin) for fin in element.findall("Fin")],
    }


def parse_cdx1(path: str | Path) -> dict[str, Any]:
    """Parse the fields displayed by the summary sheet."""

    source = Path(path).expanduser().resolve()
    if source.suffix.casefold() != ".cdx1":
        raise ValueError(f"expected a .CDX1 file, received {source.name!r}")
    root = ET.parse(source).getroot()
    design = root.find("RocketDesign")
    if design is None:
        raise ValueError("CDX1 file does not contain RocketDesign")
    parts = [
        _parse_part(element, index)
        for index, element in enumerate(
            (child for child in design if child.tag in PART_TAGS), start=1
        )
    ]
    if not parts:
        raise ValueError("CDX1 RocketDesign does not contain recognizable components")
    _resolve_geometry(parts)

    simulations = []
    simulation_list = root.find("SimulationList")
    for index, simulation in enumerate(
        simulation_list.findall("Simulation") if simulation_list is not None else (),
        start=1,
    ):
        simulations.append(
            {
                "index": index,
                "sustainer_engine": _text(simulation, "SustainerEngine"),
                "sustainer_mass": _float(simulation, "SustainerLaunchWt"),
                "sustainer_cg": _float(simulation, "SustainerCG"),
                "sustainer_ignition_delay": _float(
                    simulation, "SustainerIgnitionDelay"
                ),
                "sustainer_nozzle": _float(simulation, "SustainerNozzleDiameter"),
                "saved_results": {
                    "flight_time": _optional_float(simulation, "FlightTime"),
                    "time_to_apogee": _optional_float(simulation, "TimetoApogee"),
                    "max_altitude": _optional_float(simulation, "MaxAltitude"),
                    "max_velocity": _optional_float(simulation, "MaxVelocity"),
                    "optimum_weight": _optional_float(simulation, "OptimumWt"),
                    "optimum_max_altitude": _optional_float(
                        simulation, "OptimumMaxAlt"
                    ),
                },
                "boosters": [
                    {
                        "number": booster_number,
                        "has_fields": any(
                            child.tag.startswith(f"Booster{booster_number}")
                            or child.tag == f"IncludeBooster{booster_number}"
                            for child in simulation
                        ),
                        "engine": _text(
                            simulation, f"Booster{booster_number}Engine"
                        ),
                        "included": _boolean(
                            simulation, f"IncludeBooster{booster_number}"
                        ),
                        "mass": _float(
                            simulation, f"Booster{booster_number}LaunchWt"
                        ),
                        "cg": _float(simulation, f"Booster{booster_number}CG"),
                        "ignition_delay": _float(
                            simulation, f"Booster{booster_number}IgnitionDelay"
                        ),
                        "separation_delay": _float(
                            simulation,
                            f"Booster{booster_number}SeparationDelay",
                            _float(simulation, f"Booster{booster_number}Delay"),
                        ),
                        "nozzle": _float(
                            simulation, f"Booster{booster_number}NozzleDiameter"
                        ),
                    }
                    for booster_number in (1, 2)
                ],
            }
        )

    recovery_element = root.find("Recovery")
    recovery = []
    for event_number in (1, 2):
        enabled = _boolean(recovery_element, f"Event{event_number}")
        if recovery_element is not None and any(
            recovery_element.find(f"{field}{event_number}") is not None
            for field in ("Event", "Altitude", "DeviceType", "EventType", "Size", "CD")
        ):
            recovery.append(
                {
                    "number": event_number,
                    "enabled": enabled,
                    "device": _text(
                        recovery_element, f"DeviceType{event_number}", "None"
                    ),
                    "event_type": _text(
                        recovery_element, f"EventType{event_number}", "None"
                    ),
                    "altitude": _float(recovery_element, f"Altitude{event_number}"),
                    "size": _float(recovery_element, f"Size{event_number}"),
                    "cd": _float(recovery_element, f"CD{event_number}"),
                }
            )

    return {
        "source": source,
        "file_version": _text(root, "FileVersion", "Unknown"),
        "parts": parts,
        "simulations": simulations,
        "launch_site": root.find("LaunchSite"),
        "recovery": recovery,
        "surface": _text(design, "Surface", "Not specified"),
        "comments": _text(design, "Comments"),
        "additional_inputs": _additional_input_cards(root),
    }


def _additional_input_cards(root: ET.Element) -> list[tuple[str, list[list[str]]]]:
    """Collect only the requested aerodynamic and launch-hardware inputs."""
    allowed = {
        "RocketDesign": {"ModifiedBarrowman", "Turbulence"},
        **{tag: {"LaunchShoeArea", "RailGuideDiameter"} for tag in PART_TAGS},
    }
    cards = []

    def visit(node: ET.Element, title: str, in_protuberance: bool = False) -> None:
        in_protuberance = in_protuberance or node.tag == "Protuberance"
        rows = []
        for child in node:
            if len(child):
                continue
            if not in_protuberance and child.tag not in allowed.get(node.tag, set()):
                continue
            label = child.tag
            if label == "RailGuideDiameter":
                label += " (in)"
            elif label in {"LaunchShoeArea", "InclinedPlate1FrontalArea", "InclinedPlate2FrontalArea"}:
                label += " (in²)"
            elif label in {"InclinedPlate1Angle", "InclinedPlate2Angle"}:
                label += " (deg)"
            rows.append([label, (child.text or "").strip() or "(empty)"])
        # Limit each card to a readable size; long/unknown field values wrap.
        for start in range(0, len(rows), 10):
            suffix = f" ({start // 10 + 1})" if len(rows) > 10 else ""
            cards.append((title + suffix, rows[start:start + 10]))
        counts: dict[str, int] = {}
        part_index = 0
        for child in node:
            if not len(child):
                continue
            counts[child.tag] = counts.get(child.tag, 0) + 1
            if child.tag in PART_TAGS:
                part_index += 1
                child_title = f"{part_index} {child.tag}"
            elif child.tag == "Simulation":
                child_title = f"Simulation {counts[child.tag]}"
            elif node is root:
                child_title = child.tag
            else:
                child_title = f"{title} / {child.tag} {counts[child.tag]}"
            visit(child, child_title, in_protuberance)

    visit(root, "Document settings")
    return cards


def _extra_card_rows(cards: Sequence[tuple[str, list[list[str]]]]) -> list[list[Any]]:
    """Prepare three-card rows with wrapped text and physical row heights."""
    batches = []
    for start in range(0, len(cards), 3):
        batch = []
        for title, fields in cards[start:start + 3]:
            wrapped = [
                [textwrap.fill(label, 28), textwrap.fill(value, 28)]
                for label, value in fields
            ]
            lines = sum(max(label.count("\n"), value.count("\n")) + 1
                        for label, value in wrapped)
            batch.append((textwrap.fill(title, 42), wrapped, 0.65 + 0.23 * lines))
        batches.append(batch)
    return batches


def _draw_extra_cards(axis: Any, batches: list[list[Any]]) -> None:
    axis.axis("off")
    axis.set_title("Additional saved CDX1 inputs (original field names)",
                   loc="left", fontsize=11, fontweight="bold", pad=7,
                   color=SECTION_COLORS["additional"][0])
    heights = [max(card[2] for card in batch) for batch in batches]
    total = sum(heights)
    top = 1.0
    for batch, height in zip(batches, heights):
        row_height = height / total
        width, gap = 0.32, 0.02
        left = (1 - (len(batch) * width + (len(batch) - 1) * gap)) / 2
        for index, (title, fields, _) in enumerate(batch):
            # Reserve a fixed physical gap for headings even on short rows.
            card_axis = axis.inset_axes([left + index * (width + gap),
                                        top - row_height + 0.07 / total,
                                        width, (height - 0.45) / total])
            _style_table(card_axis, title, ("Field", "Saved value"), fields,
                         font_size=7.5, section="additional")
            card_axis.title.set_fontsize(9)
            table = next(iter(card_axis.tables))
            # Allocate space by wrapped line count instead of clipping text.
            row_lines = [1] + [max(a.count("\n"), b.count("\n")) + 1 for a, b in fields]
            for (row, column), cell in table.get_celld().items():
                cell.set_height(0.92 * row_lines[row] / sum(row_lines))
                cell.get_text().set_ha("left")
        top -= row_height


def _inches(value_in: float) -> str:
    return f"{value_in:g}"


def _pounds(value_lb: float) -> str:
    return f"{value_lb:g}"


def _style_table(
    axis: Any,
    title: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    font_size: float = 7.5,
    empty_message: str = "None defined",
    section: str = "additional",
) -> None:
    header_color, row_color = SECTION_COLORS[section]
    axis.axis("off")
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=7,
                   color=header_color)
    if not rows:
        axis.text(0, 0.75, empty_message, fontsize=9, color="#64748b")
        return
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        bbox=[0, 0, 1, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row % 2 == 0:
            cell.set_facecolor(row_color)


def _style_transposed_tables(
    axis: Any,
    title: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    title_columns: int = 1,
    max_columns: int = 3,
    font_size: float = 7.2,
    empty_message: str = "None defined",
    section: str = "additional",
) -> None:
    """Draw each record as its own two-column field/value table."""

    header_color, row_color = SECTION_COLORS[section]
    axis.axis("off")
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=7,
                   color=header_color)
    if not rows:
        axis.text(0, 0.75, empty_message, fontsize=9, color="#64748b")
        return

    table_columns = min(max_columns, len(rows))
    table_rows = math.ceil(len(rows) / table_columns)
    horizontal_gap = 0.012
    vertical_gap = 0.045
    width = (1 - horizontal_gap * (table_columns - 1)) / table_columns
    height = (0.92 - vertical_gap * (table_rows - 1)) / table_rows

    for index, row_values in enumerate(rows):
        grid_row, grid_column = divmod(index, table_columns)
        cards_in_row = min(table_columns, len(rows) - grid_row * table_columns)
        row_width = cards_in_row * width + (cards_in_row - 1) * horizontal_gap
        x = (1 - row_width) / 2 + grid_column * (width + horizontal_gap)
        y = 0.92 - (grid_row + 1) * height - grid_row * vertical_gap
        card_title = " · ".join(str(value) for value in row_values[:title_columns])
        field_rows = [
            [str(field).replace("\n", " "), str(value)]
            for field, value in zip(
                columns[title_columns:], row_values[title_columns:]
            )
        ]
        table = axis.table(
            cellText=field_rows,
            colLabels=("Field", card_title),
            cellLoc="left",
            colLoc="left",
            colWidths=(0.52, 0.48),
            bbox=[x, y, width, height],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        for (table_row, table_column), cell in table.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            cell.set_linewidth(0.6)
            if table_row == 0:
                cell.set_facecolor(header_color)
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            elif table_row % 2 == 0:
                cell.set_facecolor(row_color)
            if table_column == 0 and table_row > 0:
                cell.get_text().set_fontweight("bold")


def _resolve_geometry(parts: list[dict[str, Any]]) -> None:
    """Resolve CDX1 attachment references into nose-relative drawing stations.

    A fin can ends at Location and sleeves the preceding tube. Its shoulder
    precedes its cylindrical Length. A booster starts at Location, followed by
    its shoulder, cylindrical Length, and optional appended boattail. Fin
    Location is measured forward from the cylinder's aft end, not the boattail.
    """

    previous_diameter = 0.0
    for part in parts:
        tag = part["tag"].casefold()
        shoulder = part["shoulder_length"] if tag in {"fincan", "booster"} else 0.0
        start = part["location"]
        if tag == "fincan":
            start -= part["length"] + shoulder
        body_start = start + shoulder
        body_end = body_start + part["length"]
        boattail = (
            part["boattail_length"]
            if part["boattail_rear_diameter"] > 0 else 0.0
        )
        forward_diameter = part["diameter"]
        if tag in {"fincan", "booster"}:
            forward_diameter = part["inside_diameter"] or previous_diameter or forward_diameter
        elif tag in {"transition", "boattail"}:
            forward_diameter = previous_diameter or forward_diameter
        part.update(
            profile_start=start,
            body_start=body_start,
            body_end=body_end,
            profile_end=body_end + boattail,
            forward_diameter=forward_diameter,
        )
        previous_diameter = (
            part["boattail_rear_diameter"] if boattail else
            (part["rear_diameter"] or part["diameter"])
        )


def _fin_root_le(part: dict[str, Any], fin: dict[str, Any]) -> float:
    return part["body_end"] - fin["location"]


def _part_polygon(part: dict[str, Any]) -> list[tuple[float, float]]:
    x0 = part["profile_start"]
    x1 = part["body_end"]
    radius = part["diameter"] / 2
    rear_radius = (
        part["rear_diameter"] / 2 if part["rear_diameter"] > 0 else radius
    )
    tag = part["tag"].casefold()
    if tag == "nosecone":
        return [(x0, 0), (x1, radius), (x1, -radius)]
    if tag in {"transition", "boattail"}:
        radius = part["forward_diameter"] / 2
        return [
            (x0, radius),
            (x1, rear_radius),
            (x1, -rear_radius),
            (x0, -radius),
        ]
    top = [(x0, part["forward_diameter"] / 2), (part["body_start"], radius), (x1, radius)]
    if part["profile_end"] > x1:
        top.append((part["profile_end"], part["boattail_rear_diameter"] / 2))
    return top + [(x, -y) for x, y in reversed(top)]


def _draw_rocket(axis: Any, data: dict[str, Any]) -> None:
    parts = data["parts"]
    maximum_span = max(
        (fin["span"] for part in parts for fin in part["fins"]), default=0.0
    )
    maximum_radius = max(part["diameter"] for part in parts) / 2
    total_length = max(part["profile_end"] for part in parts)
    vertical_extent = maximum_radius + maximum_span

    # Fin-can sleeves must cover the underlying tube, while the booster starts
    # at their aft attachment plane rather than underneath the sleeve.
    draw_order = sorted(parts, key=lambda part: part["tag"].casefold() == "fincan")
    for part in draw_order:
        key = part["tag"].casefold()
        color = PART_COLORS.get(key, "#94a3b8")
        polygon = Polygon(
            _part_polygon(part),
            closed=True,
            facecolor=color,
            edgecolor="#0f172a",
            linewidth=1.0,
            alpha=1.0,
        )
        axis.add_patch(polygon)
        x_center = (part["profile_start"] + part["profile_end"]) / 2
        axis.text(
            x_center,
            maximum_radius * 1.22,
            f"{part['index']} {part['part_type']}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=25 if part["length"] < total_length * 0.08 else 0,
        )
        for fin_index, fin in enumerate(part["fins"], start=1):
            root_le = _fin_root_le(part, fin)
            root_te = root_le + fin["root_chord"]
            tip_le = root_le + fin["sweep"]
            tip_te = tip_le + fin["tip_chord"]
            radius = part["diameter"] / 2
            top = [(root_le, radius), (tip_le, radius + fin["span"]),
                   (tip_te, radius + fin["span"]), (root_te, radius)]
            bottom = [(x, -y) for x, y in top]
            for points in (top, bottom):
                axis.add_patch(
                    Polygon(
                        points,
                        closed=True,
                        facecolor="#ef4444",
                        edgecolor="#7f1d1d",
                        linewidth=1.0,
                        alpha=0.8,
                    )
                )
            axis.text(
                (root_le + tip_te) / 2,
                radius + fin["span"] + vertical_extent * 0.05,
                f"F{part['index']}.{fin_index}",
                ha="center",
                fontsize=7,
                color="#991b1b",
            )

    first_simulation = data["simulations"][0] if data["simulations"] else None
    if first_simulation:
        cg_items = [
            ("Sustainer CG", first_simulation["sustainer_cg"], "#059669")
        ]
        cg_items.extend(
            (
                f"Booster {booster['number']} stack CG",
                booster["cg"],
                "#7c3aed",
            )
            for booster in first_simulation["boosters"]
            if booster["included"] and booster["cg"] > 0
        )
        for label, cg, color in cg_items:
            axis.axvline(cg, color=color, linestyle="--", linewidth=1.4, label=label)

    dimension_y = -vertical_extent * 1.25
    axis.annotate(
        "",
        xy=(0, dimension_y),
        xytext=(total_length, dimension_y),
        arrowprops={"arrowstyle": "<->", "color": "#334155", "linewidth": 1},
    )
    axis.text(
        total_length / 2,
        dimension_y - vertical_extent * 0.08,
        f"Overall component envelope: {total_length:.3f} in",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
    )
    axis.axhline(0, color="#475569", linewidth=0.6)
    axis.set_xlim(-total_length * 0.025, total_length * 1.025)
    axis.set_ylim(-vertical_extent * 1.55, vertical_extent * 1.55)
    axis.set_xlabel("Distance from nose tip (inches)")
    axis.set_yticks([])
    axis.grid(axis="x", color="#e2e8f0", linewidth=0.6)
    axis.set_title("Dimensioned side profile", loc="left", fontsize=12, fontweight="bold")
    if first_simulation:
        axis.legend(loc="lower right", fontsize=7, framealpha=0.9)


def render_summary(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    dpi: int = 180,
) -> Path:
    """Render one CDX1 file and return the generated PNG path."""

    data = parse_cdx1(input_path)
    source: Path = data["source"]
    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else source.with_name(
            f"{source.stem}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
    )
    if output.suffix.casefold() != ".png":
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    part_rows = []
    fin_rows = []
    for part in data["parts"]:
        rear = part["rear_diameter"] or part["boattail_rear_diameter"]
        detail_items = [part["shape"]] if part["shape"] else []
        if rear:
            detail_items.append(f"Rear OD {rear:g} in")
        details = "; ".join(detail_items) or "—"
        part_rows.append(
            [
                str(part["index"]),
                part["part_type"],
                _inches(part["profile_start"]),
                _inches(part["length"]),
                _inches(part["shoulder_length"]),
                _inches(part["boattail_length"]),
                _inches(part["profile_end"] - part["profile_start"]),
                _inches(part["diameter"]),
                _inches(part["inside_diameter"]),
                details,
                f"{part['tip_radius']:g}" if part["tip_radius"] is not None else "—",
            ]
        )
        for fin_index, fin in enumerate(part["fins"], start=1):
            root_le = _fin_root_le(part, fin)
            fin_rows.append(
                [
                    f"F{part['index']}.{fin_index}",
                    part["part_type"],
                    str(fin["count"]),
                    _inches(fin["root_chord"]),
                    _inches(fin["tip_chord"]),
                    _inches(fin["span"]),
                    _inches(fin["sweep"]),
                    _inches(fin["thickness"]),
                    _inches(root_le),
                    fin["airfoil"],
                    f"{fin['le_radius']:g}" if fin["le_radius"] is not None else "—",
                ]
            )

    simulation_rows = []
    saved_result_rows = []
    for simulation in data["simulations"]:
        simulation_rows.append(
            [
                str(simulation["index"]),
                "Sustainer",
                "Yes",
                simulation["sustainer_engine"] or "—",
                _pounds(simulation["sustainer_mass"]),
                _inches(simulation["sustainer_cg"]),
                f"{simulation['sustainer_ignition_delay']:g}",
                "—",
                _inches(simulation["sustainer_nozzle"]),
            ]
        )
        for booster in simulation["boosters"]:
            if not booster["has_fields"]:
                continue
            simulation_rows.append(
                [
                    str(simulation["index"]),
                    f"Booster {booster['number']} stack",
                    "Yes" if booster["included"] else "No",
                    booster["engine"] or "—",
                    _pounds(booster["mass"]),
                    _inches(booster["cg"]),
                    f"{booster['ignition_delay']:g}",
                    f"{booster['separation_delay']:g}",
                    _inches(booster["nozzle"]),
                ]
            )

        results = simulation["saved_results"]
        if any(value not in (None, 0.0) for value in results.values()):
            def saved_value(name: str, unit: str) -> str:
                value = results[name]
                return f"{value:g} {unit}" if value not in (None, 0.0) else "—"

            saved_result_rows.append(
                [
                    str(simulation["index"]),
                    saved_value("flight_time", "s"),
                    saved_value("time_to_apogee", "s"),
                    saved_value("max_altitude", "ft"),
                    saved_value("max_velocity", "ft/s"),
                    (
                        f"{results['max_velocity'] / SEA_LEVEL_SOUND_SPEED_FPS:.3f} (derived)"
                        if results["max_velocity"] is not None and results["max_velocity"] > 0
                        else "—"
                    ),
                    saved_value("optimum_weight", "lb"),
                    saved_value("optimum_max_altitude", "ft"),
                ]
            )

    launch = data["launch_site"]
    altitude_ft = _float(launch, "Altitude")
    pressure_inhg = _float(launch, "Pressure")
    temperature_f = _float(launch, "Temperature")
    wind_mph = _float(launch, "WindSpeed")
    rod_length_ft = _float(launch, "RodLength")
    launch_rows = [
        ["Altitude", f"{altitude_ft:g} ft"],
        ["Pressure", f"{pressure_inhg:g} inHg"],
        ["Temperature", f"{temperature_f:g} °F"],
        ["Wind speed", f"{wind_mph:g} mph"],
        ["Rod length", f"{rod_length_ft:g} ft"],
        ["Rod angle", f"{_float(launch, 'RodAngle'):g}°"],
    ]
    recovery_rows = [
        [
            str(event["number"]),
            "Yes" if event["enabled"] else "No",
            event["device"],
            event["event_type"],
            f"{event['altitude']:g} ft",
            _inches(event["size"]),
            f"{event['cd']:g}",
        ]
        for event in data["recovery"]
    ]

    def card_rows(count: int, columns: int = 3) -> int:
        return max(1, math.ceil(count / columns))

    extra_batches = _extra_card_rows(data["additional_inputs"])
    extra_height = sum(max(card[2] for card in batch) for batch in extra_batches)
    section_heights = [
        3.2,
        3.0 * card_rows(len(part_rows)),
        2.75 * card_rows(len(fin_rows)),
        2.7 * card_rows(len(simulation_rows)),
        2.5 * card_rows(len(saved_result_rows)),
        max(1.8, 2.0 * card_rows(len(recovery_rows), 2)),
    ]
    if extra_batches:
        section_heights.append(extra_height)
    figure_height = max(16.0, 3.0 + sum(section_heights))
    figure = plt.figure(figsize=(18, figure_height), facecolor="white")
    grid = figure.add_gridspec(
        len(section_heights),
        2,
        height_ratios=section_heights,
        left=0.05,
        right=0.95,
        top=1 - 1.6 / figure_height,
        bottom=0.8 / figure_height,
        hspace=0.20,
        wspace=0.16,
    )
    diagram_axis = figure.add_subplot(grid[0, :])
    parts_axis = figure.add_subplot(grid[1, :])
    fins_axis = figure.add_subplot(grid[2, :])
    simulations_axis = figure.add_subplot(grid[3, :])
    results_axis = figure.add_subplot(grid[4, :])
    launch_axis = figure.add_subplot(grid[5, 0])
    recovery_axis = figure.add_subplot(grid[5, 1])

    total_length = max(part["profile_end"] for part in data["parts"])
    max_diameter = max(part["diameter"] for part in data["parts"])
    figure.suptitle(
        f"{source.stem} — RASAero design summary",
        x=0.5,
        y=1 - 0.35 / figure_height,
        ha="center",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    figure.text(
        0.5,
        1 - 0.85 / figure_height,
        f"CDX1 version {data['file_version']}  •  Surface: {data['surface']}  •  "
        f"Envelope: {total_length:g} in long × {max_diameter:g} in maximum body diameter\n"
        "Dimensions: inches; masses: pounds; other values use native imperial units",
        ha="center",
        va="center",
        fontsize=9,
        color="#475569",
    )

    _draw_rocket(diagram_axis, data)
    _style_transposed_tables(
        parts_axis,
        "Components",
        ("#", "Component", "Front from nose\nin", "Body/part length\nin",
         "Shoulder/transition\nin", "Boattail length\nin", "Total part length\nin",
         "OD\nin", "ID\nin", "Shape or aft detail", "Nose tip radius\nin"),
        part_rows,
        title_columns=2,
        section="components",
    )
    _style_transposed_tables(
        fins_axis,
        "Fin sets",
        ("ID", "Component", "Count", "Root chord\nin", "Tip chord\nin",
         "Span\nin", "Sweep\nin", "Thickness\nin",
         "Root LE from nose\nin", "Airfoil", "LE radius\nin"),
        fin_rows,
        title_columns=2,
        section="fins",
        font_size=7.0,
    )
    _style_transposed_tables(
        simulations_axis,
        "Simulation masses, centers of gravity and motors",
        ("Sim", "Configuration", "Included", "Motor", "Launch mass\nlb",
         "CG from nose\nin", "Ignition delay\ns", "Separation delay\ns",
         "Nozzle diameter\nin"),
        simulation_rows,
        title_columns=2,
        section="motors",
        font_size=7.0,
    )
    _style_transposed_tables(
        results_axis,
        "Saved RASAero simulation results",
        ("Sim", "Flight time", "Time to apogee", "Maximum altitude",
         "Maximum velocity", "Sea-level Mach (59 °F)",
         "Optimum weight", "Optimum max altitude"),
        saved_result_rows,
        title_columns=1,
        section="results",
        font_size=7.4,
        empty_message="No nonzero saved simulation results in this CDX1 file",
    )
    _style_table(launch_axis, "Launch site", ("Field", "Value"), launch_rows,
                 font_size=8.0, section="launch")
    _style_transposed_tables(
        recovery_axis,
        "Recovery events",
        ("Event", "Enabled", "Device", "Trigger", "Altitude", "Size\nin", "Cd"),
        recovery_rows,
        title_columns=1,
        section="recovery",
        max_columns=2,
        font_size=7.3,
    )
    if extra_batches:
        _draw_extra_cards(figure.add_subplot(grid[6, :]), extra_batches)

    figure.text(
        0.5,
        0.30 / figure_height,
        textwrap.fill(
            f"Source: {source}"
            + (f"  •  Comments: {data['comments']}" if data["comments"] else ""),
            width=180,
        ),
        fontsize=7.5,
        color="#64748b",
        ha="center",
        va="center",
    )
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="CDX1 file(s) to render")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output PNG path; valid only when one input is supplied",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution")
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_output",
        help="open each PNG after it is generated (Windows)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output is not None and len(args.inputs) != 1:
        print("error: --output can only be used with one input file", file=sys.stderr)
        return 2
    if args.dpi <= 0:
        print("error: --dpi must be positive", file=sys.stderr)
        return 2

    outputs = []
    for input_path in args.inputs:
        try:
            output = render_summary(input_path, args.output, dpi=args.dpi)
        except (OSError, ValueError, ET.ParseError) as exc:
            print(f"error: {input_path}: {exc}", file=sys.stderr)
            return 1
        outputs.append(output)
        print(f"Wrote {output}")

    if args.open_output and hasattr(os, "startfile"):
        for output in outputs:
            os.startfile(output)  # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
