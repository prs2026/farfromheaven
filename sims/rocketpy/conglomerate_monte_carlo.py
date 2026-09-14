"""Combine Project Blaze Monte Carlo pickle files into one streamed output.

Usage examples::

    python conglomerate_monte_carlo.py -o combined.pkl run_1.pkl run_2.pkl
    python conglomerate_monte_carlo.py -o combined.pkl "results/run_*.pkl"
    python conglomerate_monte_carlo.py -o combined.pkl results_directory

Only load pickle files you trust. Python pickle files can execute code while
being read.
"""

from __future__ import annotations

import argparse
import glob
import pickle
import shutil
import sys
import time
from collections.abc import Iterator, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MONTE_CARLO_FORMAT = "projectblaze.rocketpy.monte_carlo"


def _expand_inputs(values: Sequence[str | Path]) -> list[Path]:
    """Resolve files, directories, and glob expressions without duplicates."""

    paths: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        text = str(value)
        matches = [Path(match) for match in glob.glob(text)]
        if not matches:
            matches = [Path(text)]
        for match in matches:
            candidates = sorted(match.glob("*.pkl")) if match.is_dir() else [match]
            for candidate in candidates:
                resolved = candidate.expanduser().resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(resolved)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"input pickle does not exist: {missing[0]}")
    if not paths:
        raise ValueError("no input pickle files were found")
    return paths


def _read_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as pickle_file:
        payload = pickle.load(pickle_file)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not begin with a dictionary payload")
    if payload.get("format") != MONTE_CARLO_FORMAT:
        raise ValueError(f"{path} is not a Project Blaze Monte Carlo pickle")
    return payload


def _record_count(payload: dict[str, Any], path: Path) -> int:
    if payload.get("storage") == "pickle_stream":
        count = int(payload.get("flight_record_count", -1))
        if count < 0:
            raise ValueError(f"{path} has an invalid flight_record_count")
        return count
    flights = payload.get("flights")
    if not isinstance(flights, list):
        raise ValueError(f"{path} contains neither a pickle stream nor a flights list")
    return len(flights)


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from streamed files one at a time."""

    with path.open("rb") as pickle_file:
        payload = pickle.load(pickle_file)
        count = _record_count(payload, path)
        if payload.get("storage") == "pickle_stream":
            for record_index in range(count):
                try:
                    record = pickle.load(pickle_file)
                except EOFError as exc:
                    raise ValueError(
                        f"{path} ended before flight record {record_index + 1}/{count}"
                    ) from exc
                if not isinstance(record, dict):
                    raise TypeError(f"{path} flight record {record_index} is not a dictionary")
                yield record
        else:
            yield from payload["flights"]


def _simulation_count(payload: dict[str, Any], record_count: int) -> int:
    configuration = payload.get("configuration", {})
    if isinstance(configuration, dict):
        value = configuration.get("number_of_simulations")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return record_count


def _combined_header(
    paths: Sequence[Path], payloads: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[int]]:
    first = payloads[0]
    state_columns = tuple(first.get("state_columns", ()))
    environments = [payload.get("environment") for payload in payloads]
    record_counts = [
        _record_count(payload, path) for path, payload in zip(paths, payloads)
    ]
    simulation_counts = [
        _simulation_count(payload, count)
        for payload, count in zip(payloads, record_counts)
    ]
    for path, payload in zip(paths[1:], payloads[1:]):
        if tuple(payload.get("state_columns", ())) != state_columns:
            raise ValueError(f"{path} uses different flight-state columns")
        if payload.get("environment") != environments[0]:
            raise ValueError(f"{path} uses different environment metadata")

    configuration = deepcopy(first.get("configuration", {}))
    if not isinstance(configuration, dict):
        configuration = {}
    list_fields = (
        "sampled_parameters",
        "sampled_launch_angles",
        "sampled_headings",
        "weather_samples",
    )
    for field in list_fields:
        combined: list[Any] = []
        for payload in payloads:
            source_configuration = payload.get("configuration", {})
            values = (
                source_configuration.get(field, [])
                if isinstance(source_configuration, dict)
                else []
            )
            if isinstance(values, list):
                combined.extend(deepcopy(values))
        configuration[field] = combined
    configuration["number_of_simulations"] = sum(simulation_counts)
    configuration["random_seed"] = None
    configuration["source_random_seeds"] = [
        payload.get("configuration", {}).get("random_seed")
        if isinstance(payload.get("configuration"), dict)
        else None
        for payload in payloads
    ]

    merged_sources: list[dict[str, Any]] = []
    simulation_offset = 0
    for path, payload, record_count, simulation_count in zip(
        paths, payloads, record_counts, simulation_counts
    ):
        nested_sources = payload.get("merged_sources")
        if isinstance(nested_sources, list) and sum(
            int(source.get("flight_record_count", 0))
            for source in nested_sources
            if isinstance(source, dict)
        ) == record_count:
            for source in nested_sources:
                nested_source = deepcopy(source)
                nested_source["simulation_offset"] = simulation_offset + int(
                    source.get("simulation_offset", 0)
                )
                merged_sources.append(nested_source)
        else:
            merged_sources.append(
                {
                    "path": str(path),
                    "flight_record_count": record_count,
                    "number_of_simulations": simulation_count,
                    "simulation_offset": simulation_offset,
                }
            )
        simulation_offset += simulation_count

    header = {
        "format": MONTE_CARLO_FORMAT,
        "format_version": 2,
        "storage": "pickle_stream",
        "flight_record_count": sum(record_counts),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "state_columns": state_columns,
        "configuration": configuration,
        "environment": deepcopy(environments[0]),
        "merged_source_count": len(paths),
        "merged_from": [str(path) for path in paths],
        "merged_sources": merged_sources,
    }
    return header, simulation_counts


def conglomerate_monte_carlo_outputs(
    input_paths: Sequence[str | Path], output_path: str | Path
) -> dict[str, Any]:
    """Stream compatible Monte Carlo files into one combined pickle."""

    paths = _expand_inputs(input_paths)
    output = Path(output_path).expanduser().resolve()
    if output in paths:
        raise ValueError("the output file cannot also be an input file")
    payloads = [_read_header(path) for path in paths]
    header, simulation_counts = _combined_header(paths, payloads)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("wb") as output_file:
        pickle.dump(header, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        for path, payload in zip(paths, payloads):
            if payload.get("storage") == "pickle_stream":
                # The stream is already a sequence of complete pickle objects.
                # Skip its metadata object and copy all flight records as raw
                # bytes, avoiding costly deserialization and serialization of
                # every full trajectory. The loader applies index offsets.
                with path.open("rb") as input_file:
                    pickle.load(input_file)
                    shutil.copyfileobj(input_file, output_file, length=16 * 1024 * 1024)
            else:
                for record in _iter_records(path):
                    pickle.dump(record, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    return header


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", help="input .pkl files, directories, or glob expressions"
    )
    parser.add_argument("-o", "--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start_time = time.perf_counter()
    try:
        header = conglomerate_monte_carlo_outputs(args.inputs, args.output)
    except (OSError, TypeError, ValueError, pickle.PickleError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Combined {header['flight_record_count']} flight records from "
        f"{header['merged_source_count']} files into "
        f"{Path(args.output).expanduser().resolve()} in "
        f"{time.perf_counter() - start_time:.2f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
