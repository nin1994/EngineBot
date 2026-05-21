"""
data_layer.py
-------------
Accepts a VRP JSON (dict) and dynamically flattens the relevant arrays into
three pandas DataFrames using pd.json_normalize().  No column names are
hardcoded — all keys are preserved exactly as they appear in the JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import pandas as pd

# ── Placeholder sample data (used when no file has been uploaded yet) ─────────
SAMPLE_VRP_JSON: dict = {
    "drivers": [
        {
            "id": "D001",
            "name": "Alice Johnson",
            "earliest_start_time": "08:00",
            "latest_end_time": "18:00",
            "max_hours": 8,
            "home_location": {"lat": 37.7749, "lon": -122.4194},
            "license_class": "A",
            "available": True,
        },
        {
            "id": "D002",
            "name": "Bob Smith",
            "earliest_start_time": "07:00",
            "latest_end_time": "17:00",
            "max_hours": 9,
            "home_location": {"lat": 37.8044, "lon": -122.2712},
            "license_class": "B",
            "available": True,
        },
        {
            "id": "D003",
            "name": "Carol White",
            "earliest_start_time": "09:00",
            "latest_end_time": "19:00",
            "max_hours": 7,
            "home_location": {"lat": 37.6879, "lon": -122.4702},
            "license_class": "A",
            "available": False,
        },
    ],
    "vehicles": [
        {
            "id": "V001",
            "type": "truck",
            "capacity_kg": 5000,
            "capacity_vol_m3": 20,
            "fuel_type": "diesel",
            "max_speed_kmh": 90,
            "requires_license": "A",
            "cost_per_km": 1.5,
        },
        {
            "id": "V002",
            "type": "van",
            "capacity_kg": 1500,
            "capacity_vol_m3": 8,
            "fuel_type": "electric",
            "max_speed_kmh": 120,
            "requires_license": "B",
            "cost_per_km": 0.8,
        },
        {
            "id": "V003",
            "type": "truck",
            "capacity_kg": 8000,
            "capacity_vol_m3": 35,
            "fuel_type": "diesel",
            "max_speed_kmh": 80,
            "requires_license": "A",
            "cost_per_km": 2.1,
        },
    ],
    "restrictions": [
        {"driver_id": "D001", "vehicle_id": "V003", "reason": "weight limit"},
        {"driver_id": "D002", "vehicle_id": "V001", "reason": "license mismatch"},
        {"driver_id": "D003", "vehicle_id": "V002", "reason": "training required"},
    ],
    "depot": {"id": "DEP1", "lat": 37.7749, "lon": -122.4194},
}


# ── Internal state ─────────────────────────────────────────────────────────────
_dataframes: dict[str, pd.DataFrame] = {}


def _flatten_array(data: dict, key: str) -> pd.DataFrame:
    """Flatten a top-level JSON array via pd.json_normalize()."""
    records = data.get(key, [])
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records)


def load_vrp_json(source: Union[str, Path, dict, None] = None) -> dict[str, pd.DataFrame]:
    """
    Parse a VRP JSON source and populate the three target DataFrames.

    Parameters
    ----------
    source : str | Path | dict | None
        - Path/str  → read JSON from file
        - dict      → use directly
        - None      → fall back to built-in SAMPLE_VRP_JSON

    Returns
    -------
    dict mapping df names → DataFrames
    """
    global _dataframes

    if source is None:
        data = SAMPLE_VRP_JSON
    elif isinstance(source, dict):
        data = source
    else:
        with open(source, "r", encoding="utf-8") as fh:
            data = json.load(fh)

    df_drivers = _flatten_array(data, "drivers")
    df_vehicles = _flatten_array(data, "vehicles")
    df_vehicle_driver_restrictions = _flatten_array(data, "restrictions")

    _dataframes = {
        "df_drivers": df_drivers,
        "df_vehicles": df_vehicles,
        "df_vehicle_driver_restrictions": df_vehicle_driver_restrictions,
    }

    return _dataframes


def get_dataframes() -> dict[str, pd.DataFrame]:
    """Return the currently loaded DataFrames (loading sample data if needed)."""
    if not _dataframes:
        load_vrp_json(None)
    return _dataframes


def get_dataframe(name: str) -> pd.DataFrame:
    """Return a single DataFrame by name."""
    frames = get_dataframes()
    if name not in frames:
        raise KeyError(f"Unknown dataframe '{name}'. Available: {list(frames)}")
    return frames[name]


# ── Convenience: load sample data on import so the app works out-of-the-box ───
load_vrp_json(None)
