"""
schema_layer.py
---------------
Builds a rich schema dictionary from the loaded DataFrames and persists it
as a YAML file.  All inference is fully dynamic — no column names are hardcoded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
import pandas as pd

from data_layer import get_dataframes

# ── Known metadata that is static (aliases & join keys) ───────────────────────
_DF_META = {
    "df_drivers": {
        "entity": "drivers",
        "aliases": ["driver", "drivers"],
    },
    "df_vehicles": {
        "entity": "vehicles",
        "aliases": ["vehicle", "vehicles", "truck"],
    },
    "df_vehicle_driver_restrictions": {
        "entity": "driver-vehicle restrictions",
        "aliases": ["restriction", "restrictions"],
    },
}

# Join keys are described relationally — no column names assumed.
_JOIN_KEYS: list[dict] = [
    {
        "left": "df_drivers",
        "right": "df_vehicle_driver_restrictions",
        "description": "shared driver id column (auto-detected at runtime)",
    },
    {
        "left": "df_vehicles",
        "right": "df_vehicle_driver_restrictions",
        "description": "shared vehicle id column (auto-detected at runtime)",
    },
]

SCHEMA_PATH = Path("schema.yaml")


def _sample_values(series: pd.Series, n: int = 3) -> list[Any]:
    """Return up to *n* non-null sample values from a Series."""
    non_null = series.dropna()
    samples = non_null.head(n).tolist()
    # Make sure values are JSON-safe primitives for YAML serialisation
    return [_yaml_safe(v) for v in samples]


def _yaml_safe(val: Any) -> Any:
    """Convert numpy/pandas scalar to a Python primitive."""
    if hasattr(val, "item"):          # numpy scalar
        return val.item()
    if isinstance(val, bool):
        return bool(val)
    if isinstance(val, (int, float, str)):
        return val
    return str(val)


def _detect_join_key(df_left: pd.DataFrame, df_right: pd.DataFrame) -> str | None:
    """
    Heuristic: find the first column name that exists in both DataFrames.
    Falls back to None if there is no overlap.
    """
    shared = set(df_left.columns) & set(df_right.columns)
    if shared:
        return sorted(shared)[0]
    return None


def build_schema(dataframes: dict[str, pd.DataFrame] | None = None) -> dict:
    """
    Dynamically build a schema dict from the current DataFrames.

    The schema contains:
    - per-dataframe: entity name, aliases, column→(dtype, samples) mapping
    - cross-dataframe join key hints (auto-detected from column overlap)
    """
    if dataframes is None:
        dataframes = get_dataframes()

    schema: dict = {"dataframes": {}, "join_keys": []}

    for df_name, df in dataframes.items():
        meta = _DF_META.get(df_name, {"entity": df_name, "aliases": [df_name]})

        columns: dict[str, Any] = {}
        for col in df.columns:
            columns[col] = {
                "dtype": str(df[col].dtype),
                "sample_values": _sample_values(df[col]),
                "description": "",          # left blank for manual enrichment
            }

        schema["dataframes"][df_name] = {
            "entity": meta["entity"],
            "aliases": meta["aliases"],
            "row_count": len(df),
            "columns": columns,
        }

    # Build join key entries with actual detected column names
    for jk in _JOIN_KEYS:
        left_df = dataframes.get(jk["left"], pd.DataFrame())
        right_df = dataframes.get(jk["right"], pd.DataFrame())
        detected_col = _detect_join_key(left_df, right_df)
        schema["join_keys"].append(
            {
                "left": jk["left"],
                "right": jk["right"],
                "on": detected_col,
                "description": jk["description"],
            }
        )

    return schema


def save_schema(schema: dict | None = None, path: Path = SCHEMA_PATH) -> Path:
    """Persist the schema to *path* as YAML and return the path."""
    if schema is None:
        schema = build_schema()
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(schema, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return path


def load_schema_from_yaml(path: Path = SCHEMA_PATH) -> dict:
    """Read schema back from the YAML file."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_schema_for_prompt(schema: dict | None = None) -> str:
    """
    Serialise the schema into a compact, LLM-friendly text block.
    Each dataframe section lists aliases, join keys, and columns with
    dtype + sample values so the model can reason about the data.
    """
    if schema is None:
        schema = build_schema()

    lines: list[str] = []
    join_map: dict[str, list] = {}

    # Pre-index join keys by dataframe
    for jk in schema.get("join_keys", []):
        for side in (jk["left"], jk["right"]):
            join_map.setdefault(side, []).append(jk)

    for df_name, df_info in schema["dataframes"].items():
        lines.append(f"=== {df_name} ===")
        lines.append(f"Entity  : {df_info['entity']}")
        lines.append(f"Aliases : {', '.join(df_info['aliases'])}")
        lines.append(f"Rows    : {df_info['row_count']}")

        if df_name in join_map:
            for jk in join_map[df_name]:
                other = jk["right"] if jk["left"] == df_name else jk["left"]
                lines.append(f"Join    : {df_name} JOIN {other} ON {jk['on']}")

        lines.append("Columns :")
        for col, cinfo in df_info["columns"].items():
            samples = ", ".join(str(s) for s in cinfo["sample_values"])
            lines.append(f"  - {col} [{cinfo['dtype']}] samples=[{samples}]")
        lines.append("")

    return "\n".join(lines)
