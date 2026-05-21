#!/usr/bin/env python3
"""
schema_tool.py
--------------
Standalone CLI: point at any VRP JSON file to inspect its schema and generate
schema.yaml — which the EngineBot service uses at startup.

Usage
-----
    python schema_tool.py <json_file>
    python schema_tool.py <json_file> --output custom_schema.yaml
    python schema_tool.py <json_file> --show-only       # print only, no file write
    python schema_tool.py <json_file> --no-color        # plain text output

Examples
--------
    python schema_tool.py sample_vrp.json
    python schema_tool.py /data/fleet_data.json --output /data/schema.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Try to import rich for pretty output (graceful fallback) ──────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    from rich.padding import Padding
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ── Core logic (no rich dependency) ──────────────────────────────────────────
def _load_and_build(json_path: Path) -> tuple[dict, dict]:
    """Load VRP JSON, build DataFrames + schema. Returns (dataframes_dict, schema_dict)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from data_layer import load_vrp_json
    from schema_layer import build_schema

    dfs = load_vrp_json(json_path)
    schema = build_schema(dfs)
    return dfs, schema


def _save_schema_yaml(schema: dict, output: Path) -> None:
    from schema_layer import save_schema
    save_schema(schema, output)


# ── Rich display ──────────────────────────────────────────────────────────────
def _display_rich(json_path: Path, dfs: dict, schema: dict, console: "Console") -> None:
    """Full rich terminal display."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]VRP Schema Tool[/] · [dim]EngineBot[/]",
        border_style="bright_blue",
        padding=(0, 2),
    ))
    console.print()

    # Summary row
    summary_parts = []
    for df_name, df_info in schema["dataframes"].items():
        summary_parts.append(
            f"[green]•[/] [bold]{df_name}[/]  "
            f"[dim]{df_info['row_count']} rows × {len(df_info['columns'])} cols[/]"
        )
    console.print(f"[bold]File:[/] [yellow]{json_path}[/]")
    for part in summary_parts:
        console.print(f"  {part}")
    console.print()

    # Per-dataframe tables
    for df_name, df_info in schema["dataframes"].items():
        title = (
            f"[bold cyan]{df_name}[/]  "
            f"[dim]({df_info['entity']} · {df_info['row_count']} rows)[/]"
        )
        t = Table(
            title=title,
            box=box.ROUNDED,
            header_style="bold magenta",
            border_style="bright_blue",
            show_lines=False,
            expand=False,
        )
        t.add_column("Column", style="cyan", no_wrap=True)
        t.add_column("Dtype", style="yellow", no_wrap=True)
        t.add_column("Sample values", style="white")

        for col, cinfo in df_info["columns"].items():
            samples = "  ·  ".join(str(s) for s in cinfo["sample_values"])
            t.add_row(col, cinfo["dtype"], samples)

        console.print(t)
        console.print()

    # Aliases
    console.print("[bold]Aliases[/]")
    for df_name, df_info in schema["dataframes"].items():
        aliases = ", ".join(f"[italic]{a}[/]" for a in df_info.get("aliases", []))
        console.print(f"  [cyan]{df_name}[/]  →  {aliases}")
    console.print()

    # Join keys
    console.print("[bold]Detected join keys[/]")
    for jk in schema.get("join_keys", []):
        on = f"[green]{jk['on']}[/]" if jk.get("on") else "[red]none detected[/]"
        console.print(
            f"  [cyan]{jk['left']}[/] ⟷ [cyan]{jk['right']}[/]  on {on}"
        )
    console.print()


# ── Plain text display ────────────────────────────────────────────────────────
def _display_plain(json_path: Path, dfs: dict, schema: dict) -> None:
    """Fallback plain-text display when rich is not installed."""
    sep = "─" * 70
    print(f"\n  VRP Schema Tool  ·  EngineBot")
    print(sep)
    print(f"File: {json_path}")
    for df_name, df_info in schema["dataframes"].items():
        print(f"  • {df_name}  {df_info['row_count']} rows × {len(df_info['columns'])} cols")
    print()

    for df_name, df_info in schema["dataframes"].items():
        print(sep)
        print(f"  {df_name}  ({df_info['entity']} · {df_info['row_count']} rows)")
        print(sep)
        col_w = max(len(c) for c in df_info["columns"]) + 2
        dtype_w = 12
        print(f"  {'Column':<{col_w}}  {'Dtype':<{dtype_w}}  Sample values")
        print(f"  {'─'*col_w}  {'─'*dtype_w}  {'─'*30}")
        for col, cinfo in df_info["columns"].items():
            samples = "  ·  ".join(str(s) for s in cinfo["sample_values"])
            print(f"  {col:<{col_w}}  {cinfo['dtype']:<{dtype_w}}  {samples}")
        print()

    print("Aliases:")
    for df_name, df_info in schema["dataframes"].items():
        aliases = ", ".join(df_info.get("aliases", []))
        print(f"  {df_name}  →  {aliases}")
    print()

    print("Join keys:")
    for jk in schema.get("join_keys", []):
        on = jk.get("on") or "none detected"
        print(f"  {jk['left']}  ⟷  {jk['right']}  on {on}")
    print()


# ── CLI entry point ───────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        prog="schema_tool",
        description="Generate and display the schema for a VRP JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("json_file", type=Path, help="Path to the VRP JSON file")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("schema.yaml"),
        help="Output path for schema.yaml (default: schema.yaml)",
    )
    parser.add_argument(
        "--show-only", "-s",
        action="store_true",
        help="Print schema to terminal only — do not write the YAML file",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable rich colored output",
    )
    args = parser.parse_args()

    json_path: Path = args.json_file.resolve()

    # ── Validate input ────────────────────────────────────────────────────────
    if not json_path.exists():
        print(f"ERROR: File not found: {json_path}", file=sys.stderr)
        return 1

    # ── Load & build ──────────────────────────────────────────────────────────
    try:
        dfs, schema = _load_and_build(json_path)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON — {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # ── Display ───────────────────────────────────────────────────────────────
    use_rich = HAS_RICH and not args.no_color
    if use_rich:
        console = Console()
        _display_rich(json_path, dfs, schema, console)
    else:
        _display_plain(json_path, dfs, schema)

    # ── Save YAML ─────────────────────────────────────────────────────────────
    if not args.show_only:
        try:
            _save_schema_yaml(schema, args.output)
            msg = f"Schema saved → {args.output.resolve()}"
            if use_rich:
                console.print(f"[bold green]✓[/]  {msg}\n")
            else:
                print(f"✓  {msg}\n")
        except Exception as exc:
            print(f"ERROR saving schema: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
