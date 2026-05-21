"""
execution_layer.py
------------------
Validates and executes a structured JSON query against the loaded DataFrames.
Execution order: filter → join → select → aggregate → sort → limit.
"""

from __future__ import annotations

import operator as op_module
from typing import Any

import pandas as pd

from data_layer import get_dataframes

# ── Allowed operators & aggregate functions ───────────────────────────────────
ALLOWED_OPERATORS: set[str] = {">", "<", "==", "!=", ">=", "<=", "contains", "isnull", "notnull"}
ALLOWED_AGGREGATES: set[str] = {"sum", "mean", "count", "min", "max"}
ALLOWED_SORT_ORDERS: set[str] = {"asc", "desc"}

_OPERATOR_FN: dict[str, Any] = {
    ">":  op_module.gt,
    "<":  op_module.lt,
    "==": op_module.eq,
    "!=": op_module.ne,
    ">=": op_module.ge,
    "<=": op_module.le,
}


# ── Validation ────────────────────────────────────────────────────────────────
def _validate_query(query: dict, dataframes: dict[str, pd.DataFrame]) -> list[str]:
    """Return a list of error strings; empty list means the query is valid."""
    errors: list[str] = []

    def check_df(name: str, ctx: str) -> None:
        if name not in dataframes:
            errors.append(
                f"[{ctx}] Unknown dataframe '{name}'. "
                f"Available: {list(dataframes)}"
            )

    def check_col(df_name: str, col: str, ctx: str) -> None:
        df = dataframes.get(df_name)
        if df is not None and col not in df.columns:
            errors.append(
                f"[{ctx}] Column '{col}' not found in '{df_name}'. "
                f"Available: {list(df.columns)}"
            )

    def resolve_prefixed_col(prefixed: str, ctx: str) -> tuple[str, str] | None:
        """Split 'df_name.col_name'; validate both exist."""
        if "." not in prefixed:
            errors.append(f"[{ctx}] Column '{prefixed}' must be prefixed with dataframe name (e.g. df_drivers.id)")
            return None
        df_name, col = prefixed.split(".", 1)
        check_df(df_name, ctx)
        check_col(df_name, col, ctx)
        return df_name, col

    # dataframe / join
    if "join" in query:
        j = query["join"]
        check_df(j.get("left", ""), "join.left")
        check_df(j.get("right", ""), "join.right")
        left_df = dataframes.get(j.get("left", ""), pd.DataFrame())
        right_df = dataframes.get(j.get("right", ""), pd.DataFrame())
        on_col = j.get("on", "")
        if on_col:
            if on_col not in left_df.columns:
                errors.append(f"[join.on] Column '{on_col}' not in '{j.get('left')}'")
            if on_col not in right_df.columns:
                errors.append(f"[join.on] Column '{on_col}' not in '{j.get('right')}'")
    elif "dataframe" in query:
        check_df(query["dataframe"], "dataframe")

    # filters
    _join_dfs: set[str] = set()
    if "join" in query:
        j = query["join"]
        _join_dfs = {j.get("left", ""), j.get("right", "")}

    for i, f in enumerate(query.get("filter", [])):
        ctx = f"filter[{i}]"
        fdf = f.get("dataframe", "")
        check_df(fdf, ctx)
        check_col(fdf, f.get("column", ""), ctx)
        if f.get("operator") not in ALLOWED_OPERATORS:
            errors.append(
                f"[{ctx}] Invalid operator '{f.get('operator')}'. "
                f"Allowed: {sorted(ALLOWED_OPERATORS)}"
            )
        # In non-join queries the filter dataframe must match the target dataframe
        if "join" not in query and "dataframe" in query:
            target_df = query["dataframe"]
            if fdf and fdf != target_df:
                errors.append(
                    f"[{ctx}] filter references dataframe '{fdf}' but query targets '{target_df}'. "
                    "Use a join query to filter across multiple dataframes."
                )

    # select
    for s in query.get("select", []):
        resolve_prefixed_col(s, "select")

    # aggregate
    if "aggregate" in query:
        agg = query["aggregate"]
        resolve_prefixed_col(agg.get("column", ""), "aggregate.column")
        if agg.get("function") not in ALLOWED_AGGREGATES:
            errors.append(
                f"[aggregate.function] Invalid function '{agg.get('function')}'. "
                f"Allowed: {sorted(ALLOWED_AGGREGATES)}"
            )

    # sort
    if "sort" in query:
        srt = query["sort"]
        resolve_prefixed_col(srt.get("column", ""), "sort.column")
        if srt.get("order") not in ALLOWED_SORT_ORDERS:
            errors.append(f"[sort.order] Must be 'asc' or 'desc', got '{srt.get('order')}'")

    # limit
    if "limit" in query:
        try:
            int(query["limit"])
        except (TypeError, ValueError):
            errors.append(f"[limit] Must be an integer, got '{query['limit']}'")

    return errors


# ── Execution ─────────────────────────────────────────────────────────────────
def _apply_filter(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    col: str = f["column"]
    oper: str = f["operator"]
    val = f.get("value")

    if oper == "contains":
        return df[df[col].astype(str).str.contains(str(val), na=False)]
    if oper == "isnull":
        return df[df[col].isnull()]
    if oper == "notnull":
        return df[df[col].notnull()]

    fn = _OPERATOR_FN[oper]
    return df[fn(df[col], val)]


def execute_query(query: dict, dataframes: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """
    Validate and execute *query* against *dataframes*.

    Returns a DataFrame result.  Raises ValueError with a structured
    message if validation fails.
    """
    if dataframes is None:
        dataframes = get_dataframes()

    errors = _validate_query(query, dataframes)
    if errors:
        raise ValueError("Query validation failed:\n" + "\n".join(f"  • {e}" for e in errors))

    # ── Step 1: Determine working DataFrames ──────────────────────────────────
    if "join" in query:
        j = query["join"]
        left_df = dataframes[j["left"]].copy()
        right_df = dataframes[j["right"]].copy()

        # Prefix columns to avoid ambiguity
        left_df.columns = [f"{j['left']}.{c}" for c in left_df.columns]
        right_df.columns = [f"{j['right']}.{c}" for c in right_df.columns]
        join_on_left = f"{j['left']}.{j['on']}"
        join_on_right = f"{j['right']}.{j['on']}"

        # ── Step 2: Filters on individual DFs before join ─────────────────
        for f in query.get("filter", []):
            fdf_name = f["dataframe"]
            prefixed_col = f"{fdf_name}.{f['column']}"
            if fdf_name == j["left"]:
                left_df = left_df[_apply_filter(
                    left_df.rename(columns={prefixed_col: f["column"]}), f
                ).index]
            elif fdf_name == j["right"]:
                right_df = right_df[_apply_filter(
                    right_df.rename(columns={prefixed_col: f["column"]}), f
                ).index]

        result = pd.merge(left_df, right_df, left_on=join_on_left, right_on=join_on_right)

    else:
        df_name = query.get("dataframe", "")
        result = dataframes[df_name].copy()
        result.columns = [f"{df_name}.{c}" for c in result.columns]

        # ── Step 2: Filters ───────────────────────────────────────────────
        for f in query.get("filter", []):
            prefixed_col = f"{f['dataframe']}.{f['column']}"
            tmp = result.rename(columns={prefixed_col: f["column"]})
            mask = _apply_filter(tmp, f).index
            result = result.loc[mask]

    # ── Step 3: Select ────────────────────────────────────────────────────────
    # Track columns before select so aggregate can still reference them if needed
    pre_select_result = result
    if "select" in query:
        cols_to_keep = [c for c in query["select"] if c in result.columns]
        missing = [c for c in query["select"] if c not in result.columns]
        if missing:
            raise ValueError(
                f"select references columns not available in the result: {missing}. "
                f"Available: {list(result.columns)}"
            )
        if cols_to_keep:
            result = result[cols_to_keep]

    # ── Step 4: Aggregate ─────────────────────────────────────────────────────
    if "aggregate" in query:
        agg = query["aggregate"]
        col = agg["column"]
        func = agg["function"]
        # If select narrowed away the aggregate column, fall back to pre-select frame
        agg_source = result if col in result.columns else pre_select_result
        if col in agg_source.columns:
            val = getattr(agg_source[col], func)()
            result = pd.DataFrame([{col: val, "aggregate_function": func}])
        else:
            raise ValueError(
                f"aggregate column '{col}' not found in result. "
                f"Available columns: {list(agg_source.columns)}"
            )

    # ── Step 5: Sort ──────────────────────────────────────────────────────────
    if "sort" in query:
        srt = query["sort"]
        col = srt["column"]
        ascending = srt.get("order", "asc") == "asc"
        if col in result.columns:
            result = result.sort_values(col, ascending=ascending)

    # ── Step 6: Limit ─────────────────────────────────────────────────────────
    if "limit" in query:
        result = result.head(int(query["limit"]))

    return result.reset_index(drop=True)
