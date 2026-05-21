"""
llm_layer.py
------------
Runs a local Qwen2.5-7B GGUF model via llama-cpp-python.
Builds a schema-aware prompt using the Qwen2.5 chat template
(im_start / im_end tokens) and extracts a strict JSON query from the output.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Model path — override with env var MODEL_PATH if needed ──────────────────
DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(Path.home() / "models" / "qwen2.5-7b-instruct-q4_k_m.gguf"),
)

_llm = None   # lazy-loaded singleton


def _load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Lazy-load the llama-cpp-python Llama instance."""
    global _llm
    if _llm is not None:
        return _llm

    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. "
            "Run: pip install llama-cpp-python"
        ) from exc

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Set the MODEL_PATH environment variable to the correct path."
        )

    logger.info("Loading model from %s …", model_path)
    _llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        n_gpu_layers=-1,   # offload all layers to GPU if available
        verbose=False,
    )
    logger.info("Model loaded.")
    return _llm


# ── Prompt construction ───────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a precise query-generation assistant for a Vehicle Routing Problem (VRP) database.
You are given a schema describing three pandas DataFrames.
Your task is to convert the user's natural-language question into a structured JSON query object.

Rules:
1. Return ONLY a single valid JSON object — no explanation, no markdown, no surrounding text.
2. The JSON must conform to this schema:
{
  "dataframe": "<df_name> (for single-table queries)",
  "join": { "left": "<df_name>", "right": "<df_name>", "on": "<column>" },
  "filter": [
    { "dataframe": "<df_name>", "column": "<col>", "operator": "<op>", "value": <val> }
  ],
  "select": ["<df_name>.<col>", ...],
  "aggregate": { "column": "<df_name>.<col>", "function": "<func>" },
  "sort": { "column": "<df_name>.<col>", "order": "asc|desc" },
  "limit": <integer>
}
3. Allowed operators : >, <, ==, !=, >=, <=, contains, isnull, notnull
4. Allowed aggregate functions: sum, mean, count, min, max
5. Omit any key you do not need — do not include null or empty values.
6. Column references in "select", "aggregate.column", and "sort.column" MUST be prefixed
   with the dataframe name AND use the EXACT column name from the schema above.
   NEVER invent column names — only use names that appear in the DATABASE SCHEMA section.
7. Dataframe names must exactly match one of: df_drivers, df_vehicles, df_vehicle_driver_restrictions

CRITICAL RULES — READ CAREFULLY:
- Comparison operators (>, <, >=, <=, ==, !=, contains, isnull, notnull) ALWAYS go inside the
  "filter" array as a filter object. They MUST NEVER appear inside "aggregate".
- "aggregate" is ONLY for computing a single summary value over a column using one of:
  sum, mean, count, min, max
- If the user says "greater than", "more than", "above", "over", "at least", "less than",
  "below", "under", "at most", "equal to", "not equal to" — these are ALWAYS filters.
- Column names ARE case-sensitive. Copy them exactly from the schema.

Examples of correct structure (column names here are placeholders — use real names from schema):
  Q: "Show vehicles with capacity greater than 3000"
  A: {"dataframe": "df_vehicles", "filter": [{"dataframe": "df_vehicles", "column": "<capacity_column>", "operator": ">", "value": 3000}]}

  Q: "How many drivers are there?"
  A: {"dataframe": "df_drivers", "aggregate": {"column": "df_drivers.<id_column>", "function": "count"}}

  Q: "Show drivers sorted by hours descending"
  A: {"dataframe": "df_drivers", "sort": {"column": "df_drivers.<hours_column>", "order": "desc"}}
"""

_QUERY_TEMPLATE = """\
### DATABASE SCHEMA
{schema}

### USER QUESTION
{question}

### JSON QUERY (no explanation, no markdown, output JSON only):
"""


def build_prompt(question: str, schema_text: str) -> str:
    """
    Construct a Qwen2.5-style chat prompt using im_start / im_end tokens.
    """
    user_content = _QUERY_TEMPLATE.format(schema=schema_text, question=question)

    prompt = (
        "<|im_start|>system\n"
        f"{_SYSTEM_PROMPT.strip()}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_content.strip()}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt


def _extract_json(raw: str) -> dict:
    """
    Find the first complete JSON object in *raw* by locating the
    outermost { … } and parsing it.  Raises ValueError on failure.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output:\n{raw!r}")
    candidate = raw[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from model output: {exc}\n{candidate!r}") from exc


def generate_query(question: str, schema_text: str, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    Send *question* + *schema_text* to the local LLM and return the parsed
    JSON query dict.
    """
    llm = _load_model(model_path)
    prompt = build_prompt(question, schema_text)

    logger.info("Sending prompt to LLM …")
    response = llm(
        prompt,
        max_tokens=512,
        temperature=0.1,
        stop=["<|im_end|>", "<|im_start|>"],
        echo=False,
    )

    raw_text: str = response["choices"][0]["text"]
    logger.debug("Raw LLM output: %r", raw_text)

    return _extract_json(raw_text)


# ── Comparison phrase → operator mapping ─────────────────────────────────────
# Each entry is (list_of_phrases, operator).  Phrases are checked as substrings
# of the lowercased question *except* those marked with a leading '^' which are
# checked as whole words to avoid false-positive matches (e.g. 'over' in 'discover').
_COMPARISON_PHRASES: list[tuple[list[str], str]] = [
    (["greater than or equal", "at least", ">="], ">="),
    (["less than or equal", "at most", "<="], "<="),
    (["greater than", "more than", "^above", "^over", "higher than", ">"], ">"),
    (["less than", "fewer than", "^below", "^under", "lower than", "<"], "<"),
    (["not equal", "!=", "different from"], "!="),
    (["equal to", "^equals", "=="], "=="),
]


def _try_parse_number(tokens: list[str]) -> float | int | None:
    """Try to extract a numeric value from a list of word tokens."""
    for tok in tokens:
        tok = tok.strip(".,;:")
        try:
            v = float(tok)
            return int(v) if v == int(v) else v
        except ValueError:
            continue
    return None


def _phrase_matches(phrase: str, q: str, words: list[str]) -> bool:
    """
    Check whether *phrase* occurs in *q*.
    Phrases prefixed with '^' are matched as whole words only (using the
    split word list) to avoid substring false-positives like 'over' in 'discover'.
    """
    if phrase.startswith("^"):
        return phrase[1:] in words
    return phrase in q


def _has_comparison(q: str, words: list[str]) -> str | None:
    """Return the detected comparison operator string, or None."""
    for phrases, op in _COMPARISON_PHRASES:
        if any(_phrase_matches(p, q, words) for p in phrases):
            return op
    return None


# ── Semantic patterns: question keywords → column-name substrings ──────────────
# Each entry: (question_keywords, col_name_substrings, preferred_df)
# col_name_substrings are matched case-insensitively against actual column names
# so they work regardless of whether the JSON uses snake_case, PascalCase, etc.
_SEMANTIC_PATTERNS: list[tuple[list[str], list[str], str]] = [
    # question keywords            # substrings in actual col name     # preferred df
    (["capacity", "weight", "load"],       ["capacity", "weight", "load"],   "df_vehicles"),
    (["volume", "vol", "cubic"],           ["volume", "vol", "cubic"],       "df_vehicles"),
    (["cost", "price", "rate", "pay"],     ["cost", "rate", "pay"],          "df_vehicles"),
    (["speed", "velocity"],               ["speed", "kmh", "mph"],          "df_vehicles"),
    (["hours", "duration", "shift"],      ["hour", "duration", "shift"],    "df_drivers"),
    (["start time", "earliest", "begin"], ["start", "earliest", "begin"],   "df_drivers"),
    (["end time", "latest", "finish"],    ["end", "latest", "finish"],      "df_drivers"),
    (["overtime", "over time"],           ["overtime", "over"],             "df_drivers"),
    (["available", "active"],             ["available", "active"],          "df_drivers"),
    (["pay rate", "payrate", "salary"],   ["payrate", "pay"],               "df_drivers"),
    (["license", "licence", "class"],     ["license", "licence", "class"],   "df_vehicles"),
    (["fuel"],                            ["fuel"],                          "df_vehicles"),
    (["driver"],                          ["driver", "driverid"],            "df_vehicle_driver_restrictions"),
    (["vehicle", "truck", "van"],         ["vehicle", "vehicleid"],          "df_vehicle_driver_restrictions"),
]


def _normalize_col(name: str) -> str:
    """Lowercase and strip separators for fuzzy column matching."""
    return name.lower().replace("_", "").replace(" ", "").replace("-", "")


def _find_col_by_substrings(
    df_columns: list[str], substrings: list[str]
) -> str | None:
    """
    Return the first column whose normalised name contains any of *substrings*.
    Case-insensitive, ignores underscores/spaces/dashes so it matches
    both snake_case and PascalCase variants of the same concept.
    """
    for col in df_columns:
        col_norm = _normalize_col(col)
        if any(s in col_norm for s in substrings):
            return col
    return None


def _build_dynamic_hints(
    dataframes: dict,
) -> list[tuple[list[str], str, str]]:
    """
    Scan the live DataFrames and build a concrete hint list:
      (question_keywords, df_name, actual_column_name)
    using _SEMANTIC_PATTERNS to fuzzy-match column names.
    Falls back to _FALLBACK_COL_HINTS (using sample column names) when
    no dataframes are provided or a pattern finds no match.
    """
    hints: list[tuple[list[str], str, str]] = []
    seen: set[tuple[str, str]] = set()  # (df_name, col) dedupe

    for q_keywords, col_subs, preferred_df in _SEMANTIC_PATTERNS:
        resolved = False
        # Try preferred df first, then all others
        candidates = [preferred_df] + [d for d in dataframes if d != preferred_df]
        for df_name in candidates:
            df = dataframes.get(df_name)
            if df is None or df.empty:
                continue
            col = _find_col_by_substrings(list(df.columns), col_subs)
            if col and (df_name, col) not in seen:
                hints.append((q_keywords, df_name, col))
                seen.add((df_name, col))
                resolved = True
                break
        # If nothing resolved (e.g. no dataframes loaded yet), skip silently
    return hints


def _find_id_col(df_columns: list[str]) -> str | None:
    """Return the most likely ID column from a dataframe's columns."""
    for col in df_columns:
        if _normalize_col(col) in ("id", "$id", "uid", "uuid", "key"):
            return col
    # Fallback: first column that ends with 'id'
    for col in df_columns:
        if _normalize_col(col).endswith("id"):
            return col
    return df_columns[0] if df_columns else None


def _find_bool_col(df_columns: list[str]) -> str | None:
    """Return the first column that looks like a boolean availability flag."""
    bool_subs = ["available", "active", "enabled", "isavailable", "isactive"]
    for col in df_columns:
        if _normalize_col(col) in bool_subs:
            return col
    return None


def generate_query_stub(
    question: str,
    schema_text: str,
    dataframes: dict | None = None,
) -> dict:
    """
    Rule-based fallback that produces a valid JSON query without a real LLM.

    When *dataframes* is provided the stub resolves all column names from the
    actual loaded data (PascalCase, snake_case, any naming convention).
    Without *dataframes* it still works but column-level hints are omitted.
    """
    from data_layer import get_dataframes  # local import to avoid circularity

    if dataframes is None:
        dataframes = get_dataframes()

    # Build column hints from the live dataframes — no hardcoded names
    col_hints = _build_dynamic_hints(dataframes)

    q = question.lower()
    words = q.split()

    # ── 1. Determine target dataframe ─────────────────────────────────────────
    if any(k in q for k in ("restrict", "restriction")):
        base: dict = {"dataframe": "df_vehicle_driver_restrictions"}
    elif any(k in q for k in ("vehicle", "truck", "van")):
        base = {"dataframe": "df_vehicles"}
    else:
        base = {"dataframe": "df_drivers"}

    df_name: str = base["dataframe"]

    # ── 2. Detect comparison filters (NEVER goes in aggregate) ────────────────
    detected_operator: str | None = _has_comparison(q, words)

    if detected_operator is not None:
        detected_col: str | None = None
        detected_df: str = df_name
        for q_keywords, col_df, col_name in col_hints:
            if any(kw in q for kw in q_keywords):
                detected_col = col_name
                detected_df = col_df
                base["dataframe"] = col_df
                break

        numeric_value = _try_parse_number(words)

        if detected_col and numeric_value is not None:
            base.setdefault("filter", []).append({
                "dataframe": detected_df,
                "column": detected_col,
                "operator": detected_operator,
                "value": numeric_value,
            })
        elif detected_col and detected_operator in ("isnull", "notnull"):
            base.setdefault("filter", []).append({
                "dataframe": detected_df,
                "column": detected_col,
                "operator": detected_operator,
            })

    # ── 3. Boolean / availability filter ─────────────────────────────────────
    if any(k in q for k in ("available", "active")) and detected_operator is None:
        driver_df = dataframes.get("df_drivers")
        bool_col = _find_bool_col(list(driver_df.columns)) if driver_df is not None else None
        if bool_col:
            base.setdefault("filter", []).append(
                {"dataframe": "df_drivers", "column": bool_col,
                 "operator": "==", "value": True}
            )

    # ── 4. Aggregate — only for explicit count/sum/average/min/max requests ───
    if any(k in q for k in ("count", "how many", "total number")) and detected_operator is None:
        target_df_obj = dataframes.get(df_name)
        id_col = _find_id_col(list(target_df_obj.columns)) if target_df_obj is not None else "id"
        base["aggregate"] = {"column": f"{df_name}.{id_col}", "function": "count"}

    elif ("average" in q or "mean" in q) and detected_operator is None:
        for q_keywords, col_df, col_name in col_hints:
            if any(kw in q for kw in q_keywords):
                base["aggregate"] = {"column": f"{col_df}.{col_name}", "function": "mean"}
                break

    elif ("total" in q or "sum" in q) and detected_operator is None:
        for q_keywords, col_df, col_name in col_hints:
            if any(kw in q for kw in q_keywords):
                base["aggregate"] = {"column": f"{col_df}.{col_name}", "function": "sum"}
                break

    elif ("maximum" in q or "highest" in q or ("max" in words and detected_operator is None)):
        for q_keywords, col_df, col_name in col_hints:
            if any(kw in q for kw in q_keywords):
                base["aggregate"] = {"column": f"{col_df}.{col_name}", "function": "max"}
                break

    elif ("minimum" in q or "lowest" in q or ("min" in words and detected_operator is None)):
        for q_keywords, col_df, col_name in col_hints:
            if any(kw in q for kw in q_keywords):
                base["aggregate"] = {"column": f"{col_df}.{col_name}", "function": "min"}
                break

    # ── 5. Sort ───────────────────────────────────────────────────────────────
    if "sort" in q or "order" in q or "rank" in q:
        order = "asc" if any(k in q for k in (
            "asc", "ascending", "lowest first", "cheapest", "earliest"
        )) else "desc"
        for q_keywords, col_df, col_name in col_hints:
            if any(kw in q for kw in q_keywords):
                base["sort"] = {"column": f"{col_df}.{col_name}", "order": order}
                break

    # ── 6. Limit ──────────────────────────────────────────────────────────────────
    # 'top N' / 'first N' require a following number to trigger limit.
    # Bare 'top speed', 'top cost' etc. should NOT set a limit.
    # 'limit' as a bare word always triggers.
    _limit_n = _try_parse_number(words)
    _top_n_match = (
        "limit" in words
        or (_limit_n is not None and "top" in words)
        or (_limit_n is not None and "first" in words)
    )
    if _top_n_match:
        base["limit"] = int(_limit_n) if _limit_n is not None else 5

    return base
