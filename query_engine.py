"""
query_engine.py
---------------
LlamaIndex PandasQueryEngine-based query layer for VRP data.
Replaces the custom llm_layer.py + execution_layer.py pipeline.

LLM mode priority:
  1. Local LlamaCPP (MODEL_PATH env var or ~/models/qwen2.5-*.gguf)
  2. OpenAI (OPENAI_API_KEY env var)
  3. Demo stub (old rule-based fallback — no LLM required)

Each loaded DataFrame gets its own PandasQueryEngine instance.
A keyword router selects the right DataFrame per question.
Cross-entity questions get a pre-merged DataFrame on the fly.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Model path ────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(Path.home() / "models" / "qwen2.5-7b-instruct-q4_k_m.gguf"),
)


# ── LLM setup ─────────────────────────────────────────────────────────────────
def _create_llm():
    """
    Try LlamaCPP → OpenAI → None (demo mode).
    Returns an LlamaIndex LLM instance, or None for demo stub mode.
    """
    model_path = Path(DEFAULT_MODEL_PATH)

    # ── Option 1: Local LlamaCPP ─────────────────────────────────────────────
    if model_path.exists():
        try:
            from llama_index.llms.llama_cpp import LlamaCPP  # type: ignore
            logger.info("Loading local LlamaCPP model from %s …", model_path)
            llm = LlamaCPP(
                model_path=str(model_path),
                temperature=0.1,
                max_new_tokens=512,
                context_window=4096,
                verbose=False,
            )
            logger.info("✓ LlamaCPP model loaded successfully.")
            return llm
        except ImportError:
            logger.warning(
                "llama-index-llms-llama-cpp is not installed.\n"
                "  Run: pip install llama-index-llms-llama-cpp\n"
                "  Falling back to next LLM option."
            )
        except Exception as exc:
            logger.warning(
                "Failed to load LlamaCPP model (%s: %s). Trying next option.",
                type(exc).__name__, exc,
            )
    else:
        if DEFAULT_MODEL_PATH != str(Path.home() / "models" / "qwen2.5-7b-instruct-q4_k_m.gguf"):
            # User explicitly set MODEL_PATH but the file doesn't exist — warn them
            logger.warning(
                "MODEL_PATH is set to '%s' but file does not exist. "
                "Skipping LlamaCPP.",
                DEFAULT_MODEL_PATH,
            )

    # ── Option 2: OpenAI ─────────────────────────────────────────────────────
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from llama_index.llms.openai import OpenAI  # type: ignore
            logger.info("Using OpenAI LLM (gpt-3.5-turbo)")
            return OpenAI(model="gpt-3.5-turbo", temperature=0.1)
        except ImportError:
            logger.warning(
                "llama-index-llms-openai is not installed.\n"
                "  Run: pip install llama-index-llms-openai\n"
                "  Falling back to demo stub mode."
            )
        except Exception as exc:
            logger.warning(
                "Failed to init OpenAI LLM (%s: %s). Falling back to demo stub.",
                type(exc).__name__, exc,
            )

    # ── Option 3: Demo stub ───────────────────────────────────────────────────
    logger.warning(
        "\n"
        "  ╔══════════════════════════════════════════════════════════════╗\n"
        "  ║  ⚠  No LLM configured — running in DEMO STUB mode          ║\n"
        "  ║                                                              ║\n"
        "  ║  To enable real NL queries, do ONE of:                      ║\n"
        "  ║  • Local model:  export MODEL_PATH=/path/to/model.gguf      ║\n"
        "  ║                  pip install llama-index-llms-llama-cpp      ║\n"
        "  ║  • OpenAI:       export OPENAI_API_KEY=sk-...               ║\n"
        "  ╚══════════════════════════════════════════════════════════════╝\n"
    )
    return None


# ── Instruction builder ───────────────────────────────────────────────────────
def _build_instruction(df_name: str, schema: dict) -> str:
    """
    Build a rich PandasQueryEngine instruction string from the dynamic schema.
    Uses actual column names (whatever case they are) so the LLM uses them
    correctly in generated code.
    """
    df_info = schema.get("dataframes", {}).get(df_name, {})
    entity = df_info.get("entity", df_name)
    columns = df_info.get("columns", {})

    col_lines = []
    for col, cinfo in columns.items():
        samples = ", ".join(str(s) for s in cinfo.get("sample_values", [])[:3])
        col_lines.append(f"  - {col}  [{cinfo['dtype']}]  e.g. {samples}")
    col_text = "\n".join(col_lines)

    return (
        f"The pandas DataFrame `df` represents {entity}.\n"
        f"Column names are CASE-SENSITIVE — copy them exactly as shown:\n"
        f"{col_text}\n\n"
        "Given a user question, write a single valid Python/pandas expression "
        "that evaluates to a DataFrame, Series, or scalar.\n"
        "Return ONLY the expression — no explanation, no markdown, no code fences.\n"
        "Never invent column names. Only use names listed above."
    )


# ── DataFrame routing ─────────────────────────────────────────────────────────
def _select_dataframe(question: str) -> str:
    """Route a question to the most likely target DataFrame."""
    q = question.lower()
    if any(k in q for k in ("restrict", "restriction")):
        return "df_vehicle_driver_restrictions"
    if any(k in q for k in ("vehicle", "truck", "van")):
        return "df_vehicles"
    return "df_drivers"


def _detect_join_need(question: str) -> bool:
    """Return True if the question seems to span multiple entities."""
    q = question.lower()
    has_drivers = any(k in q for k in ("driver", "drivers"))
    has_vehicles = any(k in q for k in ("vehicle", "vehicles", "truck", "van"))
    has_restrictions = any(k in q for k in ("restrict", "restriction"))
    return sum([has_drivers, has_vehicles, has_restrictions]) >= 2


def _build_joined_df(dataframes: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    """
    Merge all three DataFrames on detected shared keys.
    Returns (merged_df_name, merged_DataFrame).
    """
    df_drivers = dataframes.get("df_drivers", pd.DataFrame())
    df_vehicles = dataframes.get("df_vehicles", pd.DataFrame())
    df_restr = dataframes.get("df_vehicle_driver_restrictions", pd.DataFrame())

    # Find driver join key
    driver_cols = set(df_drivers.columns)
    restr_cols = set(df_restr.columns)
    shared_driver = sorted(driver_cols & restr_cols)
    driver_key = shared_driver[0] if shared_driver else None

    # Find vehicle join key
    vehicle_cols = set(df_vehicles.columns)
    shared_vehicle = sorted(vehicle_cols & restr_cols)
    vehicle_key = shared_vehicle[0] if shared_vehicle else None

    merged = df_restr.copy()
    if driver_key and not df_drivers.empty:
        merged = merged.merge(df_drivers, on=driver_key, how="left", suffixes=("", "_driver"))
    if vehicle_key and not df_vehicles.empty:
        merged = merged.merge(df_vehicles, on=vehicle_key, how="left", suffixes=("", "_vehicle"))

    return "df_joined", merged


# ── Safe code execution ───────────────────────────────────────────────────────
def _safe_exec(code: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely execute the pandas expression generated by PandasQueryEngine.
    Returns a DataFrame regardless of whether the result is a scalar or Series.
    """
    # Strip markdown fences if the LLM added them despite instructions
    code = re.sub(r"```(?:python)?\s*", "", code).strip("`").strip()

    local_vars: dict[str, Any] = {"df": df, "pd": pd}
    try:
        result = eval(code, {"__builtins__": {}}, local_vars)  # noqa: S307
    except Exception as exc:
        raise ValueError(f"Pandas code execution failed: {exc}\nCode: {code}") from exc

    if isinstance(result, pd.Series):
        return result.to_frame()
    if isinstance(result, pd.DataFrame):
        return result
    # Scalar / list → wrap
    return pd.DataFrame([{"result": result}])


# ── Main engine class ─────────────────────────────────────────────────────────
class VRPQueryEngine:
    """
    LlamaIndex PandasQueryEngine wrapper for VRP data.

    Usage::
        engine = VRPQueryEngine()
        result = engine.query("Show drivers with shift duration above 6 hours", dfs)
    """

    def __init__(self):
        self._llm = _create_llm()
        self._engine_cache: dict = {}   # (df_name, df_hash) → PandasQueryEngine

    # ── internal helpers ──────────────────────────────────────────────────────
    def _get_pandas_engine(self, df_name: str, df: pd.DataFrame, schema: dict):
        try:
            from llama_index.experimental.query_engine import PandasQueryEngine  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "llama-index-experimental is not installed.\n"
                "Run: pip install llama-index-experimental"
            ) from exc

        cache_key = (df_name, id(df), len(df))
        if cache_key not in self._engine_cache:
            instruction = _build_instruction(df_name, schema)
            self._engine_cache[cache_key] = PandasQueryEngine(
                df=df,
                llm=self._llm,
                instruction_str=instruction,
                verbose=True,
                synthesize_response=False,
            )
        return self._engine_cache[cache_key]

    def _demo_stub_query(self, question: str, dataframes: dict) -> dict:
        """Rule-based fallback when no LLM is available."""
        from llm_layer import generate_query_stub
        from execution_layer import execute_query

        try:
            structured = generate_query_stub(question, "", dataframes=dataframes)
            result_df = execute_query(structured, dataframes)
            return {
                "dataframe": result_df,
                "pandas_code": f"# demo stub — structured query: {structured}",
                "response_text": f"{len(result_df)} row(s) returned (demo mode)",
                "df_name": structured.get("dataframe", "unknown"),
                "mode": "demo_stub",
            }
        except Exception as exc:
            return {
                "dataframe": pd.DataFrame(),
                "pandas_code": "",
                "response_text": str(exc),
                "df_name": "unknown",
                "error": str(exc),
                "mode": "demo_stub",
            }

    # ── public API ────────────────────────────────────────────────────────────
    def query(
        self,
        question: str,
        dataframes: dict[str, pd.DataFrame] | None = None,
    ) -> dict:
        """
        Query VRP data using natural language.

        Returns a dict with:
          dataframe     — result as pd.DataFrame
          pandas_code   — generated pandas expression (for transparency)
          response_text — string answer from LLM
          df_name       — which DataFrame was queried
          mode          — 'llm' | 'demo_stub'
          error         — present only on failure
        """
        from data_layer import get_dataframes
        from schema_layer import build_schema

        if dataframes is None:
            dataframes = get_dataframes()

        # Demo mode — no LLM
        if self._llm is None:
            return self._demo_stub_query(question, dataframes)

        schema = build_schema(dataframes)

        # ── select DataFrame(s) ───────────────────────────────────────────────
        if _detect_join_need(question):
            df_name, df = _build_joined_df(dataframes)
            # For joined DF, build a minimal ad-hoc schema entry
            schema.setdefault("dataframes", {})[df_name] = {
                "entity": "joined drivers, vehicles, and restrictions",
                "aliases": [],
                "row_count": len(df),
                "columns": {
                    col: {
                        "dtype": str(df[col].dtype),
                        "sample_values": df[col].dropna().head(3).tolist(),
                        "description": "",
                    }
                    for col in df.columns
                },
            }
        else:
            df_name = _select_dataframe(question)
            df = dataframes.get(df_name, pd.DataFrame())

        if df is None or df.empty:
            return {
                "dataframe": pd.DataFrame(),
                "pandas_code": "",
                "response_text": f"No data available in '{df_name}'.",
                "df_name": df_name,
                "error": f"DataFrame '{df_name}' is empty or not loaded.",
                "mode": "llm",
            }

        # ── run PandasQueryEngine ─────────────────────────────────────────────
        try:
            engine = self._get_pandas_engine(df_name, df, schema)
            response = engine.query(question)
        except Exception as exc:
            logger.error("PandasQueryEngine failed: %s", exc)
            # Fall back to demo stub on LLM error
            return self._demo_stub_query(question, dataframes)

        pandas_code: str = response.metadata.get("pandas_instruction_str", "")
        response_text: str = str(response).strip()

        # ── execute generated code to get DataFrame ───────────────────────────
        try:
            result_df = _safe_exec(pandas_code, df)
        except ValueError as exc:
            return {
                "dataframe": pd.DataFrame(),
                "pandas_code": pandas_code,
                "response_text": response_text,
                "df_name": df_name,
                "error": str(exc),
                "mode": "llm",
            }

        return {
            "dataframe": result_df.reset_index(drop=True),
            "pandas_code": pandas_code,
            "response_text": response_text,
            "df_name": df_name,
            "mode": "llm",
        }

    def invalidate_cache(self):
        """Call after loading new data to force engine rebuild."""
        self._engine_cache.clear()


# ── Module-level singleton ────────────────────────────────────────────────────
_instance: VRPQueryEngine | None = None


def get_engine() -> VRPQueryEngine:
    global _instance
    if _instance is None:
        _instance = VRPQueryEngine()
    return _instance


def reset_engine():
    """Invalidate cached engines (call after /load). LLM singleton is kept."""
    global _instance
    if _instance is not None:
        _instance.invalidate_cache()
    logger.info("Query engine cache cleared — engines will rebuild on next query.")
