"""
main.py
-------
FastAPI application exposing:
  POST /load    — upload a VRP JSON file
  POST /query   — natural-language question → table results
  GET  /schema  — inspect the current schema
  GET  /        — serves the embedded chat UI
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import data_layer
import schema_layer
import query_engine          # ← PandasQueryEngine layer
import llm_layer            # kept for demo stub (used by query_engine internally)
import execution_layer      # kept for demo stub (used by query_engine internally)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="VRP Query Chatbot", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Startup: load sample data & build schema ──────────────────────────────────
@app.on_event("startup")
async def startup_event():
    dfs = data_layer.load_vrp_json(None)
    schema = schema_layer.build_schema(dfs)
    schema_layer.save_schema(schema)
    logger.info("Sample VRP data loaded. Schema saved to %s", schema_layer.SCHEMA_PATH)


# ── /load ─────────────────────────────────────────────────────────────────────
@app.post("/load")
async def load_vrp(file: UploadFile = File(...)):
    """Accept a VRP JSON file, parse it, and regenerate the schema."""
    try:
        contents = await file.read()
        vrp_data = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")

    dfs = data_layer.load_vrp_json(vrp_data)
    schema = schema_layer.build_schema(dfs)
    path = schema_layer.save_schema(schema)

    # Invalidate cached per-DataFrame engines so next query uses fresh data
    query_engine.reset_engine()

    summary = {
        df_name: {"rows": len(df), "columns": list(df.columns)}
        for df_name, df in dfs.items()
    }
    return {"status": "loaded", "schema_saved_to": str(path), "dataframes": summary}


# ── /schema ───────────────────────────────────────────────────────────────────
@app.get("/schema")
async def get_schema():
    """Return the current schema dict."""
    dfs = data_layer.get_dataframes()
    schema = schema_layer.build_schema(dfs)
    return JSONResponse(content=schema)


# ── /query ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    use_stub: bool = False   # force demo stub even if LLM is available


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame to JSON-serialisable records."""
    return json.loads(df.to_json(orient="records", default_handler=str))


@app.post("/query")
async def run_query(req: QueryRequest):
    """
    Convert a natural-language question into a pandas query via PandasQueryEngine,
    execute it, and return results.
    """
    dfs = data_layer.get_dataframes()

    # ── Force demo stub if requested ──────────────────────────────────────────
    if req.use_stub:
        from llm_layer import generate_query_stub
        from execution_layer import execute_query
        schema_text = schema_layer.get_schema_for_prompt(schema_layer.build_schema(dfs))
        try:
            sq = llm_layer.generate_query_stub(req.question, schema_text, dataframes=dfs)
            result_df = execution_layer.execute_query(sq, dfs)
            records = _df_to_records(result_df)
            return {
                "status": "ok",
                "question": req.question,
                "pandas_code": f"# stub: {sq}",
                "response_text": f"{len(records)} row(s) returned (stub mode)",
                "row_count": len(records),
                "columns": list(result_df.columns),
                "results": records,
                "mode": "demo_stub",
            }
        except ValueError as exc:
            return JSONResponse(status_code=422, content={
                "status": "validation_error",
                "question": req.question,
                "error": str(exc),
                "mode": "demo_stub",
            })

    # ── PandasQueryEngine path ─────────────────────────────────────────────────
    try:
        engine = query_engine.get_engine()
        result = engine.query(req.question, dfs)
    except Exception as exc:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={
            "status": "engine_error",
            "question": req.question,
            "error": str(exc),
        })

    if "error" in result:
        return JSONResponse(status_code=422, content={
            "status": "query_error",
            "question": req.question,
            "pandas_code": result.get("pandas_code", ""),
            "response_text": result.get("response_text", ""),
            "error": result["error"],
            "mode": result.get("mode", "llm"),
        })

    result_df: pd.DataFrame = result["dataframe"]
    records = _df_to_records(result_df)
    return {
        "status": "ok",
        "question": req.question,
        "pandas_code": result.get("pandas_code", ""),
        "response_text": result.get("response_text", ""),
        "row_count": len(records),
        "columns": list(result_df.columns),
        "results": records,
        "mode": result.get("mode", "llm"),
    }


# ── Chat UI ───────────────────────────────────────────────────────────────────
CHAT_UI_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>VRP Query Chatbot | EngineBot</title>
  <meta name="description" content="Conversational interface for querying Vehicle Routing Problem data using natural language." />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <style>
    /* ── Design Tokens ───────────────────────────────────────────────────── */
    :root {
      --bg-base:    #0d0f17;
      --bg-surface: #13172a;
      --bg-card:    #1a1f38;
      --bg-input:   #1e2440;
      --accent-1:   #6c63ff;
      --accent-2:   #a78bfa;
      --accent-3:   #38bdf8;
      --text-primary:   #e8eaf6;
      --text-secondary: #8891b3;
      --text-muted:     #4a5280;
      --success:  #34d399;
      --warning:  #fbbf24;
      --error:    #f87171;
      --border:   rgba(108,99,255,0.2);
      --glow:     rgba(108,99,255,0.15);
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
      --shadow-card: 0 4px 32px rgba(0,0,0,0.4);
      --transition: 0.2s ease;
      --font-sans: 'Inter', system-ui, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    /* ── Reset & Base ────────────────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { height: 100%; }
    body {
      font-family: var(--font-sans);
      background: var(--bg-base);
      color: var(--text-primary);
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      overflow-x: hidden;
    }
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background:
        radial-gradient(ellipse 80% 50% at 20% -20%, rgba(108,99,255,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(56,189,248,0.08) 0%, transparent 60%);
      pointer-events: none;
      z-index: 0;
    }

    /* ── Layout ──────────────────────────────────────────────────────────── */
    #app {
      position: relative;
      z-index: 1;
      display: flex;
      flex-direction: column;
      width: 100%;
      max-width: 960px;
      height: 100vh;
      padding: 0 16px;
    }

    /* ── Header ──────────────────────────────────────────────────────────── */
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 20px 0 16px;
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
    }
    .logo {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .logo-icon {
      width: 38px; height: 38px;
      background: linear-gradient(135deg, var(--accent-1), var(--accent-3));
      border-radius: var(--radius-sm);
      display: flex; align-items: center; justify-content: center;
      font-size: 20px;
      box-shadow: 0 0 20px rgba(108,99,255,0.4);
    }
    .logo-text { display: flex; flex-direction: column; }
    .logo-title {
      font-size: 17px; font-weight: 700;
      background: linear-gradient(90deg, var(--accent-2), var(--accent-3));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .logo-sub { font-size: 11px; color: var(--text-muted); font-weight: 400; letter-spacing: 0.04em; }

    .header-actions { display: flex; align-items: center; gap: 10px; }

    .chip {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 5px 12px;
      border-radius: 100px;
      font-size: 12px; font-weight: 500;
      border: 1px solid var(--border);
      background: var(--bg-card);
      color: var(--text-secondary);
      cursor: default;
      transition: var(--transition);
    }
    .chip.active { border-color: var(--success); color: var(--success); background: rgba(52,211,153,0.08); }
    .chip-dot {
      width: 6px; height: 6px; border-radius: 50%;
      background: currentColor;
      animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

    /* ── Upload bar ──────────────────────────────────────────────────────── */
    #upload-bar {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 14px 0;
      flex-shrink: 0;
    }
    #file-input { display: none; }
    .btn {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 9px 18px;
      border-radius: var(--radius-sm);
      font-size: 13px; font-weight: 500;
      border: none; cursor: pointer;
      transition: var(--transition);
      font-family: var(--font-sans);
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent-1), #7c6bff);
      color: #fff;
      box-shadow: 0 2px 16px rgba(108,99,255,0.35);
    }
    .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 24px rgba(108,99,255,0.5); }
    .btn-primary:active { transform: translateY(0); }
    .btn-ghost {
      background: var(--bg-card);
      color: var(--text-secondary);
      border: 1px solid var(--border);
    }
    .btn-ghost:hover { border-color: var(--accent-1); color: var(--accent-2); }

    #file-label {
      font-size: 12px; color: var(--text-muted);
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 220px;
    }
    .upload-status {
      margin-left: auto;
      font-size: 12px; font-weight: 500; padding: 4px 10px; border-radius: 100px;
    }
    .upload-status.ok  { color: var(--success); background: rgba(52,211,153,0.1); }
    .upload-status.err { color: var(--error);   background: rgba(248,113,113,0.1); }

    /* ── Chat window ─────────────────────────────────────────────────────── */
    #chat-window {
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 20px;
      padding: 16px 0;
      scroll-behavior: smooth;
    }
    #chat-window::-webkit-scrollbar { width: 4px; }
    #chat-window::-webkit-scrollbar-track { background: transparent; }
    #chat-window::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 4px; }

    /* ── Messages ────────────────────────────────────────────────────────── */
    .msg { display: flex; gap: 12px; animation: fadeUp 0.3s ease; }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(10px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    .msg.user { flex-direction: row-reverse; }

    .msg-avatar {
      width: 32px; height: 32px; border-radius: var(--radius-sm);
      display: flex; align-items: center; justify-content: center;
      font-size: 15px; flex-shrink: 0;
    }
    .msg.user   .msg-avatar { background: linear-gradient(135deg,var(--accent-1),var(--accent-2)); }
    .msg.system .msg-avatar { background: linear-gradient(135deg,#1e3a5f,var(--accent-3)); }
    .msg.bot    .msg-avatar { background: linear-gradient(135deg,var(--bg-card),#2a2060); border:1px solid var(--border); }

    .msg-body { display: flex; flex-direction: column; gap: 6px; max-width: 82%; }
    .msg.user .msg-body { align-items: flex-end; }

    .msg-bubble {
      padding: 12px 16px;
      border-radius: var(--radius-md);
      font-size: 14px; line-height: 1.6;
    }
    .msg.user   .msg-bubble {
      background: linear-gradient(135deg, rgba(108,99,255,0.25), rgba(124,107,255,0.18));
      border: 1px solid rgba(108,99,255,0.35);
      border-bottom-right-radius: 4px;
      color: var(--text-primary);
    }
    .msg.system .msg-bubble {
      background: rgba(56,189,248,0.07);
      border: 1px solid rgba(56,189,248,0.2);
      border-bottom-left-radius: 4px;
      color: var(--text-secondary);
      font-style: italic; font-size: 13px;
    }
    .msg.bot .msg-bubble {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-bottom-left-radius: 4px;
      color: var(--text-primary);
    }

    .msg-meta { font-size: 11px; color: var(--text-muted); }

    /* ── Query badge ─────────────────────────────────────────────────────── */
    .query-badge {
      background: rgba(108,99,255,0.08);
      border: 1px solid rgba(108,99,255,0.2);
      border-radius: var(--radius-sm);
      padding: 8px 12px;
      font-size: 12px;
      font-family: var(--font-mono);
      color: var(--accent-2);
      white-space: pre-wrap; word-break: break-all;
      max-height: 120px; overflow-y: auto;
    }

    /* ── Result table ────────────────────────────────────────────────────── */
    .result-wrapper {
      overflow-x: auto;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
      background: var(--bg-surface);
    }
    table.result-table {
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }
    .result-table th {
      background: rgba(108,99,255,0.15);
      color: var(--accent-2);
      font-weight: 600;
      padding: 9px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
      font-family: var(--font-mono);
      font-size: 12px;
    }
    .result-table td {
      padding: 8px 14px;
      color: var(--text-primary);
      border-bottom: 1px solid rgba(255,255,255,0.04);
      white-space: nowrap;
    }
    .result-table tr:last-child td { border-bottom: none; }
    .result-table tr:hover td { background: rgba(108,99,255,0.06); }

    .result-meta {
      font-size: 12px; color: var(--text-muted);
      padding: 6px 0 0;
    }

    /* ── Error bubble ────────────────────────────────────────────────────── */
    .error-bubble {
      background: rgba(248,113,113,0.08);
      border: 1px solid rgba(248,113,113,0.25);
      border-radius: var(--radius-sm);
      padding: 10px 14px;
      color: var(--error);
      font-size: 13px;
      white-space: pre-wrap;
    }

    /* ── Typing indicator ────────────────────────────────────────────────── */
    .typing-dots {
      display: inline-flex; gap: 4px; align-items: center; padding: 4px 0;
    }
    .typing-dots span {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--accent-2);
      animation: bounce 1.2s ease-in-out infinite;
    }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-6px)} }

    /* ── Input area ──────────────────────────────────────────────────────── */
    #input-area {
      display: flex;
      align-items: flex-end;
      gap: 10px;
      padding: 14px 0 20px;
      flex-shrink: 0;
      border-top: 1px solid var(--border);
    }
    #question-input {
      flex: 1;
      resize: none;
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      color: var(--text-primary);
      font-family: var(--font-sans);
      font-size: 14px;
      padding: 12px 16px;
      line-height: 1.5;
      min-height: 48px;
      max-height: 140px;
      outline: none;
      transition: var(--transition);
    }
    #question-input::placeholder { color: var(--text-muted); }
    #question-input:focus {
      border-color: var(--accent-1);
      box-shadow: 0 0 0 3px rgba(108,99,255,0.15);
    }

    #send-btn {
      width: 44px; height: 44px; border-radius: var(--radius-sm); padding: 0;
      display: flex; align-items: center; justify-content: center;
      background: linear-gradient(135deg, var(--accent-1), #7c6bff);
      color: #fff; border: none; cursor: pointer;
      box-shadow: 0 2px 14px rgba(108,99,255,0.4);
      transition: var(--transition); flex-shrink: 0;
    }
    #send-btn:hover { transform: translateY(-1px) scale(1.05); box-shadow: 0 4px 24px rgba(108,99,255,0.55); }
    #send-btn:disabled { opacity: 0.45; pointer-events: none; }
    #send-btn svg { width: 20px; height: 20px; }

    /* ── Suggestions ─────────────────────────────────────────────────────── */
    #suggestions {
      display: flex; flex-wrap: wrap; gap: 8px;
      padding: 0 0 12px;
      flex-shrink: 0;
    }
    .sugg-chip {
      padding: 6px 14px;
      border-radius: 100px;
      font-size: 12px; font-weight: 500;
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: var(--text-secondary);
      cursor: pointer;
      transition: var(--transition);
    }
    .sugg-chip:hover { border-color: var(--accent-1); color: var(--accent-2); background: rgba(108,99,255,0.08); }

    /* ── Scrolled-away floater ───────────────────────────────────────────── */
    #scroll-btn {
      position: fixed; bottom: 90px; right: calc(50% - 480px + 20px);
      background: var(--bg-card); border: 1px solid var(--border);
      color: var(--text-secondary); border-radius: 50%;
      width: 36px; height: 36px;
      display: none; align-items: center; justify-content: center;
      cursor: pointer; font-size: 18px; box-shadow: var(--shadow-card);
      transition: var(--transition);
    }
    #scroll-btn:hover { border-color: var(--accent-1); color: var(--accent-2); }

    @media (max-width: 640px) {
      .logo-sub { display: none; }
      #suggestions { display: none; }
      #scroll-btn { right: 12px; }
    }
  </style>
</head>
<body>
<div id="app">

  <!-- Header -->
  <header>
    <div class="logo">
      <div class="logo-icon">🚚</div>
      <div class="logo-text">
        <span class="logo-title">VRP Query Chatbot</span>
        <span class="logo-sub">EngineBot • Powered by Qwen2.5</span>
      </div>
    </div>
    <div class="header-actions">
      <div class="chip active" id="status-chip">
        <div class="chip-dot"></div>
        <span id="status-label">Ready</span>
      </div>
      <button class="btn btn-ghost" id="schema-btn" onclick="showSchema()">🔍 Schema</button>
    </div>
  </header>

  <!-- Upload bar -->
  <div id="upload-bar">
    <input type="file" id="file-input" accept=".json" onchange="handleFileSelect(this)" />
    <button class="btn btn-primary" onclick="document.getElementById('file-input').click()">
      📂 Load VRP JSON
    </button>
    <span id="file-label">No file selected — using sample data</span>
    <span class="upload-status ok" id="upload-status">● Sample loaded</span>
  </div>

  <!-- Suggestions -->
  <div id="suggestions">
    <span class="sugg-chip" onclick="askQuestion('List all available drivers')">👤 Available drivers</span>
    <span class="sugg-chip" onclick="askQuestion('Show all vehicles with capacity above 3000 kg')">🚛 High-capacity vehicles</span>
    <span class="sugg-chip" onclick="askQuestion('How many drivers are there in total?')">🔢 Count drivers</span>
    <span class="sugg-chip" onclick="askQuestion('Show all driver-vehicle restrictions')">🚫 All restrictions</span>
    <span class="sugg-chip" onclick="askQuestion('Which vehicle has the lowest cost per km?')">💰 Cheapest vehicle</span>
    <span class="sugg-chip" onclick="askQuestion('List drivers who start earliest')">⏰ Early starters</span>
  </div>

  <!-- Chat window -->
  <div id="chat-window">
    <div class="msg system">
      <div class="msg-avatar">ℹ️</div>
      <div class="msg-body">
        <div class="msg-bubble">
          Sample VRP data is pre-loaded with <strong>3 drivers</strong>, <strong>3 vehicles</strong>, and <strong>3 restrictions</strong>.
          Upload your own JSON file or ask a question to get started.
        </div>
        <span class="msg-meta">System • just now</span>
      </div>
    </div>
  </div>

  <!-- Input area -->
  <div id="input-area">
    <textarea
      id="question-input"
      placeholder="Ask a question about your VRP data…"
      rows="1"
      onkeydown="handleKey(event)"
      oninput="autoResize(this)"
    ></textarea>
    <button id="send-btn" onclick="sendQuestion()" title="Send (Enter)">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="22" y1="2" x2="11" y2="13"/>
        <polygon points="22 2 15 22 11 13 2 9 22 2"/>
      </svg>
    </button>
  </div>
</div>

<!-- Scroll-to-bottom -->
<button id="scroll-btn" onclick="scrollToBottom()">↓</button>

<script>
/* ── State ─────────────────────────────────────────────────────────────────── */
let isLoading = false;

/* ── File upload ───────────────────────────────────────────────────────────── */
async function handleFileSelect(input) {
  const file = input.files[0];
  if (!file) return;
  document.getElementById('file-label').textContent = file.name;
  setStatus('Uploading…', false);

  const fd = new FormData();
  fd.append('file', file);

  try {
    const res = await fetch('/load', { method: 'POST', body: fd });
    const data = await res.json();
    if (res.ok) {
      document.getElementById('upload-status').textContent = '● Loaded';
      document.getElementById('upload-status').className = 'upload-status ok';
      addSystemMsg(`✅ Loaded <strong>${file.name}</strong> — ` +
        Object.entries(data.dataframes).map(([k,v]) => `${k}: ${v.rows} rows`).join(', '));
      setStatus('Ready', true);
    } else {
      throw new Error(data.detail || JSON.stringify(data));
    }
  } catch (err) {
    document.getElementById('upload-status').textContent = '● Error';
    document.getElementById('upload-status').className = 'upload-status err';
    addSystemMsg(`❌ Upload failed: ${err.message}`);
    setStatus('Error', false);
  }
}

/* ── Schema viewer ─────────────────────────────────────────────────────────── */
async function showSchema() {
  try {
    const res = await fetch('/schema');
    const data = await res.json();
    const lines = [];
    for (const [dfName, dfInfo] of Object.entries(data.dataframes || {})) {
      lines.push(`<strong>${dfName}</strong> (${dfInfo.entity}, ${dfInfo.row_count} rows)`);
      for (const [col, cinfo] of Object.entries(dfInfo.columns || {})) {
        const samples = (cinfo.sample_values || []).join(', ');
        lines.push(`  <span style="color:var(--accent-2)">${col}</span> [${cinfo.dtype}] → ${samples}`);
      }
      lines.push('');
    }
    addBotMsg(`<div style="font-family:var(--font-mono);font-size:12px;white-space:pre-wrap;line-height:1.8">${lines.join('\n')}</div>`, null, null);
  } catch (err) {
    addBotMsg(null, null, 'Failed to load schema: ' + err.message);
  }
}

/* ── Textarea auto-resize ──────────────────────────────────────────────────── */
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
}

function askQuestion(q) {
  document.getElementById('question-input').value = q;
  sendQuestion();
}

/* ── Send query ────────────────────────────────────────────────────────────── */
async function sendQuestion() {
  const input = document.getElementById('question-input');
  const question = input.value.trim();
  if (!question || isLoading) return;

  input.value = '';
  autoResize(input);
  addUserMsg(question);
  const typingId = addTyping();

  isLoading = true;
  document.getElementById('send-btn').disabled = true;
  setStatus('Thinking…', false);

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, use_stub: false }),
    });
    const data = await res.json();
    removeTyping(typingId);

    if (data.status === 'ok') {
      addBotMsg(null, data);
    } else {
      addBotMsg(null, data, data.error || 'Unknown error');
    }
  } catch (err) {
    removeTyping(typingId);
    addBotMsg(null, null, err.message);
  } finally {
    isLoading = false;
    document.getElementById('send-btn').disabled = false;
    setStatus('Ready', true);
  }
}

/* ── Message helpers ───────────────────────────────────────────────────────── */
function now() {
  return new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
}

function addUserMsg(text) {
  const html = `
    <div class="msg user">
      <div class="msg-avatar">👤</div>
      <div class="msg-body">
        <div class="msg-bubble">${escHtml(text)}</div>
        <span class="msg-meta">${now()}</span>
      </div>
    </div>`;
  appendToChat(html);
}

function addSystemMsg(html) {
  const el = `
    <div class="msg system">
      <div class="msg-avatar">ℹ️</div>
      <div class="msg-body">
        <div class="msg-bubble">${html}</div>
        <span class="msg-meta">System • ${now()}</span>
      </div>
    </div>`;
  appendToChat(el);
}

function addBotMsg(customHtml, data, errorText) {
  let inner = '';

  if (errorText) {
    inner += `<div class="error-bubble">⚠️ ${escHtml(errorText)}</div>`;
  }

  if (data && data.pandas_code) {
    inner += `<div class="query-badge">${escHtml(data.pandas_code)}</div>`;
  }

  if (data && data.response_text && !errorText) {
    inner += `<div style="font-size:13px;color:var(--text-secondary);margin:6px 0 4px;font-style:italic">${escHtml(data.response_text)}</div>`;
  }

  if (data && data.results && data.results.length > 0) {
    inner += buildTable(data.columns, data.results);
    inner += `<div class="result-meta">${data.row_count} row${data.row_count !== 1 ? 's' : ''} returned</div>`;
  } else if (data && data.status === 'ok' && data.results && data.results.length === 0) {
    inner += `<p style="color:var(--text-muted);font-size:13px">No results found.</p>`;
  }

  if (customHtml) {
    inner += customHtml;
  }

  if (!inner) inner = `<p style="color:var(--text-muted);font-size:13px">No data returned.</p>`;

  const el = `
    <div class="msg bot">
      <div class="msg-avatar">🤖</div>
      <div class="msg-body" style="max-width:100%;width:100%">
        <div class="msg-bubble" style="max-width:100%">${inner}</div>
        <span class="msg-meta">EngineBot • ${now()}</span>
      </div>
    </div>`;
  appendToChat(el);
}

function buildTable(columns, rows) {
  const headers = columns.map(c => `<th>${escHtml(c)}</th>`).join('');
  const bodyRows = rows.map(row =>
    `<tr>${columns.map(c => `<td>${escHtml(String(row[c] ?? ''))}</td>`).join('')}</tr>`
  ).join('');
  return `<div class="result-wrapper"><table class="result-table"><thead><tr>${headers}</tr></thead><tbody>${bodyRows}</tbody></table></div>`;
}

function addTyping() {
  const id = 'typing-' + Date.now();
  const el = `
    <div class="msg bot" id="${id}">
      <div class="msg-avatar">🤖</div>
      <div class="msg-body">
        <div class="msg-bubble">
          <div class="typing-dots"><span></span><span></span><span></span></div>
        </div>
      </div>
    </div>`;
  appendToChat(el);
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendToChat(html) {
  const cw = document.getElementById('chat-window');
  cw.insertAdjacentHTML('beforeend', html);
  scrollToBottom();
}

function scrollToBottom() {
  const cw = document.getElementById('chat-window');
  cw.scrollTop = cw.scrollHeight;
}

function setStatus(label, active) {
  document.getElementById('status-label').textContent = label;
  document.getElementById('status-chip').className = 'chip' + (active ? ' active' : '');
}

function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}

/* ── Scroll observer ───────────────────────────────────────────────────────── */
const cw = document.getElementById('chat-window');
cw.addEventListener('scroll', () => {
  const btn = document.getElementById('scroll-btn');
  const atBottom = cw.scrollHeight - cw.scrollTop - cw.clientHeight < 80;
  btn.style.display = atBottom ? 'none' : 'flex';
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return CHAT_UI_HTML
