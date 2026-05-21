# VRP Query Chatbot — EngineBot MVP

A conversational interface for querying **Vehicle Routing Problem (VRP)** data using natural language, powered by a local Qwen2.5-7B GGUF model.

---

## Architecture

```
main.py               FastAPI server + chat UI
data_layer.py         JSON → pandas DataFrames (dynamic, schema-less)
schema_layer.py       Schema inference + YAML persistence
llm_layer.py          Qwen2.5 GGUF prompt building + JSON extraction
execution_layer.py    Structured query validation + execution
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install fastapi "uvicorn[standard]" pydantic pandas PyYAML python-multipart
```

### 2. Install llama-cpp-python (for the real LLM)

```bash
# CPU only
pip install llama-cpp-python

# macOS Metal (GPU)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall

# CUDA (Linux/Windows)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall
```

### 3. Download the Qwen2.5-7B GGUF model

```bash
mkdir -p ~/models
# Download from HuggingFace (Qwen/Qwen2.5-7B-Instruct-GGUF)
# Recommended: qwen2.5-7b-instruct-q4_k_m.gguf  (~4.4 GB)
```

Set a custom path if needed:
```bash
export MODEL_PATH=/path/to/your/model.gguf
```

### 4. Run the server

```bash
cd /Users/snimgole/Documents/EngineBot
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/`  | Chat UI |
| `POST` | `/load` | Upload a VRP JSON file |
| `POST` | `/query` | Ask a natural-language question |
| `GET`  | `/schema` | Inspect current schema |

### POST /query body

```json
{
  "question": "Show all available drivers",
  "use_stub": false
}
```

Set `use_stub: true` to use the rule-based fallback without a model file.

---

## Structured Query Format

The LLM produces (and the execution layer consumes) JSON in this shape:

```json
{
  "dataframe": "df_drivers",
  "join": { "left": "df_drivers", "right": "df_vehicle_driver_restrictions", "on": "id" },
  "filter": [
    { "dataframe": "df_drivers", "column": "available", "operator": "==", "value": true }
  ],
  "select": ["df_drivers.name", "df_drivers.license_class"],
  "aggregate": { "column": "df_vehicles.capacity_kg", "function": "max" },
  "sort": { "column": "df_drivers.max_hours", "order": "desc" },
  "limit": 10
}
```

**Allowed operators:** `>`, `<`, `==`, `!=`, `>=`, `<=`, `contains`, `isnull`, `notnull`  
**Allowed aggregates:** `sum`, `mean`, `count`, `min`, `max`

---

## VRP JSON Input Format

The system expects a JSON with at least these top-level keys:

```json
{
  "drivers":      [ { "id": "D001", ... } ],
  "vehicles":     [ { "id": "V001", ... } ],
  "restrictions": [ { "driver_id": "D001", "vehicle_id": "V001", ... } ]
}
```

All other keys are ignored. All column names are preserved **exactly** as they appear.

---

## Demo Mode (no model file)

If `MODEL_PATH` does not exist, the server automatically falls back to the rule-based stub LLM. The app is fully functional for demos and testing.

---

## Generated Files

| File | Description |
|------|-------------|
| `schema.yaml` | Auto-generated schema (column names, dtypes, sample values) — edit to add descriptions |
| `sample_vrp.json` | Sample VRP data for testing |
