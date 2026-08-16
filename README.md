# Desmos Graph Validation Sandbox (OOP Refactor)

A Flask + SSE sandbox where a Qwen model (HauHauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive), served via a local `llama-server` OpenAI-compatible API, is prompted to graph math expressions. Every LaTeX string is validated in a headless browser against the live Desmos API before a final graph JSON payload is emitted.

## Architecture

```
Session      — one conversation. Holds messages, state, iteration counter, disk persistence & crash recovery.
Iteration    — one model call + tool response cycle. Immutable snapshot, self-contained.
ToolHandler  — one handler per tool (ABC + concrete implementations).
```

Each `Iteration` is immutable, so the final `complete` SSE event always pulls from the last iteration — stale data from an earlier turn is structurally impossible.

## Tools (8)

`declare_conditions`, `validate_desmos`, `validate_desmos_multiple`, `graph`, `search_internet`, `search_past_successes`, `desmos_latex_lookup`, `desmos_reference`.

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
cp .env.example .env        # then set your Desmos API key
```

Environment notes:
- `desmos_validator.py` needs a real Chromium. If you use a manual Chrome-for-Testing install, point `PLAYWRIGHT_BROWSERS_PATH` at `~/.cache/ms-playwright`.
- Unset any stale `LD_PRELOAD` that could break browser launch.
- The model server must be running and reachable (default `http://localhost:8003/v1`).
- Set `DESMOS_API_KEY` in `.env` (or the environment). Get a key at https://desmos.com/my-api. The key is loaded at runtime and never hardcoded.

## Run

```bash
gunicorn -w 1 -b 0.0.0.0:7778 --timeout 0 sandbox_app:app
```

The web UI is served at `http://localhost:7778/`.

## Test

```bash
python test_app.py          # unit/state tests (73/73 pass)
python test_new_tools.py    # handler + browser-dependent tests
python test_validator.py    # Desmos LaTeX validator tests
```

## Batch evaluation

```bash
python scripts/run_batch.py scripts/EE-graph-problems.txt
```

Results are written to `scripts/batch_results.json`.

## Project layout

```
sandbox_app.py            # Flask app, Session/Iteration/ToolHandler
graph_brain.py            # OpenAI-compatible client wrapper
desmos_validator.py       # Playwright + live Desmos API LaTeX validation
templates/desmos_harness.html
docs/                     # Desmos API reference (used by desmos_reference tool)
scripts/                  # run_batch.py + graph problem sets
desmos_latex_ref.json     # LaTeX command lookup table
```

## Not shipped

Model binaries (`*.gguf`), logs, runtime state (`session_state.json`, `conversation_history.json`, `*.jsonl`), and `__pycache__` are excluded via `.gitignore`.
