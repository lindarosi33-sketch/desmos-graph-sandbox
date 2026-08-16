#!/usr/bin/env python3
"""Standalone sandbox for testing Desmos /graph pipeline with llama-server.
Port 7778. No auth. No connection to main application.
OOP refactor: Session with crash recovery, Iteration per turn, clean streaming."""
import sys, os, json, logging, re, asyncio, httpx, threading
from logging.handlers import RotatingFileHandler
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_from_directory, Response
from dotenv import load_dotenv
from graph_brain import GraphBrain
from desmos_validator import validate_latex, validate_multiple as validate_multiple_latex, cleanup as cleanup_validator

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('sandbox')
logger.setLevel(logging.INFO)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_sandbox_log = os.path.join(PROJECT_ROOT, 'sandbox.log')
learning_log_file = os.path.join(PROJECT_ROOT, 'learning_log.jsonl')
CONVERSATION_FILE = os.path.join(PROJECT_ROOT, 'conversation_history.json')
SESSION_STATE_FILE = os.path.join(PROJECT_ROOT, 'session_state.json')
NEXT_SESSION_CONTEXT_FILE = os.path.join(PROJECT_ROOT, 'NEXT_SESSION_CONTEXT.md')
_FAILED_ATTEMPTS_FILE = os.path.join(PROJECT_ROOT, 'failed_attempts.jsonl')

_log_dir = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(_log_dir, exist_ok=True)
for _f in [learning_log_file, _sandbox_log]:
    try:
        if os.path.getsize(_f) > 0:
            _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            _name = os.path.basename(_f) + '.' + _ts
            os.rename(_f, os.path.join(_log_dir, _name))
    except Exception:
        pass
    try:
        _base = os.path.basename(_f)
        _backs = sorted([p for p in os.listdir(_log_dir) if p.startswith(_base + '.')])
        for _old in _backs[:-3]:
            os.remove(os.path.join(_log_dir, _old))
    except Exception:
        pass

if not logger.handlers:
    _console = logging.StreamHandler()
    _console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(_console)
    _fileh = RotatingFileHandler(_sandbox_log, maxBytes=10_485_760, backupCount=3)
    _fileh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(_fileh)
    logger.propagate = False

app = Flask(__name__)

# ─── Model (lazy init) ───
_graph_brain = None

def get_model():
    global _graph_brain
    if _graph_brain is None:
        _graph_brain = GraphBrain(model_path="Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf")
        loaded = _graph_brain.load()
        logger.info("Model loaded: %s", loaded)
    return _graph_brain

ABORT_FLAG_FILE = "/tmp/sandbox_abort.flag"

# ─── Constants ───
_DESMOS_GRAPH_FIELDS = [
    "format", "title", "description",
    "points", "bounds", "color",
    "lineStyle", "lineWidth", "lineOpacity",
    "fill", "fillOpacity", "fillColor",
    "pointStyle", "pointSize", "pointOpacity", "lines",
    "sliderBounds", "playing", "regressions", "slider",
    "logModeRegressions", "forceLogModeRegressions", "tableOfResults",
    "xAxisLabel", "yAxisLabel", "xAxisStep", "yAxisStep",
    "showGrid", "squareAxes", "xAxisArrowMode", "yAxisArrowMode",
    "hidden", "secret", "label", "showLabel", "labelSize", "labelOrientation",
    "dragMode", "domain", "parametricDomain", "polarDomain",
    "lockViewport", "polarMode", "projectorMode",
    "xAxisMinorGridlines", "yAxisMinorGridlines",
    "xAxisLabelMode", "yAxisLabelMode",
]

# ====================================================================
# SESSION — manages conversation state with disk persistence & recovery
# ====================================================================

class Session:
    _lock = threading.Lock()

    def __init__(self, user_input: str, fresh: bool = False):
        self.user_input = user_input
        self.session_id = 1000
        self.iteration = 0
        self.validated_ok = False
        self.conditions_declared = False
        self.locked_conditions: set | None = None
        self.validated_expr: str | None = None
        self.consecutive_empty = 0
        self.graph_completed = False
        self.last_input: str | None = None
        self.last_prompt_tokens = 0
        self.messages: list = []
        self.aborted = False

        restored = self._try_restore()
        if restored and not fresh and self.last_input == user_input:
            if self.graph_completed:
                logger.info("Cycling session %d (graph was completed)", self.session_id)
                self._cycle_session(user_input)
            else:
                logger.info("Continuing restored session %d at iteration %d", self.session_id, self.iteration)
                self.consecutive_empty = 0
        else:
            self._start_new_session(user_input)

    def _cycle_session(self, user_input: str):
        self.session_id += 1
        self.validated_ok = False
        self.conditions_declared = False
        self.locked_conditions = None
        self.validated_expr = None
        self.consecutive_empty = 0
        self.iteration = 0
        self.last_input = user_input
        self.graph_completed = False
        self.messages = [self._build_initial_message(user_input)]

    def _start_new_session(self, user_input: str):
        self.session_id += 1
        self.validated_ok = False
        self.conditions_declared = False
        self.locked_conditions = None
        self.validated_expr = None
        self.consecutive_empty = 0
        self.iteration = 0
        self.last_input = user_input
        self.graph_completed = False
        self.messages = [self._build_initial_message(user_input)]

    def _build_initial_message(self, user_input: str) -> dict:
        context = ""
        if os.path.exists(NEXT_SESSION_CONTEXT_FILE):
            try:
                with open(NEXT_SESSION_CONTEXT_FILE) as f:
                    ctx = f.read().strip()
                if ctx:
                    context = "\n\nPrevious session context:\n" + ctx
            except Exception as e:
                logger.warning("Failed to read NEXT_SESSION_CONTEXT.md: %s", e)
        return {
            "role": "user",
            "content": (
                f"Use tools to graph {user_input}\n"
                f"\nDesmos tips:\n"
                f"- \\sgn works fine. Use \\left\\{{ and \\right\\}} for domain restrictions with y =, "
                f"e.g. y = 2.5 \\sgn(\\sin(200\\pi x)) \\left\\{{0 < x < .03\\right\\}}\n"
                f"- Use \\pi (not π symbol)\n"
                f"- No commas inside constraints: {{a < x, x < b}} is invalid; use {{a < x < b}} instead\n"
                f"- In polar mode (polarMode: true), use \\theta as the angle variable\n"
                f"- Use desmos_latex_lookup tool to check LaTeX commands\n"
                f"{context}"
            )
        }

    def _try_restore(self) -> bool:
        try:
            if not os.path.exists(SESSION_STATE_FILE) or not os.path.exists(CONVERSATION_FILE):
                return False
            with open(SESSION_STATE_FILE) as f:
                s = json.load(f)
            self.session_id = s.get("session_id", 1000)
            self.iteration = s.get("iteration", 0)
            self.validated_ok = s.get("validated_ok", False)
            self.conditions_declared = s.get("conditions_declared", False)
            raw_lc = s.get("locked_conditions")
            self.locked_conditions = set(raw_lc) if raw_lc else None
            self.validated_expr = s.get("validated_expr")
            self.consecutive_empty = s.get("consecutive_empty", 0)
            self.graph_completed = s.get("graph_completed", False)
            self.last_input = s.get("last_input")
            self.last_prompt_tokens = s.get("last_prompt_tokens", 0)
            with open(CONVERSATION_FILE) as f:
                self.messages = json.load(f)
            return True
        except Exception as e:
            logger.warning("Session restore failed: %s", e)
            return False

    def persist(self):
        with Session._lock:
            try:
                state = {
                    "session_id": self.session_id,
                    "iteration": self.iteration,
                    "validated_ok": self.validated_ok,
                    "conditions_declared": self.conditions_declared,
                    "locked_conditions": sorted(self.locked_conditions) if self.locked_conditions else None,
                    "validated_expr": self.validated_expr,
                    "consecutive_empty": self.consecutive_empty,
                    "graph_completed": self.graph_completed,
                    "last_input": self.last_input,
                    "last_prompt_tokens": self.last_prompt_tokens,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(SESSION_STATE_FILE, 'w') as f:
                    json.dump(state, f)
                with open(CONVERSATION_FILE, 'w') as f:
                    json.dump(self.messages, f)
            except Exception as e:
                logger.warning("Failed to persist session: %s", e)

    def get_available_tools(self) -> list:
        base = []
        if not self.conditions_declared:
            base.append(declare_conditions_tool)
        if self.conditions_declared:
            base.extend([desmos_reference_tool, validate_desmos_tool, search_past_successes_tool, search_tool, lookup_tool])
        if self.conditions_declared and self.validated_ok:
            base.append(graph_tool)
        return base

    def trim_to_budget(self, max_tokens: int = 32768):
        if len(self.messages) <= 2:
            return
        total = sum(len(json.dumps(m)) // 3 for m in self.messages)
        if total <= max_tokens:
            return
        keep = [self.messages[0]]
        remaining = self.messages[1:]
        budget = max_tokens - (len(json.dumps(keep[0])) // 3)
        kept = 0
        for m in reversed(remaining):
            cost = len(json.dumps(m)) // 3
            if budget - cost >= 0 and kept < 12:
                keep.insert(1, m)
                budget -= cost
                kept += 1
            else:
                break
        if len(keep) < 3:
            keep = [self.messages[0]] + remaining[-4:]
        self.messages[:] = keep

    def append_message(self, msg: dict):
        self.messages.append(msg)

    def append_tool_results(self, tool_calls: list, results: list):
        self.messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        for tid, tc_content in results:
            self.messages.append({"role": "tool", "tool_call_id": tid, "content": tc_content})

# ====================================================================
# ITERATION — one model call + tool response cycle, self-contained
# ====================================================================

@dataclass
class Iteration:
    number: int
    content: str = ""
    think: Optional[str] = None
    model_response: str = ""
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    tool_names: list = field(default_factory=list)
    args_summary: dict = field(default_factory=dict)
    validate_data: Optional[dict] = None
    graph_data: Optional[dict] = None
    validated_ok: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @classmethod
    def from_model_result(cls, number: int, result: dict) -> "Iteration":
        content = result.get("content", "")
        rtext, tcontent = extract_think(content)
        tc_list = result.get("tool_calls") or []
        tc_names = [tc["function"]["name"] for tc in tc_list]
        return cls(
            number=number,
            content=content,
            think=tcontent,
            model_response=rtext,
            tool_calls=tc_list,
            tool_names=tc_names,
            prompt_tokens=result.get("prompt_tokens", 0),
            completion_tokens=result.get("completion_tokens", 0),
        )

# ====================================================================
# TOOL HANDLERS — dispatch map, one handler per tool
# ====================================================================

class ToolHandler(ABC):
    name: str = ""
    @abstractmethod
    def handle(self, args: dict) -> dict: ...

class DeclareConditionsHandler(ToolHandler):
    name = "declare_conditions"
    def handle(self, args: dict) -> dict:
        raw = args.get("conditions", [])
        flat = []
        for c in raw:
            if isinstance(c, list):
                flat.extend(str(x) for x in c)
            else:
                flat.append(str(c))
        conditions = [re.sub(r'\s*([<>=≠≡≈∼]+)\s*', r'\1', c) for c in flat]
        return {"success": True, "locked_conditions": conditions}

class ValidateHandler(ToolHandler):
    name = "validate_desmos"
    def handle(self, args: dict) -> dict:
        expr = args.get("expression", "")
        if isinstance(expr, list):
            return validate_multiple_latex([{"id": str(i), "latex": e} for i, e in enumerate(expr)])
        return validate_latex(expr)

class ValidateMultipleHandler(ToolHandler):
    name = "validate_desmos_multiple"
    def handle(self, args: dict) -> dict:
        expressions = args.get("expressions", [])
        latex_list = []
        for item in expressions:
            expr = item.get("expression", "")
            item_id = item.get("id", str(len(latex_list)))
            latex_list.append({"id": item_id, "latex": expr})
        result = validate_multiple_latex(latex_list)
        results = result.get("results", {})
        any_error = any(r.get("isError") for r in results.values())
        first_error = ""
        for r in results.values():
            if r.get("errorMessage"):
                first_error = r["errorMessage"]
                break
        wrapped = {
            "allValid": result.get("allValid", False),
            "errorCount": result.get("errorCount", 0),
            "results": results,
            "isError": any_error,
            "isGraphable": all(r.get("isGraphable", False) for r in results.values()),
            "errorMessage": first_error,
        }
        return wrapped

class GraphHandler(ToolHandler):
    name = "graph"
    def handle(self, args: dict) -> dict:
        raw = args.get("function_to_graph", "")
        points = args.get("points", [])
        for p in points:
            if "size" in p:
                p.setdefault("pointSize", p.pop("size"))
        result = {"format": args.get("format", ""), "function_to_graph": raw, "latex": raw}
        for f in _DESMOS_GRAPH_FIELDS:
            if f in args:
                result[f] = args[f]
        return {"success": True, "desmos_json": result}

class SearchHandler(ToolHandler):
    name = "search_internet"
    def handle(self, args: dict) -> dict:
        query = args.get("query", "")
        try:
            resp = httpx.get("http://localhost:8888/search",
                params={"q": query, "format": "json", "language": "en", "limit": 5}, timeout=15)
            data = resp.json()
            results = [{"title": r.get("title", ""), "body": (r.get("content") or ""), "url": r.get("url", "")}
                       for r in data.get("results", []) if r.get("title")]
            return {"success": True, "results": results}
        except Exception as e:
            logger.warning("Search error for %r: %s", query, e)
            return {"success": True, "results": [], "error": str(e)}

class PastSuccessesHandler(ToolHandler):
    name = "search_past_successes"
    def handle(self, args: dict) -> dict:
        query = args.get("query", "").strip().lower()
        if not query:
            return {"success": True, "results": [], "error": "Empty query"}
        query_words = query.split()
        results = []
        for g in _successful_graphs:
            score = 0
            text = " ".join(str(g.get(k, "")) for k in ["expression", "latex", "title", "description"]).lower()
            for word in query_words:
                if word in text:
                    score += 1
            if score > 0:
                results.append({
                    "score": score,
                    "expression": g.get("expression", ""),
                    "latex": g.get("latex", ""),
                    "conditions": g.get("conditions", []),
                })
        results.sort(key=lambda r: -r["score"])
        top = results[:5]
        return {"success": True, "results": top}

class LookupHandler(ToolHandler):
    name = "desmos_latex_lookup"
    def handle(self, args: dict) -> dict:
        return _load_latex_ref().get(args.get("command", "").strip(), "")

class DesmosReferenceHandler(ToolHandler):
    name = "desmos_reference"
    def handle(self, args: dict) -> dict:
        query = args.get("query", "").strip().lower()
        if not query:
            return {"success": True, "results": [], "error": "Empty query"}
        docs_dir = os.path.join(PROJECT_ROOT, "docs")
        results = []
        for filename in os.listdir(docs_dir):
            if not filename.startswith("desmos-api"):
                continue
            filepath = os.path.join(docs_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                # Simple keyword search
                words = query.split()
                score = sum(1 for w in words if w.lower() in content.lower())
                if score > 0:
                    # Extract relevant sections
                    sections = []
                    for line in content.split("\n"):
                        if any(w.lower() in line.lower() for w in words):
                            sections.append(line.strip())
                            if len(sections) >= 10:
                                break
                    results.append({
                        "score": score,
                        "file": filename,
                        "sections": sections[:10],
                    })
            except Exception as e:
                logger.warning("Error reading %s: %s", filepath, e)
        results.sort(key=lambda r: -r["score"])
        return {"success": True, "results": results}

_TOOL_HANDLERS: dict[str, ToolHandler] = {
    h.name: h for h in [
        DeclareConditionsHandler(),
        ValidateHandler(),
        GraphHandler(),
        SearchHandler(),
        PastSuccessesHandler(),
        LookupHandler(),
        DesmosReferenceHandler(),
    ]
}
_TOOL_HANDLERS["validate_desmos_multiple"] = ValidateMultipleHandler()

# ====================================================================
# LOGGING HELPERS
# ====================================================================

def _rotate_file(path):
    if not os.path.exists(path):
        return
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    if size > 10_485_760:
        _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        _name = os.path.basename(path) + '.' + _ts
        os.rename(path, os.path.join(_log_dir, _name))
        _base = os.path.basename(path)
        _backs = sorted([p for p in os.listdir(_log_dir) if p.startswith(_base + '.')])
        for _old in _backs[:-3]:
            os.remove(os.path.join(_log_dir, _old))

def log_learning_entry(entry: dict):
    _rotate_file(learning_log_file)
    entry['timestamp'] = datetime.now().isoformat()
    try:
        entry['session_id'] = _session.session_id
    except Exception:
        entry['session_id'] = 0
    try:
        with open(learning_log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception as e:
        logger.warning("Failed to write learning log: %s", e)

def log_failed_attempt(entry: dict):
    """Log a failed attempt (validation failure, graph rejection, etc.) for future reference."""
    _rotate_file(_FAILED_ATTEMPTS_FILE)
    entry['timestamp'] = datetime.now().isoformat()
    try:
        with open(_FAILED_ATTEMPTS_FILE, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception as e:
        logger.warning("Failed to write failed attempt: %s", e)

# ====================================================================
# ABORT MECHANISM
# ====================================================================

def _set_abort_flag():
    open(ABORT_FLAG_FILE, "w").close()

def _clear_abort_flag():
    try:
        os.remove(ABORT_FLAG_FILE)
    except FileNotFoundError:
        pass

def _is_abort_set():
    return os.path.exists(ABORT_FLAG_FILE)

# ====================================================================
# HELPERS
# ====================================================================

def extract_think(response: str):
    close_tag = "</think>"
    end_idx = response.lower().find(close_tag)
    if end_idx != -1:
        expression = response[end_idx + len(close_tag):].strip()
        start_idx = response.lower().find("<think>")
        think_content = response[start_idx + len("<think>"):end_idx].strip() if start_idx != -1 else None
    else:
        expression = response.strip()
        think_content = None
    return expression, think_content

def _normalize_pi_inf(text: str) -> str:
    text = text.replace('\\pi', 'π').replace('Math.PI', 'π').replace('math.pi', 'π')
    text = re.sub(r'\bpi\b', 'π', text)
    text = re.sub(r'(?<=\d)pi', 'π', text)
    text = re.sub(r'\b3\.14\d*', 'π', text)
    text = text.replace('\\infty', '∞').replace('+inf', '∞').replace('-inf', '-∞')
    text = re.sub(r'\binf\b', '∞', text)
    return text

def _extract_conditions(text: str) -> set:
    if not text:
        return set()
    text = text.replace('\\left', ' _L_ ').replace('\\right', ' _R_ ')
    for old, new in sorted(
        [('\\geqslant', '>='), ('\\leqslant', '<='),
         ('\\geq', '>='), ('\\leq', '<='),
         ('\\ge', '>='), ('\\le', '<='),
         ('\\gt', '>'), ('\\lt', '<')],
        key=lambda x: -len(x[0])
    ):
        text = text.replace(old, new)
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', text)
    text = _normalize_pi_inf(text)
    text = text.replace('≠', '!=')
    text = text.replace(' _L_ ', '\\left').replace(' _R_ ', '\\right')
    _num = r'(?:-?(?:\d+(?:\.\d+)?|\.\d+|π|∞)(?:\s*\*?\s*π)?(?:/(?:\d+|π|∞))?)'
    patterns = re.findall(r'[a-zA-Z]\s*(?:<|>|<=|>=|!=)\s*' + _num, text)
    patterns += re.findall(_num + r'\s*(?:<|>|<=|>=|!=)\s*[a-zA-Z]', text)
    ranges = re.findall(_num + r'\s*(?:<|>|<=|>=|!=)\s*[a-zA-Z]\s*(?:<|>|<=|>=|!=)\s*' + _num, text)
    normalized = set()
    for c in patterns + ranges:
        c = re.sub(r'\s*([<>=]+)\s*', r'\1', c)
        normalized.add(c)
    return normalized

def _norm_conds(conds):
    _OP_REV = {'>=':'<=', '<=':'>=', '>':'<', '<':'>', '!=':'!='}
    result = set()
    for c in conds:
        c = _normalize_pi_inf(c)
        c = c.replace('\u2264', '<=').replace('\u2265', '>=')
        c = c.replace('\u2260', '!=').replace('\u2248', '~').replace('\u223c', '~').replace('\u2261', '===')
        c = re.sub(r'\s*([<>=!~+\-]+)\s*', r'\1', c)
        c = re.sub(r'(?<=\d)\s*\*\s*(?=[π\u03c0])', '', c)
        m = re.match(
            r'([a-zA-Z\u03c0\u221e])([<>=!~]+)(-?(?:[\d.]+|[\u03c0\u221e])(?:\s*\*?\s*[\u03c0])?(?:/(?:\d+|[\u03c0\u221e]))?)\s*(?:and|&&)\s*\1([<>=!~]+)(-?(?:[\d.]+|[\u03c0\u221e])(?:\s*\*?\s*[\u03c0])?(?:/(?:\d+|[\u03c0\u221e]))?)',
            c
        )
        if m:
            op = _OP_REV.get(m.group(2), m.group(2))
            c = m.group(3) + op + m.group(1) + m.group(4) + m.group(5)
        range_m = re.match(r'^([\d.π∞/+-]+)\s*([<>=!]+)\s*([a-zA-Z\u03c0\u221e])\s*([<>=!]+)\s*([\d.π∞/+-]+)$', c)
        if range_m:
            for part in (range_m.group(1)+range_m.group(2)+range_m.group(3),
                         range_m.group(3)+range_m.group(4)+range_m.group(5)):
                part = re.sub(r'^[a-zA-Z\u03c0\u221e]+(?=[<>=!])', '_', part)
                result.add(part)
            continue
        c = re.sub(r'^[a-zA-Z\u03c0\u221e]+(?=[<>=!])', '_', c)
        result.add(c)
    return result

def _check_conditions(expr: str, locked: set | None) -> tuple[bool, set]:
    if not locked or '\\{' not in expr:
        return True, set()
    domain_locked = {c for c in locked if '(' not in c}
    if not domain_locked:
        return True, set()
    expr_conds = _extract_conditions(expr)
    if not expr_conds:
        return False, domain_locked
    norm_locked = _norm_conds(domain_locked)
    norm_expr = _norm_conds(expr_conds)
    missing = norm_locked - norm_expr
    return not missing, missing

def _infer_boundary_points(conditions: set) -> set:
    bounds = {}
    for c in conditions:
        m = re.match(r'[a-zA-Z]\s*(<|>|<=|>=)\s*(-?\d+(?:\.\d+)?)', c)
        if m:
            op, val = m.group(1), m.group(2)
            if val not in bounds:
                bounds[val] = {"lt": False, "lte": False, "gt": False, "gte": False}
            if op == "<":
                bounds[val]["lt"] = True
            elif op == "<=":
                bounds[val]["lte"] = True
            elif op == ">":
                bounds[val]["gt"] = True
            elif op == ">=":
                bounds[val]["gte"] = True
    result = set()
    for val, ops in bounds.items():
        if (ops["lt"] or ops["lte"]) and (ops["gt"] or ops["gte"]):
            result.add(f"x={val}")
    return result

# ─── Desmos LaTeX reference ───
_desmos_latex_ref = None
def _load_latex_ref():
    global _desmos_latex_ref
    if _desmos_latex_ref is None:
        ref_path = os.path.join(PROJECT_ROOT, 'desmos_latex_ref.json')
        try:
            with open(ref_path) as f:
                _desmos_latex_ref = json.load(f).get("commands", {})
        except Exception as e:
            logger.warning("Failed to load lookup table: %s", e)
            _desmos_latex_ref = {}
    return _desmos_latex_ref

def _strip_tc(tc: dict) -> dict:
    name = tc["function"]["name"]
    args = json.loads(tc["function"]["arguments"])
    summary = {}
    if name == "graph":
        summary["function_to_graph"] = args.get("function_to_graph", "")
    elif name == "validate_desmos":
        summary["expression"] = args.get("expression", "")
    elif name == "search_internet":
        summary["query"] = args.get("query", "")
    elif name == "search_past_successes":
        summary["query"] = args.get("query", "")
    elif name == "declare_conditions":
        summary["conditions"] = args.get("conditions", "")
    elif name == "desmos_latex_lookup":
        summary["command"] = args.get("command", "")
    return {
        "id": tc["id"],
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(summary)}
    }

# ====================================================================
# MEMORY STORE — successful graphs
# ====================================================================

_successful_graphs = []
_MEMORY_STORE = os.path.join(PROJECT_ROOT, 'successful_graphs.jsonl')

def _load_memory_store():
    global _successful_graphs
    _successful_graphs = []
    if not os.path.exists(_MEMORY_STORE):
        return
    try:
        with open(_MEMORY_STORE) as f:
            for line in f:
                line = line.strip()
                if line:
                    _successful_graphs.append(json.loads(line))
        logger.info("Loaded %d successful graph memories", len(_successful_graphs))
    except Exception as e:
        logger.warning("Failed to load memory store: %s", e)

def _save_graph_memory(expression: str, latex: str, conditions: list, session_id: int, title: str = "", description: str = "", validated_latex: str = None):
    if any(g.get("expression") == expression and g.get("latex") == latex for g in _successful_graphs):
        return
    entry = {
        "expression": expression, "latex": latex, "validated_latex": validated_latex,
        "conditions": conditions, "title": title, "description": description,
        "timestamp": datetime.now().isoformat(), "session_id": session_id,
    }
    _successful_graphs.append(entry)
    try:
        with open(_MEMORY_STORE, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception as e:
        logger.warning("Failed to save graph memory: %s", e)

_load_memory_store()

# ====================================================================
# TOOL DEFINITIONS
# ====================================================================

graph_tool = {
    "type": "function",
    "function": {
        "name": "graph",
        "description": "Graph a mathematical expression using Desmos LaTeX. IMPORTANT: function_to_graph MUST be a complete expression with y= (e.g. y=x^2, NOT x^2). You can include points array and set lines=true to connect points with segments.",
        "parameters": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "The output format - must be Desmos-compatible LaTeX."
                },
                "function_to_graph": {
                    "type": "string",
                    "description": "A SINGLE complete LaTeX expression with y= prefix. Example: y=x^2, NOT x^2. For piecewise: y=\\{x<0: -x, x>=0: x\\}. Do NOT use arrays."
                },
                "title": {
                    "type": "string",
                    "description": "A descriptive title for the graph."
                },
                "description": {
                    "type": "string",
                    "description": "A brief explanation of the graph."
                },
                "points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                     "x": {"type": "string", "description": "Numeric x-coordinate as a plain string, e.g. '0' or '\\pi'. MUST be a number. Do NOT include operators like >=, <, etc."},
                              "y": {"type": "string", "description": "Numeric y-coordinate as a plain string, e.g. '1' or '\\sqrt{2}'. Put annotations in the label field."},
                            "type": {
                                "type": "string",
                                "enum": ["open", "closed", "point"],
                                "description": "open=hole/discontinuity, closed=included endpoint, point=regular point"
                            },
                            "label": {"type": "string", "description": "Optional text label"},
                            "showLabel": {"type": "boolean", "description": "Whether to show the label on the graph."},
                            "color": {"type": "string", "description": "Hex color for this point, e.g. '#c74440' or named color."},
                            "pointStyle": {
                                "type": "string", "enum": ["point", "open", "closed"],
                                "description": "Visual style for the point marker."
                            },
                            "dragMode": {
                                "type": "string", "enum": ["X", "Y", "XY", "NONE"],
                                "description": "Allow dragging this point."
                            },
                            "secret": {"type": "boolean", "description": "Hide this point from the expressions list."},
                            "hidden": {"type": "boolean", "description": "Hide this point from the graph."},
                            "labelSize": {
                                "type": "string", "enum": ["small", "medium", "large"],
                                "description": "Size of the point label."
                            },
                            "labelOrientation": {
                                "type": "string", "enum": ["default", "left", "right", "up", "down"],
                                "description": "Position of the point label."
                            }
                        },
                        "required": ["x", "y", "type"]
                    },
                    "description": "Points of interest. Example: [{\"x\":0,\"y\":0,\"type\":\"closed\",\"showLabel\":true,\"label\":\"(0,0)\"},{\"x\":1,\"y\":1,\"type\":\"open\"}] — closed=endpoint, open=hole. Set lines=true to connect points with line segments."
                },
                "bounds": {
                    "type": "object",
                    "properties": {
                        "left": {"type": "number"}, "right": {"type": "number"},
                        "bottom": {"type": "number"}, "top": {"type": "number"}
                    },
                    "description": "Graph window bounds. Always set all four values (left, right, bottom, top)."
                },
                "color": {"type": "string", "description": "Line color, e.g. '#2d70b3' or 'blue'."},
                "lineStyle": {"type": "string", "enum": ["solid", "dashed", "dotted"], "description": "Line style."},
                "lineWidth": {"type": "number", "description": "Line thickness (default 2)."},
                "lineOpacity": {"type": "number", "description": "Line opacity from 0 to 1."},
                "fill": {"type": "boolean", "description": "Fill under the curve."},
                "fillOpacity": {"type": "number", "description": "Fill opacity from 0 to 1."},
                "fillColor": {"type": "string", "description": "Fill color hex."},
                "pointStyle": {"type": "string", "enum": ["point", "open", "closed"], "description": "Default point style."},
                "pointSize": {"type": "number", "description": "Point marker size (default 9)."},
                "pointOpacity": {"type": "number", "description": "Point marker opacity from 0 to 1."},
                "lines": {"type": "boolean", "description": "When true, connects the points array with line segments. Set to true when you want points joined by lines."},
                "sliderBounds": {"type": "object", "properties": {"min": {"type": "number"}, "max": {"type": "number"}, "step": {"type": "number"}}, "description": "Slider bounds."},
                "playing": {"type": "boolean", "description": "Animate slider."},
                "regressions": {"type": "boolean", "description": "Allow regression models."},
                "logModeRegressions": {"type": "boolean", "description": "Log mode for regressions."},
                "forceLogModeRegressions": {"type": "boolean", "description": "Force log mode for all regressions."},
                "tableOfResults": {"type": "boolean", "description": "Show table of results for distributions."},
                "xAxisLabel": {"type": "string", "description": "Label for the x-axis."},
                "yAxisLabel": {"type": "string", "description": "Label for the y-axis."},
                "xAxisStep": {"type": "string", "description": "X-axis tick step. For trig functions use \\pi."},
                "yAxisStep": {"type": "string", "description": "Y-axis tick step."},
                "showGrid": {"type": "boolean", "description": "Show grid lines (default true)."},
                "squareAxes": {"type": "boolean", "description": "Lock aspect ratio to 1:1."},
                "xAxisArrowMode": {"type": "string", "enum": ["none", "positive", "both"], "description": "Arrow heads on x-axis."},
                "yAxisArrowMode": {"type": "string", "enum": ["none", "positive", "both"], "description": "Arrow heads on y-axis."},
                "hidden": {"type": "boolean", "description": "Hide expression from graph view."},
                "secret": {"type": "boolean", "description": "Hide expression from expressions list."},
                "label": {"type": "string", "description": "Label for the expression itself."},
                "showLabel": {"type": "boolean", "description": "Show expression label."},
                "labelSize": {"type": "string", "enum": ["small", "medium", "large"], "description": "Label size."},
                "labelOrientation": {"type": "string", "enum": ["default", "left", "right", "up", "down"], "description": "Label position."},
                "dragMode": {"type": "string", "enum": ["X", "Y", "XY", "NONE"], "description": "Drag mode."},
                "domain": {"type": "object", "properties": {"min": {"type": "number"}, "max": {"type": "number"}}, "description": "Per-expression x-domain restriction."},
                "parametricDomain": {"type": "object", "properties": {"min": {"type": "number"}, "max": {"type": "number"}}, "description": "Domain for parametric parameter t."},
                "polarDomain": {"type": "object", "properties": {"min": {"type": "number"}, "max": {"type": "number"}}, "description": "Domain for polar theta."},
                "slider": {"type": "object", "description": "Slider configuration."},
                "lockViewport": {"type": "boolean", "description": "Lock viewport."},
                "polarMode": {"type": "boolean", "description": "Enable polar mode."},
                "projectorMode": {"type": "boolean", "description": "Projector mode."},
                "xAxisMinorGridlines": {"type": "boolean", "description": "Minor gridlines on x-axis."},
                "yAxisMinorGridlines": {"type": "boolean", "description": "Minor gridlines on y-axis."},
                "xAxisLabelMode": {"type": "string", "enum": ["auto", "manual"], "description": "X-axis label mode."},
                "yAxisLabelMode": {"type": "string", "enum": ["auto", "manual"], "description": "Y-axis label mode."}
            },
            "required": ["format", "function_to_graph", "title", "description"]
        }
    }
}

validate_desmos_tool = {
    "type": "function",
    "function": {
        "name": "validate_desmos",
        "description": "Validate a LaTeX expression against Desmos to check if it will graph correctly.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The LaTeX expression to validate"
                }
            },
            "required": ["expression"]
        }
    }
}

validate_multiple_tool = {
    "type": "function",
    "function": {
        "name": "validate_desmos_multiple",
        "description": "Validate multiple LaTeX expressions against Desmos simultaneously. Useful for validating a circle equation and points in one shot.",
        "parameters": {
            "type": "object",
            "properties": {
                "expressions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Unique identifier for this expression"},
                            "expression": {"type": "string", "description": "The LaTeX expression to validate"}
                        },
                        "required": ["id", "expression"]
                    },
                    "description": "Array of expressions to validate"
                }
            },
            "required": ["expressions"]
        }
    }
}

search_tool = {
    "type": "function",
    "function": {
        "name": "search_internet",
        "description": "Search the internet.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        }
    }
}

lookup_tool = {
    "type": "function",
    "function": {
        "name": "desmos_latex_lookup",
        "description": "Look up a LaTeX command by symbol or name.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The symbol or command name"}
            },
            "required": ["command"]
        }
    }
}

search_past_successes_tool = {
    "type": "function",
    "function": {
        "name": "search_past_successes",
        "description": "Look up past successful Desmos graphs for LaTeX approaches.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, e.g. 'absolute value' or 'piecewise'"}
            },
            "required": ["query"]
        }
    }
}

desmos_reference_tool = {
    "type": "function",
    "function": {
        "name": "desmos_reference",
        "description": "Search the Desmos API documentation for syntax reference. Use this to look up how to express things like parametric equations, polar coordinates, domain restrictions, styling options, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, e.g. 'parametric equations', 'domain restriction', 'point style', 'polar coordinates'"}
            },
            "required": ["query"]
        }
    }
}

declare_conditions_tool = {
    "type": "function",
    "function": {
        "name": "declare_conditions",
        "description": "Domain splits for piecewise functions, else [] if the function has a single expression over all reals.",
        "parameters": {
            "type": "object",
            "properties": {
                "conditions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of conditions, or [] if the function has no domain splits."
                }
            },
            "required": ["conditions"]
        }
    }
}

# ====================================================================
# STREAMING VALIDATION + COMPLETE CORE
# ====================================================================

_session = Session(user_input="", fresh=True)
_session_lock = threading.Lock()

def _validate_complete_stream(data, user_input):
    global _session
    _clear_abort_flag()

    if data.get("fresh", True):
        open(learning_log_file, 'w').close()

    with _session_lock:
        _session = Session(user_input, fresh=data.get("fresh", True))

    yield f"data: session_id:{_session.session_id}\n\n"

    error_msg = None
    final_expr = None
    graph_data = None
    validate_data = None
    last_iteration: Optional[Iteration] = None
    response_text = None
    think_content = None

    while True:
        if _is_abort_set():
            _clear_abort_flag()
            error_msg = "Aborted by user."
            graph_data = None
            break

        _session.iteration += 1
        if _session.last_prompt_tokens > 24576:
            _session.trim_to_budget()
            _session.last_prompt_tokens = 0

        validate_data = None
        iteration: Optional[Iteration] = None

        # ─── Call model ───
        try:
            result = get_model().chat(_session.messages, tools=_session.get_available_tools())
            iteration = Iteration.from_model_result(_session.iteration, result)
            prompt_tokens = result.get("prompt_tokens")
            if prompt_tokens:
                _session.last_prompt_tokens = prompt_tokens
        except Exception as e:
            err_str = str(e).lower()
            if "timeout" in err_str or "timed out" in err_str:
                logger.error("Model request timed out at iteration %s: %s", _session.iteration, e)
            elif "connection" in err_str or "refused" in err_str or "resolve" in err_str:
                logger.error("Model server unreachable at iteration %s: %s", _session.iteration, e)
            else:
                logger.error("Model request failed at iteration %s: %s", _session.iteration, e)
            log_learning_entry({'endpoint': 'model_error', 'iteration': _session.iteration, 'user_input': user_input, 'error': str(e)})
            yield f"data: log_update\n\n"
            is_conn = "connection" in err_str or "refused" in err_str or "resolve" in err_str
            if not is_conn:
                _session.consecutive_empty += 1
            _session.append_message({"role": "user", "content": "The previous request failed. Try again."})
            _session.persist()
            continue

        if not iteration:
            continue

        if response_text is None:
            response_text = iteration.model_response
        think_content = iteration.think
        tool_calls = iteration.tool_calls

        # Build args summary
        args_summary = {}
        for tc in tool_calls:
            tc_name = tc["function"]["name"]
            tc_args = json.loads(tc["function"]["arguments"])
            if tc_name == "validate_desmos": args_summary["validate_expr"] = tc_args.get("expression", "")
            elif tc_name == "validate_desmos_multiple": args_summary["validate_exprs"] = tc_args.get("expressions", [])
            elif tc_name == "search_internet": args_summary["search_query"] = tc_args.get("query", "")
            elif tc_name == "search_past_successes": args_summary["past_query"] = tc_args.get("query", "")
            elif tc_name == "declare_conditions": args_summary["conditions"] = tc_args.get("conditions", [])
            elif tc_name == "desmos_latex_lookup": args_summary["lookup"] = tc_args.get("command", "")
            elif tc_name == "desmos_reference": args_summary["ref_query"] = tc_args.get("query", "")
            elif tc_name == "graph": args_summary["graph_expr"] = tc_args.get("function_to_graph", "")

        logger.info("━━━ ITERATION #%s ━━━", _session.iteration)
        logger.info("USER_INPUT: %s", user_input)
        logger.info("TOOLS: %s", iteration.tool_names)
        logger.info("VALIDATED_OK: %s", _session.validated_ok)
        logger.info("EMPTY_COUNT: %s", _session.consecutive_empty)
        if _session.locked_conditions:
            logger.info("LOCKED_CONDITIONS: %s", sorted(_session.locked_conditions))
        if iteration.think:
            logger.info("THINK:\n%s", iteration.think)
        if iteration.model_response:
            logger.info("RESPONSE:\n%s", iteration.model_response)
        logger.info("────────────────────────────────")

        # ─── Handle empty turns ───
        if not tool_calls:
            _session.consecutive_empty += 1
            log_learning_entry({
                'endpoint': 'graph_iteration', 'iteration': _session.iteration,
                'user_input': user_input, 'raw_content': iteration.content,
                'think': iteration.think, 'model_response': iteration.model_response,
                'tool_called': False, 'tool_names': [], 'tool_args': args_summary,
                'validated_ok': _session.validated_ok, 'consecutive_empty': _session.consecutive_empty,
                'locked_conditions': sorted(_session.locked_conditions) if _session.locked_conditions else None,
                'prompt_tokens': iteration.prompt_tokens, 'completion_tokens': iteration.completion_tokens,
            })
            yield f"data: log_update\n\n"
            if _session.consecutive_empty >= 3:
                logger.warning("MAX_EMPTY: iteration=%s input=%s", _session.iteration, user_input)
                _session.consecutive_empty = 0
                _session.append_message({"role": "user", "content": "You have not made a tool call for 3 consecutive turns. You must use tool calls to complete the task."})
                _session.persist()
                continue
            _session.persist()
            continue

        _session.consecutive_empty = 0
        logger.info("EMPTY_COUNT: %s", _session.consecutive_empty)

        # ─── Sort tool calls: declare/validate first, graph last ───
        tool_calls.sort(key=lambda tc: (
            2 if tc["function"]["name"] == "graph" else
            0 if tc["function"]["name"] in ("declare_conditions", "validate_desmos") else 1
        ))

        tc_strips = []
        tc_results = []
        graph_data = None

        # ─── Execute tool calls ───
        for tc in tool_calls:
            tc_name = tc["function"]["name"]
            tc_args = json.loads(tc["function"]["arguments"])
            handler = _TOOL_HANDLERS.get(tc_name)
            tc_strips.append(_strip_tc(tc))

            if tc_name == "validate_desmos" or tc_name == "validate_desmos_multiple":
                vresult = handler.handle(tc_args)
                logger.info("VALIDATE ARGS: %s", json.dumps(tc_args))
                logger.info("VALIDATE RESULT: isError=%s isGraphable=%s errorMessage=%s",
                    vresult.get("isError"), vresult.get("isGraphable"), vresult.get("errorMessage"))
                validate_data = vresult
                if not vresult.get("isError", True) and vresult.get("isGraphable") is not False:
                    _expr = tc_args.get("expression", "")
                    _session.validated_ok = True
                    _session.validated_expr = _expr
                    tc_results.append((tc["id"], json.dumps(vresult)))
                    log_learning_entry({'endpoint': 'validate_success', 'iteration': _session.iteration, 'args': tc_args, 'result': vresult, 'response_text': json.dumps(vresult)})
                    yield f"data: log_update\n\n"
                    conds = sorted(_session.locked_conditions) if _session.locked_conditions else []
                    _save_graph_memory(user_input, _expr, conds, _session.session_id)
                else:
                    logger.info("VALIDATE FAILED: expr=%s error=%s", tc_args.get("expression", ""), vresult.get("errorMessage", ""))
                    log_failed_attempt({'endpoint': 'validate_failed', 'iteration': _session.iteration, 'expression': tc_args.get("expression", ""), 'error': vresult.get("errorMessage", "")})
                    log_learning_entry({'endpoint': 'validate_failed', 'iteration': _session.iteration, 'expression': tc_args.get("expression", ""), 'error': vresult, 'response_text': json.dumps(vresult)})
                    yield f"data: log_update\n\n"
                    tc_results.append((tc["id"], json.dumps(vresult)))
                    break

            elif tc_name == "declare_conditions":
                dc_result = handler.handle(tc_args)
                try:
                    conditions = [str(c) for c in dc_result["locked_conditions"]]
                    conditions = [re.sub(r'\s*([<>=≠≡≈∼]+)\s*', r'\1', c) for c in conditions]
                    _session.locked_conditions = set(conditions)
                    _session.conditions_declared = True
                    logger.info("LOCKED CONDITIONS: %s", sorted(_session.locked_conditions))
                    conds_response = f"Conditions locked: {sorted(_session.locked_conditions)}."
                    tc_results.append((tc["id"], conds_response))
                    log_learning_entry({'endpoint': 'conditions_declared', 'iteration': _session.iteration, 'conditions': sorted(_session.locked_conditions), 'response_text': conds_response})
                    yield f"data: log_update\n\n"
                except Exception as e:
                    logger.warning("Failed to process conditions: %s", e)
                    tc_results.append((tc["id"], f"Failed to process conditions: {e}"))

            elif tc_name == "search_internet":
                sresult = handler.handle(tc_args)
                logger.info("SEARCH RESULT: %d results", len(sresult.get("results", [])))
                search_response = json.dumps({"search_results": sresult["results"]})
                tc_results.append((tc["id"], search_response))
                log_learning_entry({'endpoint': 'search_call', 'iteration': _session.iteration, 'args': tc_args, 'results': sresult.get("results", []), 'response_text': search_response})
                yield f"data: log_update\n\n"

            elif tc_name == "search_past_successes":
                mresult = handler.handle(tc_args)
                mem_response = json.dumps(mresult)
                tc_results.append((tc["id"], mem_response))
                log_learning_entry({'endpoint': 'search_past_successes', 'iteration': _session.iteration, 'args': tc_args, 'results': len(mresult.get("results", [])), 'response_text': mem_response})
                yield f"data: log_update\n\n"

            elif tc_name == "desmos_latex_lookup":
                lresult = handler.handle(tc_args)
                lookup_response = json.dumps(lresult)
                tc_results.append((tc["id"], lookup_response))
                log_learning_entry({'endpoint': 'lookup_call', 'iteration': _session.iteration, 'args': tc_args, 'result': lresult, 'response_text': lookup_response})
                yield f"data: log_update\n\n"

            elif tc_name == "desmos_reference":
                rresult = handler.handle(tc_args)
                ref_response = json.dumps(rresult)
                tc_results.append((tc["id"], ref_response))
                log_learning_entry({'endpoint': 'desmos_reference_call', 'iteration': _session.iteration, 'args': tc_args, 'results': rresult.get("results", []), 'response_text': ref_response})
                yield f"data: log_update\n\n"

            elif tc_name == "graph":
                graph_raw = tc_args.get("function_to_graph", "")
                if isinstance(graph_raw, list):
                    err_msg = "function_to_graph must be a single LaTeX string, not an array."
                    logger.warning("GRAPH INVALID: function_to_graph is a list")
                    tc_results.append((tc["id"], err_msg + " Received: " + json.dumps(graph_raw)))
                    graph_data = None
                    continue
                logger.info("GRAPH ARGS: %s", json.dumps(tc_args))
                graph_expr = tc_args.get("function_to_graph", "")
                gresult = handler.handle(tc_args)
                graph_data = gresult
                # Check boundary points
                lc = _session.locked_conditions
                if lc:
                    boundary_points = _infer_boundary_points(lc)
                    if boundary_points:
                        graph_points = tc_args.get("points") or []
                        missing_coords = set()
                        for cc in boundary_points:
                            expected_x = cc.split("=", 1)[1].strip()
                            has_point = any(str(p.get("x", "")).strip() == expected_x for p in graph_points)
                            if not has_point:
                                missing_coords.add(cc)
                        if missing_coords:
                            err_msg = f"Please add points at boundaries {sorted(missing_coords)}."
                            logger.info("GRAPH MISSING POINTS: %s", err_msg)
                            tc_results.append((tc["id"], err_msg))
                            graph_data = None
                            continue
                # Validate ALL LaTeX-bearing fields
                _latex_fields_to_validate = {
                    'function_to_graph': graph_expr,
                }
                points = tc_args.get("points", [])
                for pi, pt in enumerate(points):
                    pt_x = pt.get('x', '')
                    pt_y = pt.get('y', '')
                    if pt_x and pt_y and isinstance(pt_x, str) and isinstance(pt_y, str):
                        _latex_fields_to_validate[f'points[{pi}]'] = f'({pt_x}, {pt_y})'
                    pt_label = pt.get('label', '')
                    if pt_label and isinstance(pt_label, str):
                        _latex_fields_to_validate[f'points[{pi}].label'] = pt_label
                for fld in ('label', 'graphDescription', 'xAxisLabel', 'yAxisLabel', 'xAxisStep', 'yAxisStep'):
                    v = tc_args.get(fld, '')
                    if v and isinstance(v, str):
                        _latex_fields_to_validate[fld] = v
                for dom in ('domain', 'parametricDomain', 'polarDomain'):
                    d = tc_args.get(dom, {})
                    if isinstance(d, dict):
                        for b in ('min', 'max'):
                            v = d.get(b, '')
                            if v and isinstance(v, str):
                                _latex_fields_to_validate[f'{dom}.{b}'] = v
                sb = tc_args.get("sliderBounds", {})
                if isinstance(sb, dict):
                    for k, v in sb.items():
                        if isinstance(v, str) and v:
                            _latex_fields_to_validate[f'sliderBounds.{k}'] = v
                for name, latex in _latex_fields_to_validate.items():
                    r = validate_latex(latex)
                    if r.get("isError") or r.get("isGraphable") is False:
                        logger.info("FIELD VALIDATION FAILED: %s=%s error=%s", name, latex, r.get("errorMessage", ""))
                        log_learning_entry({'endpoint': 'graph_field_failed', 'iteration': _session.iteration, 'field': name, 'latex': latex, 'error': r.get("errorMessage", ""), 'response_text': json.dumps(r)})
                        yield f"data: log_update\n\n"
                        tc_results.append((tc["id"], json.dumps(r)))
                        _session.validated_ok = False
                        graph_data = None
                        break
                if graph_data is None:
                    continue
                log_learning_entry({'endpoint': 'graph_attempt', 'iteration': _session.iteration, 'expression': graph_expr, 'args': tc_args, 'valid': r, 'phase': 'graph', 'response_text': json.dumps(gresult)})
                yield f"data: log_update\n\n"
                logger.info("GRAPH SUCCESS: expr=%s", graph_expr)
                final_expr = graph_expr
                validated_expr = _session.validated_expr
                if validated_expr and validated_expr.strip() != graph_expr.strip():
                    _session.validated_ok = False
                    tc_results.append((tc["id"], f"Expression '{graph_expr}' does not match the validated expression: '{validated_expr}'."))
                else:
                    _save_graph_memory(user_input, graph_expr, sorted(lc) if lc else [], _session.session_id,
                                      tc_args.get("title", ""), tc_args.get("description", ""), validated_expr)
                    tc_results.append((tc["id"], "Task complete. Graph successful."))

            else:
                logger.warning("Unknown tool call: %s — skipping", tc_name)
                tc_results.append((tc["id"], f"Unknown tool '{tc_name}'. Available tools: validate_desmos, declare_conditions, search_internet, search_past_successes, desmos_latex_lookup, desmos_reference, graph."))

        # ─── Record tool calls in message history ───
        if tc_strips:
            _session.append_tool_results(tc_strips, tc_results)

        # ─── Log iteration ───
        log_learning_entry({
            'endpoint': 'graph_iteration', 'iteration': _session.iteration,
            'user_input': user_input, 'raw_content': iteration.content,
            'think': iteration.think, 'model_response': iteration.model_response,
            'tool_called': True, 'tool_names': iteration.tool_names, 'tool_args': args_summary,
            'validated_ok': _session.validated_ok, 'consecutive_empty': _session.consecutive_empty,
            'think_conditions': None,
            'locked_conditions': sorted(_session.locked_conditions) if _session.locked_conditions else None,
            'prompt_tokens': iteration.prompt_tokens, 'completion_tokens': iteration.completion_tokens,
        })
        yield f"data: log_update\n\n"

        # ─── Persist after every iteration (crash recovery) ───
        _session.persist()

        if graph_data:
            _session.graph_completed = True
            _session.persist()
            break

    # ─── Build response ───
    response = {
        "success": graph_data is not None,
        "session_id": _session.session_id,
        "response": response_text,
        "think": think_content,
        "validated_latex": final_expr,
        "graph_data": graph_data,
        "validate_data": validate_data,
        "iterations": _session.iteration,
        "error": error_msg,
    }
    yield f"data: complete:{json.dumps(response)}\n\n"

# ====================================================================
# ROUTES
# ====================================================================

@app.route('/api/validate_complete', methods=['POST'])
def validate_complete():
    _clear_abort_flag()
    data = request.json
    user_input = data.get('input', '').strip()
    if not user_input:
        return jsonify({'success': False, 'error': 'Missing input'}), 400
    if request.args.get('stream', 'true') == 'false':
        gen = _validate_complete_stream(data, user_input)
        for item in gen:
            if item.startswith('data: complete:'):
                payload = json.loads(item[len('data: complete:'):])
                return jsonify(payload)
        return jsonify({'success': False, 'error': 'Stream ended without complete event'}), 500
    return Response(_validate_complete_stream(data, user_input), mimetype='text/event-stream')

@app.route('/api/validate', methods=['POST'])
def validate_endpoint():
    try:
        data = request.json
        expr = data.get('expression', '').strip()
        if not expr:
            return jsonify({'success': False, 'isError': True, 'errorMessage': 'Missing expression'}), 400
        result = validate_latex(expr)
        return jsonify({'success': True, 'expression': expr, **result})
    except Exception as e:
        logger.error("Validate error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/abort', methods=['POST'])
def abort_endpoint():
    _set_abort_flag()
    logger.info("User requested abort")
    return jsonify({"success": True, "aborted": True})

@app.route('/')
@app.route('/<path:filename>')
def serve_test(filename='desmos_harness.html'):
    return send_from_directory(os.path.join(PROJECT_ROOT, 'templates'), filename)

@app.route('/api/config')
def config_endpoint():
    return jsonify({"desmosApiKey": os.getenv("DESMOS_API_KEY", "")})

@app.route('/api/learning_log')
def view_learning_log():
    entries = []
    session_filter = request.args.get("session", type=int)
    try:
        with open(learning_log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    if session_filter is not None and e.get('session_id') != session_filter:
                        continue
                    entries.append(e)
    except FileNotFoundError:
        pass
    return jsonify({'entries': entries[-50:]})

if __name__ == '__main__':
    port = int(os.environ.get('SANDBOX_PORT', 7778))
    logger.info("Sandbox starting on port %s", port)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    cleanup_validator()
