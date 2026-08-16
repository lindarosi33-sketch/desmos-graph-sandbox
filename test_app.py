#!/usr/bin/env python3
"""Integration tests for sandbox_app OOP refactor.
Run: cd /media/data/sandbox_oop && /home/rosco/miniconda3/envs/deepseek-14b/bin/python test_app.py
"""
import os, sys, json, tempfile, shutil, threading, time
from unittest.mock import patch, MagicMock

os.environ['SANDBOX_PORT'] = '7779'

import sandbox_app as app_module
from sandbox_app import Session, Iteration, extract_think, log_learning_entry, log_failed_attempt, _normalize_pi_inf, _extract_conditions, _norm_conds, _check_conditions, _infer_boundary_points, _strip_tc, CONVERSATION_FILE, SESSION_STATE_FILE, NEXT_SESSION_CONTEXT_FILE, _FAILED_ATTEMPTS_FILE
from sandbox_app import _TOOL_HANDLERS, ToolHandler  # noqa: F401

PASS = 0
FAIL = 0

def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {detail}")

# ─── Helper: isolated temp directory per test ───

class IsolatedFiles:
    """Swap sandbox_app file paths to a temp dir for one test."""
    def __enter__(self):
        self.tmp = tempfile.mkdtemp()
        self._olds = {}
        for attr in ['CONVERSATION_FILE', 'SESSION_STATE_FILE', 'NEXT_SESSION_CONTEXT_FILE', '_FAILED_ATTEMPTS_FILE']:
            self._olds[attr] = getattr(app_module, attr)
            setattr(app_module, attr, os.path.join(self.tmp, os.path.basename(getattr(app_module, attr))))
        return self.tmp
    def __exit__(self, *args):
        for attr, old in self._olds.items():
            setattr(app_module, attr, old)
        shutil.rmtree(self.tmp, ignore_errors=True)

# ====================================================================
# TESTS
# ====================================================================

def test_extract_think():
    print("\n━━━ extract_think ━━━")
    r, t = extract_think("<think>foo bar</think>\n\nbaz qux")
    check("extracts think content", t == "foo bar")
    check("extracts response after think", r == "baz qux")

    r, t = extract_think("no think tags here")
    check("no think tags", r == "no think tags here" and t is None)

    r, t = extract_think("<think>\nmulti\nline\n</think>\n\noutput")
    check("multi-line think", t == "multi\nline" and r == "output")

def test_iteration_from_model():
    print("\n━━━ Iteration.from_model_result ━━━")
    result = {
        "content": "<think>step by step</think>\n\ny = x^2",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "graph", "arguments": '{"x":1}'}}
        ],
        "prompt_tokens": 100,
        "completion_tokens": 50,
    }
    it = Iteration.from_model_result(1, result)
    check("iteration number", it.number == 1)
    check("think extracted", it.think == "step by step")
    check("model response", it.model_response == "y = x^2")
    check("tool names parsed", it.tool_names == ["graph"])
    check("prompt tokens", it.prompt_tokens == 100)

    # No tool calls
    result2 = {"content": "hello", "tool_calls": None}
    it2 = Iteration.from_model_result(2, result2)
    check("empty tool_calls handled", it2.tool_calls == [])

def test_session_persist_and_restore():
    print("\n━━━ Session persistence & recovery ━━━")
    with IsolatedFiles() as tmp:
        s = Session("test input", fresh=True)
        sid = s.session_id
        check("session starts fresh", s.iteration == 0 and s.validated_ok == False)
        s.iteration = 5
        s.validated_ok = True
        s.conditions_declared = True
        s.locked_conditions = {"x<0", "x>=0"}
        s.consecutive_empty = 2
        s.persist()
        check("session_state.json written", os.path.exists(app_module.SESSION_STATE_FILE))
        check("conversation_history.json written", os.path.exists(app_module.CONVERSATION_FILE))
        s2 = Session("test input", fresh=False)
        check("restored session_id", s2.session_id == sid)
        check("restored iteration", s2.iteration == 5)
        check("restored validated_ok", s2.validated_ok == True)
        check("restored conditions_declared", s2.conditions_declared == True)
        check("restored locked_conditions", s2.locked_conditions == {"x<0", "x>=0"})
        check("restored consecutive_empty (reset on continue)", s2.consecutive_empty == 0)

def test_session_fresh_vs_continue():
    print("\n━━━ Session fresh vs continue ━━━")
    with IsolatedFiles() as tmp:
        s = Session("first input", fresh=True)
        sid = s.session_id
        s.iteration = 3
        s.persist()
        s2 = Session("different input", fresh=False)
        check("different input starts new session", s2.session_id > sid)
        check("different input resets iteration", s2.iteration == 0)
        s3 = Session("first input", fresh=False)
        check("same input continues session", s3.iteration == 3)

def test_session_graph_completed_cycles():
    print("\n━━━ Session cycling on graph_completed ━━━")
    with IsolatedFiles() as tmp:
        s = Session("cycle_test", fresh=True)
        base_id = s.session_id
        s.graph_completed = True
        s.persist()
        s2 = Session("cycle_test", fresh=False)
        check("cycles session_id up on graph_completed", s2.session_id == base_id + 1)
        check("unvalidated after cycle", s2.validated_ok == False)
        check("undeclared after cycle", s2.conditions_declared == False)

def test_session_initial_message_injects_context():
    print("\n━━━ NEXT_SESSION_CONTEXT.md injection ━━━")
    with IsolatedFiles() as tmp:
        ctx_path = app_module.NEXT_SESSION_CONTEXT_FILE
        with open(ctx_path, 'w') as f:
            f.write("Accomplished: abliteration\nNext: batch tests")
        s = Session("test", fresh=True)
        msg = s.messages[0]
        check("context injected into initial message", "Accomplished: abliteration" in msg["content"])
        check("user input in message", "test" in msg["content"])

def test_session_trim_to_budget():
    print("\n━━━ Session.trim_to_budget ━━━")
    s = Session("trim test", fresh=True)
    # Add many messages
    s.messages.extend([
        {"role": "assistant", "content": "A" * 5000}
        for _ in range(20)
    ])
    s.trim_to_budget(max_tokens=100)
    check("trim reduces messages", len(s.messages) >= 1)
    check("preserves first message", s.messages[0]["content"].startswith("Use tools to graph"))

def test_get_available_tools():
    print("\n━━━ Session.get_available_tools ━━━")
    s = Session("tools test", fresh=True)
    tools = s.get_available_tools()
    check("declare_conditions when not declared", any(t["function"]["name"] == "declare_conditions" for t in tools))
    check("no validate before declare", not any(t["function"]["name"] == "validate_desmos" for t in tools))
    check("no graph before validate", not any(t["function"]["name"] == "graph" for t in tools))
    s.conditions_declared = True
    tools2 = s.get_available_tools()
    check("validate after declare", any(t["function"]["name"] == "validate_desmos" for t in tools2))
    check("search after declare", any(t["function"]["name"] == "search_internet" for t in tools2))
    check("no graph before validated", not any(t["function"]["name"] == "graph" for t in tools2))
    s.validated_ok = True
    tools3 = s.get_available_tools()
    check("graph after validated", any(t["function"]["name"] == "graph" for t in tools3))

def test_tool_handlers():
    print("\n━━━ Tool handler dispatch ━━━")
    check("all 8 handlers registered", len(_TOOL_HANDLERS) == 8)
    check("graph handler exists", "graph" in _TOOL_HANDLERS)
    check("validate handler exists", "validate_desmos" in _TOOL_HANDLERS)
    check("declare_conditions handler exists", "declare_conditions" in _TOOL_HANDLERS)
    gh = _TOOL_HANDLERS["graph"]
    result = gh.handle({"function_to_graph": "x^2", "format": "LaTeX", "title": "T", "description": "D"})
    check("graph handler returns success", result.get("success") == True)
    check("graph handler returns desmos_json", "desmos_json" in result)
    check("graph handler passes latex through", result["desmos_json"]["latex"] == "x^2")

    vh = _TOOL_HANDLERS["validate_desmos"]
    vr = vh.handle({"expression": ""})
    check("validate empty returns error", vr.get("isError") == True)

    drh = _TOOL_HANDLERS["desmos_reference"]
    dr = drh.handle({"query": "parametric"})
    check("desmos_reference returns string-like", isinstance(dr.get("results"), list))

    lh = _TOOL_HANDLERS["desmos_latex_lookup"]
    lr = lh.handle({"command": "\\sin"})
    check("lookup returns string (may be empty)", isinstance(lr, str))

    sch = _TOOL_HANDLERS["search_past_successes"]
    sr = sch.handle({"query": ""})
    check("empty query returns no results", sr.get("results") == [])

    dch = _TOOL_HANDLERS["declare_conditions"]
    dc = dch.handle({"conditions": ["x<0", "x>=0"]})
    check("declare_conditions returns locked list", "locked_conditions" in dc)
    check("declare_conditions normalizes whitespace", all(" " not in c for c in dc["locked_conditions"]))

def test_strip_tc():
    print("\n━━━ _strip_tc ━━━")
    tc = {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "graph", "arguments": '{"function_to_graph": "x^2", "title": "test"}'}
    }
    stripped = _strip_tc(tc)
    check("preserves id", stripped["id"] == "call_abc")
    args = json.loads(stripped["function"]["arguments"])
    check("summarizes arguments", "function_to_graph" in args)
    check("strips verbose fields", "title" not in args)

def test_normalize_pi_inf():
    print("\n━━━ _normalize_pi_inf ━━━")
    check("normalizes \\pi", _normalize_pi_inf("\\pi") == "π")
    check("normalizes \\infty", _normalize_pi_inf("\\infty") == "∞")
    check("normalizes inf", _normalize_pi_inf("inf") == "∞")
    check("normalizes math.pi", _normalize_pi_inf("math.pi") == "π")
    check("passes through normal text", _normalize_pi_inf("hello") == "hello")

def test_infer_boundary_points():
    print("\n━━━ _infer_boundary_points ━━━")
    bp = _infer_boundary_points({"x<0", "x>=0"})
    check("opposing bounds on same val", "x=0" in bp)
    bp2 = _infer_boundary_points({"x<0", "x>0"})
    check("strict inequalities also produce boundary", "x=0" in bp2)
    bp3 = _infer_boundary_points({"x<0", "x>=0", "x<=6", "x>6"})
    check("multiple boundaries", "x=0" in bp3 and "x=6" in bp3)

def test_log_failed_attempt():
    print("\n━━━ log_failed_attempt ━━━")
    with IsolatedFiles() as tmp:
        log_failed_attempt({"endpoint": "test", "error": "something broke"})
        check("failed_attempts.jsonl created", os.path.exists(app_module._FAILED_ATTEMPTS_FILE))
        with open(app_module._FAILED_ATTEMPTS_FILE) as f:
            line = json.loads(f.readline())
        check("failed attempt has timestamp", "timestamp" in line)
        check("failed attempt has error", line["error"] == "something broke")

def test_messages_list_preserved():
    print("\n━━━ Messages list integrity ━━━")
    s = Session("msg test", fresh=True)
    s.append_message({"role": "user", "content": "follow up"})
    check("append_message works", len(s.messages) == 2)
    s.append_tool_results(
        [{"id": "c1", "type": "function", "function": {"name": "graph", "arguments": "{}"}}],
        [("c1", "done")]
    )
    check("append_tool_results adds assistant msg", len(s.messages) == 4)
    check("tool result is tool role", s.messages[3]["role"] == "tool")

def test_flask_routes_import():
    print("\n━━━ Flask app import ━━━")
    check("app is Flask instance", hasattr(app_module.app, 'route'))
    routes = [r.rule for r in app_module.app.url_map.iter_rules()]
    check("/api/validate_complete route", "/api/validate_complete" in routes)
    check("/api/validate route", "/api/validate" in routes)
    check("/api/abort route", "/api/abort" in routes)
    check("/api/learning_log route", "/api/learning_log" in routes)
    check("root route", "/" in routes)
    check("catchall route", "/<path:filename>" in routes)

# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("SANDBOX APP TEST SUITE")
    print("=" * 55)

    test_extract_think()
    test_iteration_from_model()
    test_session_persist_and_restore()
    test_session_fresh_vs_continue()
    test_session_graph_completed_cycles()
    test_session_initial_message_injects_context()
    test_session_trim_to_budget()
    test_get_available_tools()
    test_tool_handlers()
    test_strip_tc()
    test_normalize_pi_inf()
    test_infer_boundary_points()
    test_log_failed_attempt()
    test_messages_list_preserved()
    test_flask_routes_import()

    print(f"\n{'=' * 55}")
    print(f"TOTAL: {PASS}/{PASS+FAIL} passed")
    if FAIL == 0:
        print("ALL TESTS PASSED ✅")
    else:
        print(f"{FAIL} test(s) FAILED ❌")
    print(f"{'=' * 55}")
