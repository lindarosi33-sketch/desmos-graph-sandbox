#!/usr/bin/env python3
"""Tests for new sandbox features: desmos_reference, validate_multiple, graph params.
Run: cd /media/data/sandbox_oop && python test_new_tools.py
"""
import os, sys

os.environ["SANDBOX_PORT"] = "7779"

from desmos_validator import validate_latex, validate_multiple

PASS = 0
FAIL = 0

def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print("  [PASS] " + name)
    else:
        FAIL += 1
        print("  [FAIL] " + name + " -- " + str(detail))


def test_tool_handler_count():
    sys.path.insert(0, "/media/data/sandbox_oop")
    import sandbox_app as app_module
    from sandbox_app import _TOOL_HANDLERS

    print("\n=== Tool handler count ===")
    handler_names = list(_TOOL_HANDLERS.keys())
    check("8 handlers registered", len(_TOOL_HANDLERS) == 8, "got " + str(len(_TOOL_HANDLERS)) + ": " + str(handler_names))

    expected = {"declare_conditions", "validate_desmos", "graph", "search_internet",
                "search_past_successes", "desmos_latex_lookup", "desmos_reference",
                "validate_desmos_multiple"}
    check("all expected handlers present", set(handler_names) == expected, "got " + str(set(handler_names)))

    check("no glob handler", "glob" not in _TOOL_HANDLERS)
    check("no read handler", "read" not in _TOOL_HANDLERS)
    check("desmos_reference handler exists", "desmos_reference" in _TOOL_HANDLERS)


def test_desmos_reference_handler():
    print("\n=== desmos_reference handler ===")
    sys.path.insert(0, "/media/data/sandbox_oop")
    import sandbox_app as app_module
    from sandbox_app import _TOOL_HANDLERS

    handler = _TOOL_HANDLERS["desmos_reference"]

    result = handler.handle({"query": ""})
    check("empty query returns empty results", result.get("results") == [], str(result))
    check("empty query returns success", result.get("success") == True)

    result = handler.handle({"query": "parametric"})
    check("parametric query returns results", len(result.get("results", [])) > 0, str(result))
    if result.get("results"):
        check("result has score", "score" in result["results"][0])
        check("result has file", "file" in result["results"][0])
        check("result has sections", "sections" in result["results"][0])
        sec = result["results"][0].get("sections", [])
        check("sections are strings", all(isinstance(s, str) for s in sec))

    result = handler.handle({"query": "xyzzy_no_such_term"})
    check("unknown term returns no results", len(result.get("results", [])) == 0)

    result = handler.handle({"query": "domain restriction"})
    check("domain restriction query returns results", len(result.get("results", [])) > 0)


def test_validate_single():
    print("\n=== validate_desmos single expression ===")
    sys.path.insert(0, "/media/data/sandbox_oop")
    from sandbox_app import _TOOL_HANDLERS

    vh = _TOOL_HANDLERS["validate_desmos"]

    result = vh.handle({"expression": "y = \\sin(x)"})
    check("valid sin passes", result.get("isGraphable") == True and result.get("isError") == False)

    result = vh.handle({"expression": "y = [[[bad"})
    check("bad latex fails", result.get("isError") == True)

    result = vh.handle({"expression": ""})
    check("empty expression fails", result.get("isError") == True)


def test_validate_multiple_list():
    print("\n=== validate_desmos multiple expressions ===")
    sys.path.insert(0, "/media/data/sandbox_oop")
    from sandbox_app import _TOOL_HANDLERS

    vh = _TOOL_HANDLERS["validate_desmos"]

    result = vh.handle({"expression": ["y = \\sin(x)", "y = \\cos(x)"]})
    check("multi valid expressions", result.get("allValid") == True, str(result))
    check("multi zero errors", result.get("errorCount") == 0, str(result))

    result = vh.handle({"expression": ["y = \\sin(x)", "y = [[[bad", "y = \\cos(x)"]})
    check("multi with invalid fails", result.get("allValid") == False, str(result))
    check("multi error count correct", result.get("errorCount") == 1, str(result))


def test_graph_handler_params():
    print("\n=== graph handler with new params ===")
    sys.path.insert(0, "/media/data/sandbox_oop")
    from sandbox_app import _TOOL_HANDLERS

    gh = _TOOL_HANDLERS["graph"]

    # Basic graph
    result = gh.handle({"function_to_graph": "y = x^2", "format": "LaTeX"})
    check("basic graph succeeds", result.get("success") == True)
    check("basic graph has desmos_json", "desmos_json" in result)
    check("latex passes through", result["desmos_json"]["latex"] == "y = x^2")

    # Graph with points
    result = gh.handle({
        "function_to_graph": "y = x^2",
        "points": [{"x": 1, "y": 1}, {"x": 2, "y": 4}],
    })
    check("graph with points succeeds", result.get("success") == True)
    check("points passed through", result["desmos_json"].get("points") == [{"x": 1, "y": 1}, {"x": 2, "y": 4}])

    # Point size remap
    result = gh.handle({
        "function_to_graph": "y = x^2",
        "points": [{"x": 0, "y": 0, "size": 3}],
    })
    check("point size remapped", result["desmos_json"]["points"][0].get("pointSize") == 3)
    check("size key removed", "size" not in result["desmos_json"]["points"][0])

    # Lines
    result = gh.handle({
        "function_to_graph": "y = x^2",
        "lines": [{"from": [0, 0], "to": [5, 25]}],
    })
    check("lines passed through", result["desmos_json"].get("lines") is not None)

    # Point opacity
    result = gh.handle({"function_to_graph": "y = x^2", "pointOpacity": 0.5})
    check("pointOpacity passed through", result["desmos_json"].get("pointOpacity") == 0.5)

    # Slider bounds
    result = gh.handle({"function_to_graph": "y = ax^2", "sliderBounds": {"a": [1, 5, 0.1]}})
    check("sliderBounds passed through", result["desmos_json"].get("sliderBounds") == {"a": [1, 5, 0.1]})


def test_desmos_reference_integration():
    print("\n=== desmos_reference integration ===")
    docs_dir = "/media/data/sandbox_oop/docs"
    check("docs dir exists", os.path.isdir(docs_dir))

    doc_files = [f for f in os.listdir(docs_dir) if f.startswith("desmos-api")]
    check("has desmos-api docs", len(doc_files) > 0, "found " + str(doc_files))

    for f in doc_files[:3]:
        filepath = os.path.join(docs_dir, f)
        with open(filepath) as fh:
            content = fh.read()
        check("has content " + f, len(content) > 100, "file has " + str(len(content)) + " bytes")

    sys.path.insert(0, "/media/data/sandbox_oop")
    from sandbox_app import _TOOL_HANDLERS
    handler = _TOOL_HANDLERS["desmos_reference"]

    result = handler.handle({"query": "the"})
    for r in result.get("results", []):
        fname = r.get("file", "unknown")
        check("sections <= 20 (" + fname + ")", len(r.get("sections", [])) <= 20)


if __name__ == "__main__":
    print("=" * 55)
    print("NEW TOOLS TEST SUITE")
    print("=" * 55)

    test_tool_handler_count()
    test_desmos_reference_handler()
    test_validate_single()
    test_validate_multiple_list()
    test_graph_handler_params()
    test_desmos_reference_integration()

    print("\n" + "=" * 55)
    print("TOTAL: " + str(PASS) + "/" + str(PASS + FAIL) + " passed")
    if FAIL == 0:
        print("ALL TESTS PASSED")
    else:
        print(str(FAIL) + " test(s) FAILED")
    print("=" * 55)
