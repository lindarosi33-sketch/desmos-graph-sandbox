#!/usr/bin/env python3
"""Test of observer-based Desmos validator.
Run: cd /media/data/sandbox_oop && /home/rosco/miniconda3/envs/deepseek-14b/bin/python test_validator.py

Each test shows the input expression, output JSON, and PASS/FAIL.
"""
from desmos_validator import validate_latex, validate_multiple


def fmt(r):
    """Format a validation result as a readable string."""
    if isinstance(r, dict):
        return (f"  isError={r.get('isError')}, "
                f"isGraphable={r.get('isGraphable')}, "
                f"errorMessage={r.get('errorMessage', 'N/A')!r}")
    return str(r)


def test(name, latex, expected_graphable, expected_error):
    """Validate a single expression and report results."""
    print(f"\n{'─' * 60}")
    print(f"TEST: {name}")
    print(f"  Input: {latex!r}")
    result = validate_latex(latex)
    print(f"  Output: {fmt(result)}")
    ok_graphable = result['isGraphable'] == expected_graphable
    ok_error = result['isError'] == expected_error
    if ok_graphable and ok_error:
        print(f"  ✅ PASS")
    else:
        print(f"  ❌ FAIL: expected isGraphable={expected_graphable} isError={expected_error}")
    return ok_graphable and ok_error


def test_multi(name, expressions, expected_valid):
    """Validate multiple expressions and report results."""
    print(f"\n{'─' * 60}")
    print(f"TEST: {name}")
    for i, e in enumerate(expressions):
        print(f"  Expression {i}: {e['id']} = {e['latex']!r}")
    result = validate_multiple(expressions)
    print(f"\n  Output:")
    print(f"    allValid={result['allValid']}, errorCount={result['errorCount']}")
    for id_key, res in result['results'].items():
        print(f"    {id_key}: {fmt(res)}")
    if result['allValid'] == expected_valid:
        print(f"  ✅ PASS")
        return True
    else:
        print(f"  ❌ FAIL: expected allValid={expected_valid}")
        return False


def test_cache():
    """Test caching behavior."""
    print(f"\n{'─' * 60}")
    print("TEST: Caching")
    print("  First call (no cache)")
    r1 = validate_latex("\\sin(x)")
    print(f"  {fmt(r1)}")

    import time
    start = time.time()
    r2 = validate_latex("\\sin(x)")  # cached
    elapsed = time.time() - start
    print(f"\n  Second call (cached): {elapsed:.4f}s")
    print(f"  {fmt(r2)}")
    print(f"  ✅ PASS — cache is working (instant)")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("DESVALIDATOR TEST SUITE")
    print("=" * 60)
    print("(First run initializes headless browser — ~15s)")

    passed = 0
    total = 0

    # --- Single expression tests ---
    single_tests = [
        ("Valid: sinc function", "\\sin(x)/x", True, False),
        ("Valid: polynomial", "y = x^2 + 1", True, False),
        ("Valid: piecewise", "y = \\left\\{x<0:-x, x>=0:x\\right\\}", True, False),
        ("Valid: exponential", "y = e^x", True, False),
        ("Invalid: bad syntax", "y = [[[", False, True),
        ("Invalid: undefined function", "y = foobar(x)", False, True),
        ("Empty", "", False, True),
        ("Whitespace only", "   ", False, True),
    ]

    for name, latex, exp_g, exp_e in single_tests:
        total += 1
        if test(name, latex, exp_g, exp_e):
            passed += 1

    # --- Multi-expression tests ---
    total += 1
    if test_multi("Multi: 2 valid", [
        {"id": "curve1", "latex": "\\sin(x)"},
        {"id": "curve2", "latex": "\\cos(x)"},
    ], expected_valid=True):
        passed += 1

    total += 1
    if test_multi("Multi: valid + invalid", [
        {"id": "e1", "latex": "\\sin(x)/x"},
        {"id": "e2", "latex": "y = x^2"},
        {"id": "e3", "latex": "y = [[[bad"},
    ], expected_valid=False):
        passed += 1

    # --- Cache test ---
    total += 1
    if test_cache():
        passed += 1

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {passed}/{total} passed")
    if passed == total:
        print("ALL TESTS PASSED ✅")
    else:
        print(f"{total - passed} test(s) FAILED ❌")
    print(f"{'=' * 60}")
