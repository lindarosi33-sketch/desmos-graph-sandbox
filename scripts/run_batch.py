#!/usr/bin/env python3
"""Run a batch of expressions through the sandbox and record results."""
import json
import os
import sys
import time
from urllib import error, request

SANDBOX_URL = os.environ.get('SANDBOX_URL', 'http://localhost:7778')
TIMEOUT = int(os.environ.get('BATCH_TIMEOUT', '999999'))
DELAY = int(os.environ.get('BATCH_DELAY', '3'))


def call_sandbox(expression: str, fresh: bool = True) -> dict:
    payload = json.dumps({"input": expression, "fresh": fresh}).encode()
    req = request.Request(
        f"{SANDBOX_URL}/api/validate_complete?stream=false",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = request.urlopen(req)
        return json.loads(resp.read())
    except error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    expr_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'expressions.txt')
    with open(expr_file) as f:
        expressions = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    results = []
    for i, expr in enumerate(expressions, 1):
        ok = False
        attempt = 0
        while not ok:
            attempt += 1
            print(f"[{i}/{len(expressions)}] {expr}")
            if attempt > 1:
                print(f"  (attempt {attempt})")
            result = call_sandbox(expr, fresh=(attempt == 1))
            ok = result.get("success") and result.get("graph_data") is not None
            iterations = result.get("iterations", "?")
            print(f"  {'OK' if ok else 'FAIL'}  ({iterations} iterations)")
            if ok:
                latex = result.get('validated_latex') or 'N/A'
                print(f"  LaTeX: {latex[:80]}")
            else:
                print(f"  Error: {result.get('error', 'unknown')[:80]}")
            results.append({"expression": expr, "success": ok, "result": result, "attempt": attempt})
            if not ok:
                time.sleep(DELAY)

    successes = sum(1 for r in results if r["success"])
    print(f"\n=== DONE: {successes}/{len(results)} succeeded ===")

    out_path = os.path.join(os.path.dirname(__file__), 'batch_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == '__main__':
    main()
