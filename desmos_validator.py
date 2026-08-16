import concurrent.futures
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger('desmos_validator')

_playwright = None
_browser = None
_page = None

_BROWSER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="playwright")


def _call(fn, *args):
    """Run Playwright operations in a dedicated thread to avoid async event loop conflicts."""
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            return _BROWSER_EXECUTOR.submit(fn, *args).result(timeout=60)
        except concurrent.futures.TimeoutError:
            logger.warning("Playwright call timed out (60s), retrying...")
            continue
    raise TimeoutError("Playwright validation timed out after 120 seconds")

_playwright = None
_browser = None
_page = None

def _build_html(api_key: str = "") -> str:
    """Build HTML page with Desmos API and observer-based validation functions."""
    html = """<!DOCTYPE html>
<html><head>
<script src="https://www.desmos.com/api/v1.9/calculator.js?apiKey=__DESMOS_API_KEY__"></script>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.0/jquery.min.js"></script>
<style>body{margin:0}#c{width:400px;height:300px}</style>
</head><body>
<div id="c"></div>
<script>
var v = Desmos.GraphingCalculator(document.getElementById('c'), {
    expressions: false, keypad: false, sliders: false, settingsMenu: false, border: false
});

// Observer state: accumulates analysis for ALL expression IDs over time
var _latestAnalysis = {};

// For single-expression validation
var _singleResolve = null;

v.observe('expressionAnalysis', function () {
    // Accumulate analysis for ALL IDs (not just the latest one)
    for (var id in v.expressionAnalysis) {
        _latestAnalysis[id] = {
            isError: v.expressionAnalysis[id].isError,
            isGraphable: v.expressionAnalysis[id].isGraphable,
            errorMessage: v.expressionAnalysis[id].errorMessage || null
        };
    }
    // If single-expression validation is waiting, resolve it
    if (_singleResolve && _latestAnalysis['v']) {
        var resolve = _singleResolve;
        _singleResolve = null;
        var result = _latestAnalysis['v'];
        setTimeout(function(r) { resolve(r); }, 0, result);
    }
});

function validateLatex(latex) {
    return new Promise(function(resolve) {
        try {
            v.setBlank();
            v.setExpression({ id: 'v', latex: latex });
            // Settle via observer or timeout
            _singleResolve = resolve;
            setTimeout(function() {
                if (_singleResolve) {
                    var r = _latestAnalysis['v'] || { isError: true, isGraphable: false, errorMessage: 'Observer timeout' };
                    _singleResolve(r);
                    _singleResolve = null;
                }
            }, 5000);
        } catch(e) {
            resolve({isError: true, isGraphable: false, errorMessage: 'Expression error: ' + e.message});
        }
    });
}

function validateMultiple(expressions) {
    return new Promise(function(resolve) {
        var ids = [];
        var allResults = {};
        var errors = [];

        try {
            v.setBlank();
            for (var i = 0; i < expressions.length; i++) {
                v.setExpression({ id: expressions[i].id, latex: expressions[i].latex });
                ids.push(expressions[i].id);
            }

            // Poll _latestAnalysis until all IDs have been analyzed
            var pollInterval = setInterval(function() {
                for (var i = 0; i < ids.length; i++) {
                    if (_latestAnalysis[ids[i]]) {
                        allResults[ids[i]] = {
                            latex: (expressions.find(function(e) { return e.id === ids[i]; }) || {}).latex,
                            isError: _latestAnalysis[ids[i]].isError,
                            isGraphable: _latestAnalysis[ids[i]].isGraphable,
                            errorMessage: _latestAnalysis[ids[i]].errorMessage || 'No analysis'
                        };
                        if (_latestAnalysis[ids[i]].isError) {
                            errors.push({
                                id: ids[i],
                                latex: (expressions.find(function(e) { return e.id === ids[i]; }) || {}).latex,
                                error: _latestAnalysis[ids[i]].errorMessage || 'No analysis'
                            });
                        }
                    }
                }
                // Check if all IDs have been resolved
                var allDone = ids.every(function(checkId) { return checkId in allResults; });
                if (allDone) {
                    clearInterval(pollInterval);
                    resolve({
                        results: allResults,
                        allValid: errors.length === 0,
                        errorCount: errors.length,
                        errors: errors
                    });
                }
            }, 100);

            // Safety timeout: 10 seconds
            setTimeout(function() {
                clearInterval(pollInterval);
                var allDone = ids.every(function(checkId) { return checkId in allResults; });
                if (!allDone) {
                    resolve({
                        results: allResults,
                        allValid: errors.length === 0,
                        errorCount: errors.length,
                        errors: errors
                    });
                }
            }, 10000);
        } catch(e) {
            resolve({allValid: false, errorCount: 1, errors: [{ error: 'Expression error: ' + e.message }], results: {}});
        }
    });
}
</script>
</body></html>"""
    return html.replace("__DESMOS_API_KEY__", api_key)

_CACHE: dict[str, dict] = {}

def _on_browser_console(msg):
    logger.info("BROWSER CONSOLE [%s]: %s", msg.type, msg.text[:200])

def _on_browser_error(err):
    logger.warning("BROWSER JS ERROR: %s", str(err)[:300])

def _on_browser_crash():
    logger.error("PLAYWRIGHT BROWSER CRASHED - page will be recreated on next call")

def _close_browser():
    global _playwright, _browser, _page
    try:
        if _page:
            _page.close()
    except Exception:
        pass
    try:
        if _browser:
            _browser.close()
    except Exception:
        pass
    try:
        if _playwright:
            _playwright.stop()
    except Exception:
        pass
    _page = None
    _browser = None
    _playwright = None

def start():
    global _playwright, _browser, _page
    _close_browser()
    _playwright = sync_playwright().start()
    _browser = _playwright.chromium.launch(headless=True, args=['--disable-gpu', '--disable-software-rasterizer'])
    _page = _browser.new_page()
    _page.on("console", _on_browser_console)
    _page.on("pageerror", _on_browser_error)
    _page.on("crash", _on_browser_crash)
    _page.set_viewport_size({"width": 420, "height": 400})
    _page.goto('about:blank')
    _page.set_content(_build_html(os.getenv("DESMOS_API_KEY", "")), wait_until='domcontentloaded')
    try:
        _page.wait_for_function("typeof Desmos !== 'undefined' && Desmos.GraphingCalculator", timeout=15000)
    except Exception as e:
        logger.warning("Desmos API load timeout: %s", e)
        raise
    logger.info("Desmos validator browser initialized")

def _validate_on_page(latex: str) -> dict:
    """Run single-expression validation against an initialized page using observer."""
    result = _page.evaluate("validateLatex", latex)
    _CACHE[latex] = result
    return result

def _validate_multiple_on_page(expressions: list) -> dict:
    """Run multi-expression validation against an initialized page using observer."""
    return _page.evaluate("validateMultiple", expressions)

def _browser_healthy() -> bool:
    """Quick health check: evaluate a trivial expression to confirm the browser is responsive."""
    if not _page:
        return False
    try:
        _page.evaluate("1 + 1")
        return True
    except Exception:
        return False

def _ensure_browser():
    """Ensure the browser page is alive, restart if dead."""
    if not _browser_healthy():
        logger.info("Browser health check failed, restarting...")
        _call(start)

def validate_latex(latex: str) -> dict:
    """Validate a single LaTeX expression against Desmos rendering.

    Uses observer-based validation via Desmos API's expressionAnalysis observable
    instead of fixed setTimeout delays. Provides deterministic results.
    """
    if not latex:
        return {"isError": True, "errorMessage": "Empty expression", "isGraphable": False}
    if latex in _CACHE:
        return _CACHE[latex]
    if not _page:
        _call(start)
    else:
        _ensure_browser()
    try:
        return _call(_validate_on_page, latex)
    except Exception as e:
        _err_str = str(e).lower()
        if any(x in _err_str for x in ["closed", "crashed", "target", "cannot switch", "asyncio", "sync api"]):
            logger.warning("Browser page dead, recreating: %s", e)
            _call(start)
            try:
                return _call(_validate_on_page, latex)
            except Exception as e2:
                logger.error("Validation failed after browser restart: %s", e2)
                return {"isError": True, "errorMessage": f"Browser error: {e2}", "isGraphable": False}
        logger.warning("Validation error for %r: %s", latex, e)
        return {"isError": True, "errorMessage": str(e), "isGraphable": False}

def validate_multiple(expressions: list[dict]) -> dict:
    """Validate multiple LaTeX expressions against Desmos rendering simultaneously.

    Uses observer-based validation with polling: sets all expressions on the
    calculator, then polls the accumulated analysis until all IDs are resolved.

    Args:
        expressions: List of dicts with keys:
            - id (str): Unique identifier for this expression
            - latex (str): LaTeX expression to validate

    Returns:
        Dict with:
            - results: Dict mapping id -> {latex, isError, isGraphable, errorMessage}
            - allValid: True if all expressions are graphable
            - errorCount: Number of invalid expressions
            - errors: List of failed expressions with details
    """
    if not expressions:
        return {"allValid": False, "errorCount": 0, "errors": [], "results": {}}

    if not _page:
        _call(start)
    try:
        return _call(_validate_multiple_on_page, expressions)
    except Exception as e:
        _err_str = str(e).lower()
        if any(x in _err_str for x in ["closed", "crashed", "target", "cannot switch", "asyncio", "sync api"]):
            logger.warning("Browser page dead, recreating: %s", e)
            _call(start)
            try:
                return _call(_validate_multiple_on_page, expressions)
            except Exception as e2:
                logger.error("Multi-validation failed after browser restart: %s", e2)
                return {"allValid": False, "errorCount": len(expressions), "errors": [{"error": f"Browser error: {e2}"}], "results": {}}
        logger.warning("Multi-validation error for %r: %s", [e.get('latex') for e in expressions], e)
        return {"allValid": False, "errorCount": len(expressions), "errors": [{"error": str(e)}], "results": {}}

def cleanup():
    _close_browser()
