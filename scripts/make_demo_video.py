#!/usr/bin/env python
"""Record a replay-driven demo video (GIF) of the Desmos sandbox harness.

Serves a throwaway mock of the 4 endpoints the harness needs, replays a
recorded session transcript as SSE (no llama-server, no model), drives the
UI in headless Chromium via Playwright, and encodes the frames to a GIF.

Usage:
    python scripts/make_demo_video.py [--transcript /path/to/session.json]
                                      [--out demo/demo.gif]
                                      [--pace 0.55] [--fps 6] [--width 800]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from flask import Flask, Response, jsonify, send_from_directory

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

app = Flask(__name__)
TRANSCRIPT = {}
DEMO_PACE = 0.55
PROGRESS = 0


def load_env():
    """Minimal .env parser (key=value, # comments)."""
    env = {}
    try:
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return env


@app.route("/")
@app.route("/<path:filename>")
def serve_harness(filename="desmos_harness.html"):
    return send_from_directory(os.path.join(PROJECT_ROOT, "templates"), filename)


@app.route("/api/config")
def api_config():
    env = load_env()
    return jsonify({"desmosApiKey": env.get("DESMOS_API_KEY", "")})


@app.route("/api/validate_complete", methods=["POST"])
def api_validate_complete():
    def gen():
        global PROGRESS
        PROGRESS = 0
        sid = TRANSCRIPT.get("session_id", 1011)
        entries = TRANSCRIPT.get("entries", [])
        complete = TRANSCRIPT.get("complete", {})
        yield f"data: session_id:{sid}\n\n"
        time.sleep(DEMO_PACE)
        for _ in entries:
            PROGRESS += 1
            yield "data: log_update\n\n"
            time.sleep(DEMO_PACE)
        yield f"data: complete:{json.dumps(complete)}\n\n"

    return Response(gen(), mimetype="text/event-stream")


@app.route("/api/learning_log")
def api_learning_log():
    entries = TRANSCRIPT.get("entries", [])
    return jsonify({"entries": entries[:PROGRESS] if PROGRESS else []})


def start_server(pace):
    global DEMO_PACE
    DEMO_PACE = pace
    threading.Thread(target=lambda: app.run(host="127.0.0.1", port=7799, debug=False, threaded=True), daemon=True).start()
    # wait for server to come up
    import urllib.request
    for _ in range(50):
        try:
            urllib.request.urlopen("http://127.0.0.1:7799/api/config", timeout=1)
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("mock server failed to start")


def record(transcript_path, out_path, pace, fps, width, hold=3.0):
    from playwright.sync_api import sync_playwright

    with open(transcript_path) as f:
        TRANSCRIPT.update(json.load(f))

    start_server(pace)
    entries = TRANSCRIPT.get("entries", [])

    frame_dir = tempfile.mkdtemp(prefix="demo_frames_")
    view_w = view_h = 1000
    display = ":99"
    procs = []
    try:
        # clean any stale Xvfb/chrome on our display
        subprocess.run(["pkill", "-f", f"Xvfb {display}"], capture_output=True)
        subprocess.run(["pkill", "-f", "chromium-1234/chrome"], capture_output=True)
        subprocess.run(["pkill", "-f", "ffmpeg.*x11grab"], capture_output=True)
        time.sleep(0.5)

        env = os.environ.copy()
        env["DISPLAY"] = display

        # virtual framebuffer for the headed browser
        xvfb = subprocess.Popen(["Xvfb", display, "-screen", "0", "1280x1024x24"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        procs.append(xvfb)
        time.sleep(1)

        browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", os.path.expanduser("~/.cache/ms-playwright"))
        chrome_path = None
        for name in ("chromium-1234", "chromium-1217"):
            candidate = os.path.join(browsers_path, name, "chrome-linux64", "chrome")
            if os.path.exists(candidate):
                chrome_path = candidate
                break

        with sync_playwright() as p:
            launch_args = ["--no-sandbox", "--kiosk", "--window-size=1000,1000", "--window-position=0,0"]
            if chrome_path:
                browser = p.chromium.launch(headless=False, executable_path=chrome_path, args=launch_args,
                                            env=env)
            else:
                browser = p.chromium.launch(headless=False, args=launch_args, env=env)
            page = browser.new_page(viewport={"width": view_w, "height": view_h})
            page.goto("http://127.0.0.1:7799/")
            page.wait_for_function("typeof calc !== 'undefined' && calc !== null", timeout=30000)

            # Reflow layout for a compact, video-friendly frame:
            # move Learning Log above the calculator. The log is left to grow
            # freely (no max-height / internal scroll) so its growth repaints;
            # the newest iteration block appears at the top of the log as it
            # streams, and the calculator scrolls into view on the final graph.
            page.evaluate("""() => {
                var logTitle = document.querySelector('h3');
                var log = document.getElementById('log');
                var refresh = document.querySelector('button[onclick="loadLog()"]');
                var calc = document.getElementById('calculator');
                if (calc) { calc.style.height = '300px'; }
                if (log) { log.style.overflow = 'visible'; log.style.marginBottom = '10px'; }
                var status = document.getElementById('status');
                if (status) {
                    status.parentNode.insertBefore(logTitle, calc);
                    status.parentNode.insertBefore(log, calc);
                    status.parentNode.insertBefore(refresh, calc);
                }
            }""")

            # Set the prompt
            user_input = TRANSCRIPT.get("complete", {}).get("user_input", "")
            page.fill("#input", user_input)
            time.sleep(1)

            # start x11grab capture of the headed browser on the virtual display
            pattern = os.path.join(frame_dir, "frame_%04d.png")
            cap_duration = pace * (len(entries) + 2) + 6
            ffmpeg = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "x11grab", "-framerate", str(fps),
                 "-video_size", f"{view_w}x{view_h}", "-i", display,
                 "-t", str(cap_duration), pattern],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
            procs.append(ffmpeg)
            time.sleep(0.6)

            # fire-and-forget: page.evaluate would await the async validateOnly()
            page.evaluate("setTimeout(() => validateOnly(), 50)")

            # record until complete
            settled = 0
            t0 = time.time()
            done_status = None
            while True:
                now = time.time()
                status_text = page.text_content("#status-text") or ""
                if done_status is None and ("Graphed!" in status_text or status_text.startswith("Validated")):
                    done_status = status_text
                if done_status:
                    settled += 1
                    if settled >= 125:  # ~2.5s of settle frames (graph render + scroll)
                        break
                if now - t0 > 90:
                    break
                time.sleep(0.02)

            print(f"status={done_status!r}")
            # stop capture on the stable graph frame BEFORE closing the browser,
            # otherwise the tail frames (and the hold copy) would be black.
            ffmpeg.terminate()
            ffmpeg.wait()
            browser.close()

        frames = sorted(os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".png"))
        # hold on the final (graphed) frame so the video ends on the graph
        if hold > 0 and frames:
            last = frames[-1]
            def frame_no(path):
                return int(os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[1])
            start = max(frame_no(f) for f in frames) + 1
            for i in range(int(hold * fps)):
                path = os.path.join(frame_dir, f"frame_{start + i:04d}.png")
                shutil.copy(last, path)
                frames.append(path)
        print(f"captured {len(frames)} frames")

        # encode with ffmpeg
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        pattern = os.path.join(frame_dir, "frame_%04d.png")
        tmp_gif = out_path + ".tmp.gif"
        # palette-gen then palette-use for quality (input framerate = fps so no resampling)
        pal = os.path.join(frame_dir, "palette.png")
        cmd1 = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern, "-vf",
                f"fps={fps},scale={width}:-1:flags=lanczos,palettegen", pal]
        cmd2 = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern, "-i", pal, "-lavfi",
                f"fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse", tmp_gif]
        for cmd in (cmd1, cmd2):
            r = subprocess.run(cmd)
            if r.returncode != 0:
                raise RuntimeError("ffmpeg failed: " + " ".join(cmd))
        shutil.move(tmp_gif, out_path)
        print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")
        return out_path
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default=os.path.join(PROJECT_ROOT, "demo", "demo_session.json"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "demo", "demo.gif"))
    ap.add_argument("--pace", type=float, default=0.55)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--hold", type=float, default=3.0)
    args = ap.parse_args()
    record(args.transcript, args.out, args.pace, args.fps, args.width, args.hold)


if __name__ == "__main__":
    sys.exit(main())