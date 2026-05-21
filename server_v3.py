"""Flask server for the self-learning dashboard.

Endpoints:
  GET /                  → dashboard.html
  GET /api/state         → live snapshot (rounds, current phase, agent verdicts)
  GET /api/traces        → recent Braintrust traces (or local fallback)
  GET /api/human         → pending human-judge decisions
  GET /api/events?n=200  → tail of local event stream (per-agent log)
"""
from __future__ import annotations

import json
import os
from flask import Flask, jsonify, send_from_directory

from braintrust_bridge import read_snapshot, fetch_recent_logs, read_human_queue, STATE_PATH

app = Flask(__name__)
HERE = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def root():
    return send_from_directory(HERE, "dashboard.html")


@app.route("/architecture.html")
def arch():
    return send_from_directory(HERE, "architecture.html")


@app.route("/presentation.html")
def deck():
    return send_from_directory(HERE, "presentation.html")


@app.route("/api/state")
def state():
    return jsonify(read_snapshot())


@app.route("/api/traces")
def traces():
    return jsonify(fetch_recent_logs(limit=50))


@app.route("/api/human")
def human():
    return jsonify(read_human_queue())


@app.route("/api/events")
def events():
    try:
        with open(STATE_PATH) as f:
            lines = f.readlines()[-300:]
        return jsonify([json.loads(l) for l in lines if l.strip()])
    except FileNotFoundError:
        return jsonify([])


@app.route("/api/results")
def results():
    try:
        with open(os.path.join(HERE, "results.json")) as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"status": "not_run_yet"})


if __name__ == "__main__":
    print("Dashboard: http://localhost:8001")
    app.run(host="0.0.0.0", port=8001, debug=False)
