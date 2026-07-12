"""Live registry dashboard + machine-readable manifest feed.

`python -m onramp serve` — single-page view of every registered model:
manifest scores, lifecycle status, drift alerts, and the event stream.
`python -m onramp export` — writes the same state as a JSON feed that
other dashboards (e.g. self-learning-llm-training's) can poll.

Stdlib only; no framework.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from . import events
from .capabilities import CapabilityManifest, detect_drift
from .registry import get_registry


def build_state(events_tail: int = 50) -> dict:
    """Everything a dashboard needs, as plain JSON-able data."""
    registry = get_registry()
    models = []
    for model_id in registry.model_ids():
        manifest = registry.manifest(model_id)
        models.append({
            "model_id": model_id,
            "provider": getattr(registry.get(model_id), "provider", "?"),
            "probed": manifest is not None,
            "manifest": asdict(manifest) if manifest else None,
            "drift": detect_drift(model_id),
            "snapshots": len(CapabilityManifest.history(model_id)),
        })
    return {"models": models, "events": events.tail(events_tail)}


def export_feed(path: str | Path = "manifest-feed.json") -> Path:
    path = Path(path)
    path.write_text(json.dumps(build_state(), indent=2) + "\n")
    return path


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>model-onramp</title>
<style>
 body{font:14px/1.5 -apple-system,system-ui,sans-serif;background:#0f1115;color:#d7dae0;margin:2rem}
 h1{font-size:1.3rem} h1 span{color:#6b7280;font-weight:400}
 table{border-collapse:collapse;width:100%;margin:1rem 0}
 th,td{text-align:left;padding:.45rem .8rem;border-bottom:1px solid #23262e;font-variant-numeric:tabular-nums}
 th{color:#8b90a0;font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em}
 .pill{padding:.1rem .5rem;border-radius:999px;font-size:.75rem}
 .stable{background:#123524;color:#4ade80}.candidate{background:#2b2410;color:#fbbf24}
 .retired{background:#331416;color:#f87171}.unprobed{background:#1f2330;color:#8b90a0}
 .drift{color:#f87171;font-size:.8rem}
 #events{background:#12141a;border:1px solid #23262e;border-radius:8px;padding:1rem;
         max-height:300px;overflow:auto;font:12px ui-monospace,monospace;white-space:pre}
</style></head><body>
<h1>model-onramp <span>— models are plugins, infrastructure is permanent</span></h1>
<table id="models"><thead><tr>
 <th>model</th><th>status</th><th>json</th><th>instr</th><th>tools</th>
 <th>ctx</th><th>tok/s</th><th>$out/M</th><th>snaps</th><th>drift</th>
</tr></thead><tbody></tbody></table>
<h1>events</h1><div id="events"></div>
<script>
async function refresh(){
  const s = await (await fetch('/api/state')).json();
  document.querySelector('#models tbody').innerHTML = s.models.map(m=>{
    const mf = m.manifest||{};
    const st = m.probed ? (mf.status||'candidate') : 'unprobed';
    const f = v => v==null ? '—' : v;
    return `<tr><td>${m.model_id}</td><td><span class="pill ${st}">${st}</span></td>
     <td>${f(mf.json_reliability)}</td><td>${f(mf.instruction_score)}</td>
     <td>${f(mf.tool_use_reliability)}</td><td>${f(mf.usable_context_tokens)}</td>
     <td>${f(mf.tokens_per_second)}</td><td>${f(mf.output_per_mtok)}</td>
     <td>${m.snapshots}</td><td class="drift">${m.drift.join('<br>')||''}</td></tr>`;
  }).join('');
  document.getElementById('events').textContent =
    s.events.slice().reverse().map(e=>JSON.stringify(e)).join('\\n');
}
refresh(); setInterval(refresh, 3000);
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/state":
            body = json.dumps(build_state()).encode()
            content_type = "application/json"
        elif self.path == "/":
            body = PAGE.encode()
            content_type = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):  # keep stdout clean
        pass


def serve(port: int = 8010) -> None:
    print(f"model-onramp dashboard: http://localhost:{port}")
    HTTPServer(("0.0.0.0", port), _Handler).serve_forever()
