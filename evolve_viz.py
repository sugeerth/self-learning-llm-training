"""Render an evolutionary run as two views that make selection legible.

`evolve.py` records every evaluated genome with its generation, parents and
fitness. Two things are hard to see in a table of numbers and easy to see in a
picture, so this draws both from that record (no external deps, one self-
contained HTML with inline SVG, light/dark aware):

  1. GENEALOGY — every genome is a node, parent->child edges trace descent,
     vertical position IS fitness (best at the top) and colour reinforces it.
     You watch bad lineages terminate and the winning genes get bred forward;
     an elite that keeps reproducing fans edges across several generations.

  2. CONFIG-SPACE SEARCH — every genome plotted at its architecture
     (d_model x depth), coloured by generation. Generation 0 scatters; later
     generations concentrate toward the winning region. This is the shape of
     the search itself — the thing a regret curve summarises but hides.

    python3 evolve_viz.py            # reads evolve_report.json -> evolve_report.html
"""
from __future__ import annotations

import json
import math
import os

REPORT_JSON = os.path.join(os.path.dirname(__file__), "evolve_report.json")
REPORT_HTML = os.path.join(os.path.dirname(__file__), "evolve_report.html")

D_MODELS = [192, 256, 320, 384, 448, 512]
N_LAYERS = [2, 4, 6, 8]


# ────────────────────────── colour ──────────────────────────

def _lerp(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _hex(c: tuple) -> str:
    return "#%02x%02x%02x" % c


def _ramp(t: float, stops: list[tuple]) -> str:
    """Piecewise-linear colour ramp over evenly spaced stops, t in [0,1]."""
    t = max(0.0, min(1.0, t))
    seg = t * (len(stops) - 1)
    i = min(int(seg), len(stops) - 2)
    return _hex(_lerp(stops[i], stops[i + 1], seg - i))


# best -> worst fitness (low ppl good): teal, amber, muted grey
FIT_STOPS = [(15, 157, 88), (244, 180, 0), (154, 160, 166)]
# generation 0 -> last: pale blue to the evolve arm's amber
GEN_STOPS = [(170, 200, 240), (234, 134, 0)]


# ────────────────────────── genealogy panel ──────────────────────────

def _genealogy_svg(lineage: list[dict], best_id: str, W: int, H: int) -> str:
    PADX, PADT, PADB = 60, 34, 40
    gens = max(g["gen"] for g in lineage)
    ppls = [g["ppl"] for g in lineage if math.isfinite(g["ppl"])]
    lo, hi = min(ppls), max(ppls)
    # log-ppl axis so the crowded good region isn't compressed to a sliver
    llo, lhi = math.log(lo), math.log(hi)
    span = max(lhi - llo, 1e-9)
    pmax = max(g["params_m"] for g in lineage) or 1.0

    def X(gen: int) -> float:
        if gens == 0:
            return W / 2
        return PADX + (W - 2 * PADX) * gen / gens

    def Y(ppl: float) -> float:  # best (low ppl) near the top
        return PADT + (H - PADT - PADB) * (math.log(ppl) - llo) / span

    pos = {g["id"]: (X(g["gen"]), Y(g["ppl"])) for g in lineage}
    rad = {g["id"]: 3.5 + 5.5 * (g["params_m"] / pmax) ** 0.5 for g in lineage}

    edges = []
    for g in lineage:
        cx, cy = pos[g["id"]]
        for pid in g["parents"]:
            if pid in pos:
                px, py = pos[pid]
                mx = (px + cx) / 2
                edges.append(f'<path d="M {px:.1f} {py:.1f} Q {mx:.1f} {py:.1f} '
                             f'{cx:.1f} {cy:.1f}" fill="none" stroke="var(--edge)" '
                             f'stroke-width="1" opacity="0.5"/>')

    nodes = []
    for g in lineage:
        cx, cy = pos[g["id"]]
        fill = _ramp((math.log(g["ppl"]) - llo) / span, FIT_STOPS)
        r = rad[g["id"]]
        if g["id"] == best_id:
            nodes.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r + 4:.1f}" '
                         f'fill="none" stroke="#d4a017" stroke-width="2.5"/>')
        nodes.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
                     f'stroke="var(--nodeStroke)" stroke-width="0.8">'
                     f'<title>{g["id"]} gen{g["gen"]} · ppl {g["ppl"]} · '
                     f'{g["cfg"]["d_model"]}d×{g["cfg"]["n_layers"]}L · '
                     f'{g["params_m"]}M · {g["origin"]}</title></circle>')

    # generation ticks + fitness axis endpoints
    ticks = "".join(
        f'<text x="{X(gn):.1f}" y="{H - PADB + 18}" text-anchor="middle" '
        f'class="tick">gen {gn}</text>' for gn in range(gens + 1))
    yaxis = (f'<text x="{PADX - 10}" y="{Y(lo) + 4:.1f}" text-anchor="end" '
             f'class="tick">{lo:.0f}</text>'
             f'<text x="{PADX - 10}" y="{Y(hi) + 4:.1f}" text-anchor="end" '
             f'class="tick">{hi:.0f}</text>'
             f'<text x="18" y="{(PADT + H - PADB) / 2:.1f}" class="axis" '
             f'transform="rotate(-90 18 {(PADT + H - PADB) / 2:.1f})" '
             f'text-anchor="middle">val ppl · lower is fitter →</text>')
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" class="plot">'
            f'{"".join(edges)}{"".join(nodes)}{ticks}{yaxis}</svg>')


# ────────────────────────── config-space panel ──────────────────────────

def _configspace_svg(lineage: list[dict], gens: int, W: int, H: int) -> str:
    PADX, PADT, PADB = 60, 20, 46

    def gx(dm: int, jitter: float) -> float:
        i = D_MODELS.index(dm) if dm in D_MODELS else 0
        return PADX + (W - 2 * PADX) * (i + jitter) / (len(D_MODELS) - 1 + 0.6)

    def gy(nl: int, jitter: float) -> float:
        i = N_LAYERS.index(nl) if nl in N_LAYERS else 0
        return (H - PADB) - (H - PADT - PADB) * (i + jitter) / (len(N_LAYERS) - 1 + 0.6)

    dots = []
    for g in lineage:
        c = g["cfg"]
        # deterministic jitter from the non-plotted genes so collisions separate
        jx = ((c["n_heads"] % 4) / 4.0 - 0.375) * 0.5
        jy = ((c["n_kv_heads"] % 3) / 3.0 - 0.33) * 0.5
        x, y = gx(c["d_model"], jx), gy(c["n_layers"], jy)
        fill = _ramp(g["gen"] / gens if gens else 0.0, GEN_STOPS)
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{fill}" '
                    f'fill-opacity="0.85" stroke="var(--nodeStroke)" stroke-width="0.6">'
                    f'<title>gen{g["gen"]} · {c["d_model"]}d×{c["n_layers"]}L · '
                    f'ppl {g["ppl"]}</title></circle>')

    xt = "".join(f'<text x="{gx(dm, 0):.1f}" y="{H - PADB + 18}" text-anchor="middle" '
                 f'class="tick">{dm}</text>' for dm in D_MODELS)
    yt = "".join(f'<text x="{PADX - 10}" y="{gy(nl, 0) + 4:.1f}" text-anchor="end" '
                 f'class="tick">{nl}</text>' for nl in N_LAYERS)
    labels = (f'<text x="{W / 2}" y="{H - 8}" text-anchor="middle" class="axis">'
              f'd_model (embedding width) →</text>'
              f'<text x="16" y="{(PADT + H - PADB) / 2:.1f}" class="axis" '
              f'transform="rotate(-90 16 {(PADT + H - PADB) / 2:.1f})" '
              f'text-anchor="middle">n_layers (depth) →</text>')
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" class="plot">'
            f'{"".join(dots)}{xt}{yt}{labels}</svg>')


def _gen_legend(gens: int) -> str:
    chips = "".join(
        f'<span class="chip"><i style="background:{_ramp(g / gens if gens else 0, GEN_STOPS)}">'
        f'</i>gen {g}</span>' for g in range(gens + 1))
    return f'<div class="legend">{chips}</div>'


# ────────────────────────── page ──────────────────────────

def render(report: dict) -> str:
    run = report["runs"][0]                 # visualize the first seed
    lineage = [g for g in run["lineage"] if math.isfinite(g["ppl"])]
    best = run["best"]
    gens = max(g["gen"] for g in lineage)
    bc = best["cfg"]

    # per-generation best, to state the improvement plainly
    by_gen: dict[int, float] = {}
    for g in lineage:
        by_gen[g["gen"]] = min(by_gen.get(g["gen"], math.inf), g["ppl"])
    gen0_best = by_gen.get(0, math.inf)
    improve = (gen0_best - best["ppl"]) / gen0_best * 100 if math.isfinite(gen0_best) else 0

    geneal = _genealogy_svg(lineage, best["id"], 720, 430)
    cfgspace = _configspace_svg(lineage, gens, 720, 400)

    tiles = [
        (f'{best["ppl"]:.1f}', "best val ppl found", f'gen 0 best was {gen0_best:.1f}'),
        (f'−{improve:.0f}%', "ppl vs generation-0 best", "selection pressure at work"),
        (f'{gens + 1}', "generations bred", f'{len(lineage)} genomes evaluated'),
        (f'{bc["d_model"]}d×{bc["n_layers"]}L',
         "winning architecture", f'{best["params_m"]}M params, {bc["n_heads"]}h/{bc["n_kv_heads"]}kv'),
    ]
    tilehtml = "".join(
        f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div>'
        f'<div class="s">{s}</div></div>' for v, k, s in tiles)

    fit_legend = (
        '<div class="legend"><span class="chip">'
        f'<i style="background:{_hex(FIT_STOPS[0])}"></i>fitter</span>'
        f'<span class="chip"><i style="background:{_hex(FIT_STOPS[1])}"></i>mid</span>'
        f'<span class="chip"><i style="background:{_hex(FIT_STOPS[2])}"></i>weaker</span>'
        '<span class="chip"><i style="background:#d4a017"></i>best</span>'
        '<span class="chip note">node size ∝ params</span></div>')

    return f"""<!doctype html>
<meta charset="utf-8"><title>Evolutionary architecture search — genealogy</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    --bg:#fafafa; --fg:#1a1a1a; --muted:#5f6368; --card:#fff; --border:#e3e3e6;
    --edge:#b0b3b8; --nodeStroke:#fff; --accent:#ea8600;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#151517; --fg:#e8e8ea; --muted:#9aa0a6; --card:#1f1f22;
      --border:#2c2c30; --edge:#4a4a50; --nodeStroke:#1f1f22; }}
  }}
  :root[data-theme="light"] {{ --bg:#fafafa; --fg:#1a1a1a; --muted:#5f6368;
    --card:#fff; --border:#e3e3e6; --edge:#b0b3b8; --nodeStroke:#fff; }}
  :root[data-theme="dark"] {{ --bg:#151517; --fg:#e8e8ea; --muted:#9aa0a6;
    --card:#1f1f22; --border:#2c2c30; --edge:#4a4a50; --nodeStroke:#1f1f22; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
    font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
  .wrap {{ max-width:820px; margin:0 auto; padding:32px 22px 60px; }}
  h1 {{ font-size:25px; margin:0 0 6px; letter-spacing:-0.02em; }}
  h2 {{ font-size:18px; margin:34px 0 4px; letter-spacing:-0.01em; }}
  .sub {{ color:var(--muted); margin:0 0 22px; }}
  p.d {{ color:var(--muted); font-size:13.5px; margin:2px 0 12px; }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:12px; margin:20px 0 8px; }}
  .tile {{ background:var(--card); border:1px solid var(--border);
    border-radius:12px; padding:14px 16px; }}
  .tile .v {{ font-size:23px; font-weight:700; color:var(--accent);
    letter-spacing:-0.02em; }}
  .tile .k {{ font-size:13px; margin-top:2px; }}
  .tile .s {{ font-size:11.5px; color:var(--muted); margin-top:3px; }}
  .card {{ background:var(--card); border:1px solid var(--border);
    border-radius:14px; padding:16px 16px 10px; margin-top:10px; overflow-x:auto; }}
  .plot {{ display:block; }}
  .plot .tick {{ fill:var(--muted); font-size:11px; }}
  .plot .axis {{ fill:var(--muted); font-size:12px; font-weight:500; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:6px 14px; margin:8px 2px 2px;
    font-size:12px; color:var(--muted); }}
  .chip {{ display:inline-flex; align-items:center; gap:5px; }}
  .chip i {{ width:11px; height:11px; border-radius:3px; display:inline-block; }}
  .chip.note {{ font-style:italic; }}
  code {{ background:var(--card); border:1px solid var(--border);
    border-radius:5px; padding:0 5px; font-size:12.5px; }}
  .foot {{ color:var(--muted); font-size:12.5px; margin-top:30px; }}
</style>
<div class="wrap">
  <h1>Evolutionary architecture search</h1>
  <p class="sub">A real run of the <code>evolve</code> arm on a 4-core CPU
  container: {len(lineage)} genomes over {gens + 1} generations, each scored by
  the same deterministic val-ppl every baseline arm uses. Crossover + mutation
  of the winners, offspring pre-screened by the CheapPrior surrogate.</p>

  <div class="tiles">{tilehtml}</div>

  <h2>Genealogy — who bred from whom, and did it help</h2>
  <p class="d">Each node is one evaluated architecture; edges run parent →
  child. Height is fitness (best at the top), so a downward edge is a child
  that beat its parent. Weak lineages stop having children; the winning genes
  fan forward. Gold ring = best genome found.</p>
  <div class="card">{geneal}</div>
  {fit_legend}

  <h2>Config-space — the shape of the search</h2>
  <p class="d">The same genomes placed at their architecture, coloured by
  generation. Generation 0 is scattered; later generations concentrate toward
  the region the winners occupy. That drift is selection exploring, then
  exploiting — what a single regret number can't show.</p>
  <div class="card">{cfgspace}</div>
  {_gen_legend(gens)}

  <p class="foot">Generated by <code>evolve_viz.py</code> from
  <code>evolve_report.json</code> · pop {run["pop"]}, elite {run["elite"]},
  prior-oversample {run["oversample"]} · {report["full_steps"]} steps/eval,
  {report["budget_steps"]}-step budget · deterministic {report["val_batches"]}-window eval.</p>
</div>
<script>
  // honour an explicit theme toggle if the host stamps data-theme
  (function(){{ try {{ var t = document.documentElement.getAttribute('data-theme');
    if (t) document.documentElement.setAttribute('data-theme', t); }} catch(e){{}} }})();
</script>
"""


def build(json_path: str = REPORT_JSON, html_path: str = REPORT_HTML) -> str:
    with open(json_path) as f:
        report = json.load(f)
    html = render(report)
    with open(html_path, "w") as f:
        f.write(html)
    print(f"evolve_viz: wrote {html_path} "
          f"({sum(len(r['lineage']) for r in report['runs'])} genomes)")
    return html_path


if __name__ == "__main__":
    build()
