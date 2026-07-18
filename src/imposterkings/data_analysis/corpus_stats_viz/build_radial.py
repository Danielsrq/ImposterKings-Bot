"""Render the hand-combo winrates as a radial diverging dial -> a self-contained HTML page.

    python -m imposterkings.data_analysis.corpus_stats_viz.build_radial \
        --stats results/corpus_stats/stats.json --corpus "selfplay_gen1_v3c+selfplay_v3c_k50"

Why radial: a flat bar axis runs out of room fast; a circle bounds the layout and lets radius carry
meaning. Each curated combo is an angular spoke on a central ANNULUS (the "combo axis"). From the annulus
we draw two diverging bars:

    P2 winrate  -> radially OUTWARD   (the seat that moves second, ~59% base)
    P1 winrate  -> radially INWARD    (the seat that moves first,  ~41% base)

Bar length is the raw winrate (0 at the annulus edge -- the faithful version). A dashed reference ring on
each side marks that SEAT'S average winrate, so a spoke that crosses its ring beat the average for its seat.
Arc WIDTH encodes sample size as sqrt(games) -- raw counts would make 2-card combos ~8x fatter than 4-card
ones (pure combinatorics); sqrt compresses that to ~3x. Spokes are ordered by pooled winrate, so the dial
sweeps worst -> best as a spiral.

⚠️ Hand composition is CAUSAL (the deal precedes every decision), so these winrates are a real property of
the hand -- unlike the ability-usage stats, which are decisions made from a position and must not be read as
advice. This page only shows hand combos, so the framing is honest.

This is v1: dependency-free SVG so it opens anywhere. D3's scales/transitions come in with the interactive
pass. The generator re-runs on any corpus in stats.json (e.g. the gen-2.5 self-play corpus).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

# each card -> a 2-letter code for compact labels
_CODE = {"Queen": "Qu", "Princess": "Pr", "KingsHand": "KH", "Sentry": "Se", "Warlord": "Wa",
         "Mystic": "My", "Oathbound": "Oa", "Soldier": "So", "Judge": "Ju", "Inquisitor": "In",
         "Zealot": "Ze", "Elder": "El", "Assassin": "As", "Fool": "Fo"}


def _code(combo: str) -> str:
    return "+".join(_CODE.get(c, c[:2]) for c in combo.split("+"))


def curate(combos: List[dict], min_n: int, keep: int) -> List[dict]:
    """Sizes 2-4 with enough samples, then ~`keep` spokes sampled EVENLY across the winrate rank so the
    dial spans the full worst->best range (not just the extremes). Min and max are always kept."""
    pool = sorted((c for c in combos if 2 <= c["size"] <= 4 and c["hands"] >= min_n),
                  key=lambda c: c["winrate"])
    if len(pool) <= keep:
        chosen = pool
    else:
        idx = sorted({round(i * (len(pool) - 1) / (keep - 1)) for i in range(keep)})
        chosen = [pool[i] for i in idx]
    return chosen


# The curated narrative: the hands worth pointing at, over the ambient survey.
_FAMILY = {"Princess", "Queen", "Elder"}         # royalty + the anchor it fights against
# Explicitly named hands (fetched by name -> size-1 singles are allowed, unlike the survey's 2-4 filter).
_NAMED = ["Assassin", "KingsHand", "Assassin+KingsHand", "Inquisitor+Inquisitor"]
# Spokes to SPLIT into a clean partition (base -> [base with splitter | base without]). The pair is a
# subset of the single, so 'without' = single MINUS pair by inclusion-exclusion; the counts add exactly.
_PARTITION = [("Assassin", "KingsHand"), ("KingsHand", "Assassin")]


def _part_split(single: dict, pair: dict, splitter: str):
    """single -> two disjoint parts [with splitter (=pair), without splitter (=single-pair)], per seat."""
    def rate(sup, sub, seat):
        n = sup[f"{seat}_n"] - (sub[f"{seat}_n"] if sub else 0)
        w = sup[f"{seat}_wr"] * sup[f"{seat}_n"] - (sub[f"{seat}_wr"] * sub[f"{seat}_n"] if sub else 0)
        return (w / n if n else 0.0, n)
    code = _CODE.get(splitter, splitter[:2])
    parts = []
    for label, sup, sub in ((f"+{code}", pair, None), (f"-{code}", single, pair)):   # +KH / -KH parts
        p1, n1 = rate(sup, sub, "first")
        p2, n2 = rate(sup, sub, "second")
        parts.append({"label": label, "p1": round(p1, 4), "p2": round(p2, 4), "n": n1 + n2})
    return parts


def story(combos: List[dict], min_n: int):
    """The named hands: the best & worst well-powered 2-4 combos, the {Princess,Queen,Elder} family (Elder
    is a pair, so the double-Elder ones are included), and the explicitly-named hands in _NAMED (which may
    be size-1 singles). Returns an ORDERED (row, tag) list, tag in {best, worst, story}."""
    by = {c["combo"]: c for c in combos}
    pool = [c for c in combos if 2 <= c["size"] <= 4 and c["hands"] >= min_n]
    hi = max(pool, key=lambda c: c["winrate"])
    lo = min(pool, key=lambda c: c["winrate"])
    fam = sorted((c for c in combos if 2 <= c["size"] <= 4
                  and set(c["combo"].split("+")) <= _FAMILY),
                 key=lambda c: (c["size"], -c["winrate"]))
    ordered, seen = [], set()

    def add(c, tag):
        if c and c["combo"] not in seen:
            seen.add(c["combo"])
            ordered.append((c, tag))
    add(hi, "best")
    add(lo, "worst")
    for c in fam:
        add(c, "story")
    for name in _NAMED:
        add(by.get(name), "story")
    return ordered


def build(stats_path: str, corpus: str, out_path: str, min_n: int, keep: int) -> str:
    doc = json.load(open(stats_path, encoding="utf-8"))
    if corpus not in doc:
        raise SystemExit(f"corpus {corpus!r} not in {stats_path}. Available: {list(doc)}")
    tables = doc[corpus]["tables"]
    games = doc[corpus]["games"]

    # seat baselines from the turn-order table (pooled over agents present)
    tn = tables["turn_order"]
    nf = sum(r["games"] for r in tn if r["went"] == "first")
    wf = sum(round(r["winrate"] * r["games"]) for r in tn if r["went"] == "first")
    ns = sum(r["games"] for r in tn if r["went"] == "second")
    ws = sum(round(r["winrate"] * r["games"]) for r in tn if r["went"] == "second")
    base_p1 = wf / nf if nf else 0.41
    base_p2 = ws / ns if ns else 0.59

    def row(c, tag=None):
        return {"name": c["combo"], "code": _code(c["combo"]), "size": c["size"], "n": c["hands"],
                "p1": round(c["first_wr"], 4), "p2": round(c["second_wr"], 4),
                "wr": round(c["winrate"], 4), "tag": tag}

    render: Dict[str, dict] = {c["combo"]: row(c) for c in curate(tables["hand_combos"], min_n, keep)}
    tale = story(tables["hand_combos"], min_n)
    for c, tag in tale:                       # story hands are forced in and win the tag
        render[c["combo"]] = row(c, tag)

    by = {c["combo"]: c for c in tables["hand_combos"]}
    for base, splitter in _PARTITION:         # split chosen spokes into a clean [with | without] partition
        pair = by.get("+".join(sorted([base, splitter])))
        if base in render and pair:
            render[base]["parts"] = _part_split(by[base], pair, splitter)

    data = list(render.values())
    story_list = [row(c, tag) for c, tag in tale]      # ordered, for the side panel

    meta = {"corpus": corpus, "games": games, "combos": len(data), "min_n": min_n,
            "base_p1": round(base_p1, 4), "base_p2": round(base_p2, 4)}

    html = (_TEMPLATE.replace("/*__DATA__*/", json.dumps(data))
            .replace("/*__STORY__*/", json.dumps(story_list))
            .replace("/*__META__*/", json.dumps(meta)))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {out_path}  ({len(data)} combos from {corpus}, {games} games, "
          f"P1 base {base_p1:.1%} / P2 base {base_p2:.1%})")
    return out_path


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ImposterKings — Hand-Combo Dial</title>
<style>
  :root{
    --ground:#15131b; --panel:#201c28; --ink:#ece6d8; --muted:#8b8475;
    --gold:#d4b24a; --gold-dim:rgba(212,178,74,.14); --gold-line:rgba(212,178,74,.28);
    --p1:#5f93cc; --p2:#e0873a; --p1-soft:rgba(95,147,204,.9); --p2-soft:rgba(224,135,58,.92);
    --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
    --mono:ui-monospace,"SF Mono","Cascadia Mono",Menlo,monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{
    background:
      radial-gradient(120% 90% at 50% 8%, #1c1826 0%, var(--ground) 60%) fixed,
      var(--ground);
    color:var(--ink); font-family:var(--sans);
    display:flex; flex-direction:column; min-height:100%;
  }
  header{
    padding:20px 26px 8px; display:flex; flex-wrap:wrap; align-items:baseline;
    gap:6px 22px; border-bottom:1px solid var(--gold-dim);
  }
  h1{
    font-family:var(--serif); font-weight:600; font-size:clamp(20px,2.4vw,30px);
    letter-spacing:.01em; margin:0; text-wrap:balance;
  }
  h1 .amp{color:var(--gold)}
  .sub{color:var(--muted); font-size:13px; letter-spacing:.02em}
  .sub b{color:var(--ink); font-weight:600; font-variant-numeric:tabular-nums}
  .modes{margin-left:auto; display:inline-flex; border:1px solid var(--gold-dim); border-radius:8px;
    overflow:hidden}
  .mode{appearance:none; border:0; background:transparent; color:var(--muted); cursor:pointer;
    font-family:var(--sans); font-size:12.5px; padding:6px 13px; letter-spacing:.02em;
    transition:background .12s ease, color .12s ease}
  .mode:hover{color:var(--ink)}
  .mode.on{background:var(--gold); color:#1a160e; font-weight:600}
  .mode:focus-visible{outline:2px solid var(--gold); outline-offset:-2px}
  .legend{display:flex; gap:16px; align-items:center; font-size:12.5px}
  .key{display:flex; align-items:center; gap:7px; color:var(--muted)}
  .dot{width:11px; height:11px; border-radius:2px}
  .key b{color:var(--ink); font-weight:600}
  main{flex:1; display:grid; place-items:center; padding:8px 12px 4px; min-height:0}
  .stage{position:relative; width:min(92vmin,860px); aspect-ratio:1}
  svg{width:100%; height:100%; overflow:visible; display:block}
  .spoke{cursor:pointer}
  .spoke path{transition:opacity .12s ease, filter .12s ease}
  .stage.hot .spoke:not(.on) path{opacity:.28}
  .p2{fill:var(--p2-soft)} .p1{fill:var(--p1-soft)}     /* base bar colours (overridden by .faint/.below) */
  .spoke.on .p2{filter:drop-shadow(0 0 6px rgba(224,135,58,.55))}
  .spoke.on .p1{filter:drop-shadow(0 0 6px rgba(95,147,204,.55))}
  .ring{fill:none}
  .ring.grid{stroke:var(--gold-dim); stroke-width:1}
  .ring.mean-p2{stroke:var(--p2); stroke-width:2.4; stroke-dasharray:9 5; opacity:1;
    filter:drop-shadow(0 0 3px rgba(224,135,58,.5))}
  .ring.mean-p1{stroke:var(--p1); stroke-width:2.4; stroke-dasharray:9 5; opacity:1;
    filter:drop-shadow(0 0 3px rgba(95,147,204,.5))}
  .ring-label{font-family:var(--mono); font-size:13px; font-weight:600;
    paint-order:stroke; stroke:var(--ground); stroke-width:4px; stroke-linejoin:round}
  .tick{fill:var(--muted); font-family:var(--mono); font-size:10px; opacity:.6}
  /* in LIFT mode the annulus-referenced value grid is meaningless -> fade it, mean rings become "zero" */
  .lift .ring.grid, .lift .tick{opacity:0}
  .lift .ring.mean-p2, .lift .ring.mean-p1{stroke-dasharray:none; stroke-width:2}
  /* below-average bars (lift mode) desaturate so the sign reads even before you clock the direction */
  .p2.below{fill:rgba(224,135,58,.34)} .p1.below{fill:rgba(95,147,204,.34)}
  /* a partition's "without" sub-arc is dimmer so the "with" half (the real combo) reads as the driver */
  .p2.faint{fill:rgba(224,135,58,.46)} .p1.faint{fill:rgba(95,147,204,.46)}
  .slabel.part{font-size:11px}
  /* faint gold "territory" behind a partitioned spoke, so its two sub-arcs read as ONE grouped combo */
  .group{fill:rgba(212,178,74,.07); stroke:var(--gold-line); stroke-width:1; pointer-events:none}
  /* hollow-centre live readout */
  .readout{
    position:absolute; inset:0; display:grid; place-content:center; text-align:center;
    pointer-events:none; gap:3px;
  }
  .r-combo{font-family:var(--serif); font-size:clamp(15px,2vw,21px); line-height:1.15;
    max-width:44%; margin:0 auto; text-wrap:balance}
  .r-sub{color:var(--muted); font-size:11.5px; letter-spacing:.04em; text-transform:uppercase}
  .r-nums{display:flex; gap:20px; justify-content:center; margin-top:6px; font-variant-numeric:tabular-nums}
  .r-nums div{display:flex; flex-direction:column; gap:1px}
  .r-nums .v{font-family:var(--mono); font-size:20px; font-weight:600}
  .r-nums .l{font-size:10px; color:var(--muted); letter-spacing:.06em}
  .r-nums .p1 .v{color:var(--p1)} .r-nums .p2 .v{color:var(--p2)}
  .r-idle .r-nums{opacity:.35}
  /* story layer: gold annulus ticks + always-on spoke codes */
  .marker{stroke:var(--gold); stroke-width:2.5; opacity:.9}
  .slabel{font-family:var(--mono); font-size:12px; font-weight:600; fill:var(--gold);
    paint-order:stroke; stroke:var(--ground); stroke-width:3.5px; stroke-linejoin:round}
  .spoke.tag-best .slabel, .spoke.tag-worst .slabel{font-size:13px; fill:#f3d067}
  /* the curated-story side panel */
  .layout{display:flex; gap:clamp(12px,3vw,40px); align-items:center; flex-wrap:wrap; justify-content:center}
  .panel{width:min(340px,88vw); display:flex; flex-direction:column; gap:2px;
    font-size:12.5px}
  .panel h2{font-family:var(--serif); font-weight:600; font-size:16px; margin:0 0 6px;
    color:var(--ink); letter-spacing:.01em}
  .panel .eyebrow{font-size:10.5px; letter-spacing:.14em; text-transform:uppercase; color:var(--gold);
    margin-bottom:2px}
  .srow{display:grid; grid-template-columns:1fr auto auto; align-items:center; gap:10px;
    padding:6px 9px; border-radius:6px; cursor:pointer; border:1px solid transparent;
    transition:background .12s ease, border-color .12s ease}
  .srow:hover, .srow.on{background:#241f2c; border-color:var(--gold-dim)}
  .srow .nm{color:var(--ink); line-height:1.2}
  .srow .nm small{color:var(--muted); font-size:10.5px}
  .srow .wr{font-family:var(--mono); font-variant-numeric:tabular-nums; font-size:13px; text-align:right;
    min-width:38px}
  .srow .wr.p1{color:var(--p1)} .srow .wr.p2{color:var(--p2)}
  .srow .badge{grid-column:1/-1; font-size:9.5px; letter-spacing:.12em; text-transform:uppercase;
    color:var(--gold)}
  .srow.tag-best .nm, .srow.tag-worst .nm{font-weight:600}
  footer{padding:8px 26px 16px; color:var(--muted); font-size:11.5px; letter-spacing:.01em;
    border-top:1px solid var(--gold-dim); text-align:center}
  footer b{color:var(--gold)}
</style>
</head>
<body>
<header>
  <h1>Imposter<span class="amp">·</span>Kings — the hand dial</h1>
  <div class="sub">what a starting hand is worth · <b id="m-corpus"></b> · <b id="m-games"></b> games</div>
  <div class="modes" role="group" aria-label="bar mode">
    <button class="mode on" data-mode="raw">Win&nbsp;%</button>
    <button class="mode" data-mode="lift">Lift vs avg</button>
  </div>
  <div class="legend">
    <span class="key"><span class="dot" style="background:var(--p1)"></span><b>P1</b> moves first ↓ in</span>
    <span class="key"><span class="dot" style="background:var(--p2)"></span><b>P2</b> moves second ↑ out</span>
    <span class="key"><span class="dot" style="background:var(--gold);opacity:.7"></span>seat average</span>
  </div>
</header>
<main>
  <div class="layout">
    <div class="stage r-idle" id="stage">
      <svg id="dial" viewBox="-600 -600 1200 1200" role="img" aria-label="Radial hand-combo winrate dial"></svg>
      <div class="readout" id="readout">
        <div class="r-sub">hover a spoke</div>
        <div class="r-combo" id="r-combo">the hand dial</div>
        <div class="r-nums">
          <div class="p1"><span class="v" id="r-p1">—</span><span class="l">P1 win%</span></div>
          <div class="p2"><span class="v" id="r-p2">—</span><span class="l">P2 win%</span></div>
          <div><span class="v" id="r-n" style="color:var(--muted)">—</span><span class="l">games</span></div>
        </div>
      </div>
    </div>
    <aside class="panel" id="panel">
      <div class="eyebrow">Hands worth naming</div>
      <h2>The curated few</h2>
    </aside>
  </div>
</main>
<footer>
  bar length = raw win% from the annulus · dashed ring = that seat's average
  (<b><span id="f-p1"></span></b> / <b><span id="f-p2"></span></b>) · arc width = √games ·
  hand strength is <b>causal</b> — the deal precedes every choice
</footer>

<script>
const DATA = /*__DATA__*/;
const STORY = /*__STORY__*/;
const META = /*__META__*/;

// ---- geometry -----------------------------------------------------------------------------------
// EQUAL scale for both seats (S). The space check the design predicted: with equal S the longest INWARD
// (P1) bar nearly reaches the centre, so the annulus R0 is placed just far enough out that even the
// longest P1 bar clears a fixed HUB (the hollow-centre readout). Everything self-sizes from the data, so
// the same page renders any corpus.
const T = 22, S = 300, HUB = 150, PAD = 96;   // annulus half-thickness, win% scale (px/1.0), centre hub
                                              // (PAD leaves room for story code-labels past the P2 bars)
const GAP = 0.006;                            // angular gap between spokes (radians)
const TAU = Math.PI * 2, TOP = -Math.PI/2;

const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs={}) => { const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]); return n; };
const P = (r, a) => [r*Math.cos(a), r*Math.sin(a)];

// annular wedge between radii [ri, ro] over angles [a0, a1]
function wedge(ri, ro, a0, a1){
  const [x0o,y0o]=P(ro,a0),[x1o,y1o]=P(ro,a1),[x1i,y1i]=P(ri,a1),[x0i,y0i]=P(ri,a0);
  return `M${x0o} ${y0o}A${ro} ${ro} 0 0 1 ${x1o} ${y1o}L${x1i} ${y1i}A${ri} ${ri} 0 0 0 ${x0i} ${y0i}Z`;
}

const svg = document.getElementById("dial");
const stage = document.getElementById("stage");

// order by pooled winrate -> the spiral; angular width by sqrt(games)
const rows = DATA.slice().sort((a,b)=>a.wr-b.wr);
const wsum = rows.reduce((s,r)=>s+Math.sqrt(r.n),0);
let ang = TOP;
const spokes = rows.map(r=>{ const span = TAU*Math.sqrt(r.n)/wsum;
  const o={...r, a0:ang+GAP/2, a1:ang+span-GAP/2, amid:ang+span/2}; ang+=span; return o; });

// self-size: R0 far enough out that the longest INWARD (P1) bar still clears the hub; viewBox from the
// longest OUTWARD (P2) bar. Equal S throughout -- the annulus just sits where both sides fit.
const maxP1 = Math.max(...rows.map(r=>r.p1)), maxP2 = Math.max(...rows.map(r=>r.p2));
const R0 = HUB + T + maxP1 * S;
const VB = R0 + T + maxP2 * S + PAD;
svg.setAttribute("viewBox", `${-VB} ${-VB} ${2*VB} ${2*VB}`);

// ---- reference geometry: grid arcs + seat-average rings + value ticks ---------------------------
const gGrid = el("g"); svg.appendChild(gGrid);
[0.25,0.5,0.75].forEach(v=>{
  gGrid.appendChild(el("circle",{class:"ring grid", r:R0+T+v*S}));   // outward (P2) value grid
  gGrid.appendChild(el("circle",{class:"ring grid", r:R0-T-v*S}));   // inward  (P1) value grid
  const to=el("text",{class:"tick", x:0, y:-(R0+T+v*S), "text-anchor":"middle"}); to.textContent=(v*100)+"%"; gGrid.appendChild(to);
  const ti=el("text",{class:"tick", x:0, y:-(R0-T-v*S), "text-anchor":"middle", dy:"0.9em"}); ti.textContent=(v*100)+"%"; gGrid.appendChild(ti);
});
const meanP2r = R0+T+META.base_p2*S, meanP1r = R0-T-META.base_p1*S;
gGrid.appendChild(el("circle",{class:"ring mean-p2", r:meanP2r}));
gGrid.appendChild(el("circle",{class:"ring mean-p1", r:meanP1r}));
// label each seat-average ring at 12 o'clock, with a dark halo so it reads over any spoke
function ringLabel(radius, text, fill, dy){
  const t=el("text",{class:"ring-label", x:0, y:-radius, "text-anchor":"middle", dy, fill});
  t.textContent=text; return t;
}
gGrid.appendChild(ringLabel(meanP2r, `P2 avg ${(META.base_p2*100).toFixed(0)}%`, "var(--p2)", "-0.5em"));
gGrid.appendChild(ringLabel(meanP1r, `P1 avg ${(META.base_p1*100).toFixed(0)}%`, "var(--p1)", "1.1em"));

// annulus band (the combo axis)
svg.appendChild(el("circle",{r:R0, fill:"none", stroke:"var(--panel)", "stroke-width":2*T}));
svg.appendChild(el("circle",{r:R0+T, class:"ring", stroke:"var(--gold-line)", "stroke-width":1}));
svg.appendChild(el("circle",{r:R0-T, class:"ring", stroke:"var(--gold-line)", "stroke-width":1}));

// ---- spokes + the story layer -------------------------------------------------------------------
const gSpokes = el("g"); svg.appendChild(gSpokes);
const readout = { combo:document.getElementById("r-combo"), p1:document.getElementById("r-p1"),
  p2:document.getElementById("r-p2"), n:document.getElementById("r-n"),
  sub:document.querySelector("#readout .r-sub"),
  p1l:document.querySelector(".r-nums .p1 .l"), p2l:document.querySelector(".r-nums .p2 .l") };
const spokeEl = {}, rowEl = {};                 // name -> element, for two-way highlight

// a story spoke gets a gold tick across the annulus + an always-on code label past its P2 bar
function radialLabel(a, r, text, cls){
  const [x,y] = P(r,a); let deg = a*180/Math.PI, anchor = "start";
  if (Math.cos(a) < 0){ deg += 180; anchor = "end"; }        // left half: flip upright
  const t = el("text",{class:cls, x, y, "text-anchor":anchor, "dominant-baseline":"middle",
    transform:`rotate(${deg} ${x} ${y})`}); t.textContent = text; return t;
}

const SUBGAP = 0.004;                          // tiny gap between a partition's two sub-arcs
for(const s of spokes){
  const g = el("g",{class:"spoke"+(s.tag?` tag-${s.tag}`:"")}); g.dataset.name=s.name;
  // segments: a partitioned spoke splits its arc by COUNT into [with | without]; else one full segment.
  if(s.parts){
    const ntot = s.parts.reduce((k,p)=>k+p.n,0);
    const mid = s.a0 + (s.a1-s.a0)*s.parts[0].n/ntot;
    s.segs = [{a0:s.a0, a1:mid-SUBGAP, amid:(s.a0+mid)/2, ...s.parts[0], name:s.name, faint:false},
              {a0:mid+SUBGAP, a1:s.a1, amid:(mid+s.a1)/2, ...s.parts[1], name:s.name, faint:true}];
  } else {
    s.segs = [{a0:s.a0, a1:s.a1, amid:s.amid, p1:s.p1, p2:s.p2, n:s.n, name:s.name, faint:false}];
  }
  if(s.parts){                                 // group backdrop spanning BOTH sub-arcs (drawn behind them)
    const mp2=Math.max(...s.parts.map(p=>p.p2)), mp1=Math.max(...s.parts.map(p=>p.p1));
    g.appendChild(el("path",{class:"group", d:wedge(R0-T-mp1*S, R0+T+mp2*S, s.a0, s.a1)}));
  }
  for(const seg of s.segs){
    seg._p2 = el("path",{class:"p2"+(seg.faint?" faint":"")});
    seg._p1 = el("path",{class:"p1"+(seg.faint?" faint":"")});
    g.appendChild(seg._p2); g.appendChild(seg._p1);
    seg._p2.addEventListener("pointerenter",()=>focusSeg(seg));
    seg._p1.addEventListener("pointerenter",()=>focusSeg(seg));
  }
  if(s.tag){                                   // story: gold annulus tick + code label(s) at the tip(s)
    const [xi,yi]=P(R0-T,s.amid),[xo,yo]=P(R0+T,s.amid);
    g.appendChild(el("line",{class:"marker", x1:xi,y1:yi,x2:xo,y2:yo}));
    if(s.parts) for(const seg of s.segs)       // each sub-arc: base code + its +/- splitter, e.g. "As+KH"
      g.appendChild(radialLabel(seg.amid, R0+T+seg.p2*S+12, s.code+seg.label, "slabel part"));
    else g.appendChild(radialLabel(s.amid, R0+T+s.p2*S+12, s.code, "slabel"));
  }
  gSpokes.appendChild(g); spokeEl[s.name]=g;
}

// RAW: bars grow from the annulus edge (0% base). LIFT: from the seat-average ring, so length is pp
// above/below average and direction shows the sign (the tip is the same point in both modes).
let MODE = "raw", refocus = null;
function drawBars(){
  const b2 = MODE==="lift" ? R0+T+META.base_p2*S : R0+T;
  const b1 = MODE==="lift" ? R0-T-META.base_p1*S : R0-T;
  for(const s of spokes) for(const seg of s.segs){
    const t2 = R0+T+seg.p2*S, t1 = R0-T-seg.p1*S;
    seg._p2.setAttribute("d", wedge(Math.min(b2,t2), Math.max(b2,t2), seg.a0, seg.a1));
    seg._p1.setAttribute("d", wedge(Math.min(b1,t1), Math.max(b1,t1), seg.a0, seg.a1));
    seg._p2.classList.toggle("below", MODE==="lift" && seg.p2 < META.base_p2);
    seg._p1.classList.toggle("below", MODE==="lift" && seg.p1 < META.base_p1);
  }
}
drawBars();

function paint(name){
  stage.classList.remove("r-idle"); stage.classList.add("hot");
  for(const n in spokeEl) spokeEl[n].classList.toggle("on", n===name);
  for(const n in rowEl) rowEl[n].classList.toggle("on", n===name);
}
function showNums(p1,p2,n){
  if(MODE==="lift"){
    const sg=v=>(v>=0?"+":"")+((v)*100).toFixed(0);
    readout.p1.textContent=sg(p1-META.base_p1); readout.p2.textContent=sg(p2-META.base_p2);
    readout.p1l.textContent="P1 pp vs avg"; readout.p2l.textContent="P2 pp vs avg";
  }else{
    readout.p1.textContent=(p1*100).toFixed(0); readout.p2.textContent=(p2*100).toFixed(0);
    readout.p1l.textContent="P1 win%"; readout.p2l.textContent="P2 win%";
  }
  readout.n.textContent=n.toLocaleString();
}
function focus(name){                           // panel hover -> the marginal combo
  const s=DATA.find(d=>d.name===name); if(!s) return;
  refocus=()=>focus(name); paint(name);
  readout.combo.textContent=name.replace(/\+/g," + ");
  readout.sub.textContent=name.split("+").length+"-card hand";
  showNums(s.p1,s.p2,s.n);
}
function focusSeg(seg){                          // dial hover -> this segment (a part, or a whole combo)
  refocus=()=>focusSeg(seg); paint(seg.name);
  readout.combo.textContent=seg.name.replace(/\+/g," + ")+(seg.label?("  "+seg.label):"");
  readout.sub.textContent=seg.label?"partition":(seg.name.split("+").length+"-card hand");
  showNums(seg.p1,seg.p2,seg.n);
}
function blur(){
  refocus=null;
  stage.classList.add("r-idle"); stage.classList.remove("hot");
  for(const n in spokeEl) spokeEl[n].classList.remove("on");
  for(const n in rowEl) rowEl[n].classList.remove("on");
  readout.sub.textContent="hover a spoke"; readout.combo.textContent="the hand dial";
  readout.p1.textContent="—"; readout.p2.textContent="—"; readout.n.textContent="—";
}
stage.addEventListener("pointerleave", blur);

document.querySelectorAll(".mode").forEach(btn=>{
  btn.addEventListener("click",()=>{
    MODE = btn.dataset.mode;
    document.querySelectorAll(".mode").forEach(b=>b.classList.toggle("on", b===btn));
    stage.classList.toggle("lift", MODE==="lift");
    drawBars();
    if(refocus) refocus();
  });
});

// ---- the curated-story panel --------------------------------------------------------------------
const panel = document.getElementById("panel");
const TAGNAME = {best:"Highest winrate", worst:"Lowest winrate"};
for(const s of STORY){
  const r = el2("div","srow"+(s.tag!=="story"?` tag-${s.tag}`:""));
  if(TAGNAME[s.tag]) r.appendChild(el2("div","badge", TAGNAME[s.tag]));
  const nm = el2("div","nm"); nm.innerHTML = s.name.replace(/\+/g," + ")
    + ` <small>${s.size}-card · n=${s.n.toLocaleString()}</small>`;
  r.appendChild(nm);
  r.appendChild(el2("div","wr p1", (s.p1*100).toFixed(0)+"%"));
  r.appendChild(el2("div","wr p2", (s.p2*100).toFixed(0)+"%"));
  r.addEventListener("pointerenter",()=>focus(s.name));
  r.addEventListener("pointerleave",blur);
  panel.appendChild(r); rowEl[s.name]=r;
}
function el2(tag, cls, text){ const n=document.createElement(tag); n.className=cls;
  if(text!=null) n.textContent=text; return n; }

// ---- chrome text ---------------------------------------------------------------------------------
document.getElementById("m-corpus").textContent = META.corpus.replace(/\+/g," + ");
document.getElementById("m-games").textContent = META.games.toLocaleString();
document.getElementById("f-p1").textContent = (META.base_p1*100).toFixed(1)+"%";
document.getElementById("f-p2").textContent = (META.base_p2*100).toFixed(1)+"%";
</script>
</body>
</html>
"""


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Radial hand-combo winrate dial -> self-contained HTML.")
    p.add_argument("--stats", default=os.path.join("results", "corpus_stats", "stats.json"))
    p.add_argument("--corpus", default="selfplay_gen1_v3c+selfplay_v3c_k50")
    p.add_argument("--out", default=os.path.join("src", "imposterkings", "data_analysis",
                                                 "corpus_stats_viz", "radial_combos.html"))
    p.add_argument("--min-n", type=int, default=400, help="drop combos with fewer games (noisy)")
    p.add_argument("--keep", type=int, default=40, help="how many spokes (evenly sampled across winrate)")
    args = p.parse_args(argv)
    build(args.stats, args.corpus, args.out, args.min_n, args.keep)


if __name__ == "__main__":
    main()
