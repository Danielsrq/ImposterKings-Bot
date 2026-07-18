"""The Honeycomb Mural -- every recorded hand-combo as one hexagon, stacked into columns by summed value.

    python -m imposterkings.data_analysis.corpus_stats_viz.build_honeycomb \
        --stats results/corpus_stats/selfplay_gen25/hand_combos.csv \
        --out reports/honeycomb_p2.html --seat second

WHY A HEX WALL. The radial dial ranks combos by winrate on an angular axis; it has no *quantitative* x. This
view trades the dial's paired P1/P2 bars and its sqrt(n) width encoding for a real x-axis: the SUMMED CARD
VALUE of the combo (Princess+Queen = 18, but so is Princess+Fool+Sentry). Summed value is a weak predictor --
every column from ~14 to 26 spreads 50-70pp in winrate -- so the wall's message is exactly that: *total value
barely matters, composition does.* You read it as towering vertical spread per column with a floor that slopes
gently up left-to-right (value buys a higher worst-case, not a higher ceiling).

The layout is a flat-top hexagonal column-stack: one column per distinct summed value, combos sorted by winrate
within the column and bottom-aligned into an arch (mid-value columns hold the most combos). Perfect tessellation
requires a lattice, so the y-position is winrate RANK within the column, not a shared axis -- absolute winrate is
carried by colour (a diverging ramp about the seat average) and the exact number by the hover readout.

Self-contained: the 14 card arts are downscaled and inlined as data URIs, drawn into each hexagon as vertical
stripes (Canvas, with level-of-detail: a colour field when zoomed out, card art when zoomed in). No external
dependencies -- CSP-safe for publishing as an artifact.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
from itertools import combinations
from typing import Dict, List

from ... import cards  # type: ignore

NAME_VAL: Dict[str, int] = {}
NAME_ASSET: Dict[str, str] = {}
for _cdef, _cnt in cards.DECK_SPEC:
    NAME_VAL[_cdef.name] = _cdef.value
    NAME_ASSET[_cdef.name] = _cdef.asset


def _art_data_uris(assets_dir: str, width: int) -> Dict[str, str]:
    """Downscale each of the 14 card arts to ``width`` px and return name -> data URI (JPEG)."""
    from PIL import Image

    out: Dict[str, str] = {}
    for name, fname in NAME_ASSET.items():
        path = os.path.join(assets_dir, fname)
        with Image.open(path) as im:
            im = im.convert("RGB")
            h = round(im.height * width / im.width)
            im = im.resize((width, h), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=72, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        out[name] = f"data:image/jpeg;base64,{b64}"
    return out


_WR_COL = {"second": "second_wr", "first": "first_wr", "pooled": "winrate"}
_N_COL = {"second": "second_n", "first": "first_n", "pooled": "hands"}

# extra "partition" tiles: a HELD set of cards with an EXCLUDED set, by inclusion-exclusion
#   n(A, none of B) = sum over T subset of B of (-1)^|T| * n(A + T).
# Each entry is (include_list, exclude_list); both may hold duplicates (e.g. two Elders). The only
# limit is the table's size-4 ceiling: |include| + |exclude| must be <= 4, or a required row is absent
# and the tile is skipped (with a printed note). The tile is placed at the include set's summed value.
_PARTITIONS = [
    (["Assassin"], ["KingsHand"]),
    (["KingsHand"], ["Assassin"]),
    (["Elder", "Elder"], ["Queen", "Princess"]),
    (["Queen", "Princess"], ["Elder", "Elder"]),
    (["Assassin", "KingsHand"], ["Queen", "Princess"]),
    (["Queen", "Princess"], ["Assassin", "KingsHand"])
]


def _combos(csv_path: str) -> List[dict]:
    """Read hand_combos.csv into [{cards, s, size, wr1, n1, wr2, n2}] -- BOTH seats carried, so the
    diverging (P2-up / P1-down) mural can render the same combo on both sides."""
    rows: List[dict] = []
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        n1 = int(r["first_n"]) if r["first_n"] else 0
        n2 = int(r["second_n"]) if r["second_n"] else 0
        if n1 == 0 or n2 == 0:                       # keep only combos present for BOTH seats (walls align)
            continue
        names = r["combo"].split("+")
        rows.append({
            "cards": names,
            "s": sum(NAME_VAL[c] for c in names),
            "size": int(r["size"]),
            "wr1": round(float(r["first_wr"]), 4), "n1": n1,
            "wr2": round(float(r["second_wr"]), 4), "n2": n2,
        })
    return rows


def _partition_tiles(csv_path: str) -> List[dict]:
    """Inclusion-exclusion tiles for BOTH seats: include set minus exclude set, at the include's value."""
    rows = {r["combo"]: r for r in csv.DictReader(open(csv_path, encoding="utf-8"))}

    def ie(include, exclude, wr_col, n_col):
        """n(include, none of exclude) and its winrate for one seat, or None if a row is missing."""
        n_tot = w_tot = 0.0
        for k in range(len(exclude) + 1):                 # sum over every subset T of the exclude set
            for T in combinations(exclude, k):
                r = rows.get("+".join(sorted(list(include) + list(T))))
                if not r or not r[n_col]:
                    return None
                ni, wi = int(r[n_col]), float(r[wr_col])
                w_tot += (-1 if k % 2 else 1) * wi * ni
                n_tot += (-1 if k % 2 else 1) * ni
        return (n_tot, w_tot / n_tot) if n_tot > 0 else None

    tiles: List[dict] = []
    for include, exclude in _PARTITIONS:
        # "not containing X" is binary per card type (the table counts 'contains at least'), so a
        # duplicate in the exclude set is meaningless -- dedupe it, else inclusion-exclusion double-counts.
        exclude = list(dict.fromkeys(exclude))
        r2 = ie(include, exclude, "second_wr", "second_n")
        r1 = ie(include, exclude, "first_wr", "first_n")
        if r1 is None or r2 is None:
            print(f"  partition skipped ({'+'.join(include)} \\ {'+'.join(exclude)}): missing/empty row")
            continue
        tiles.append({"cards": list(include), "excl": list(exclude),
                      "s": sum(NAME_VAL[c] for c in include), "size": len(include),
                      "wr1": round(r1[1], 4), "n1": int(round(r1[0])),
                      "wr2": round(r2[1], 4), "n2": int(round(r2[0]))})
    return tiles


_PAGE = r"""<title>The Honeycomb Mural &mdash; Imposter Kings</title>
<style>
  :root{
    --ground:#15131b; --ground2:#1d1a25; --panel:#221e2b; --line:#332d40;
    --gold:#d4b24a; --gold-line:#5a4c2a; --p2:#e0873a; --p1:#5f93cc; --ink:#efe8dd; --muted:#9b93a6;
    --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Times New Roman",serif;
    --sans:"Inter",system-ui,-apple-system,"Segoe UI",sans-serif;
    --mono:"SFMono-Regular",ui-monospace,"Cascadia Mono","Consolas",monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{background:radial-gradient(120% 90% at 50% -10%,#211c2b 0%,var(--ground) 60%);
    color:var(--ink);font-family:var(--sans);overflow:hidden}
  #wrap{position:fixed;inset:0;display:flex;flex-direction:column}
  header{padding:14px 20px 10px;display:flex;align-items:baseline;gap:18px;flex-wrap:wrap;
    border-bottom:1px solid var(--line);background:linear-gradient(180deg,rgba(0,0,0,.25),transparent);z-index:3}
  h1{margin:0;font-family:var(--serif);font-weight:600;font-size:22px;letter-spacing:.3px;
    text-wrap:balance;color:var(--ink)}
  h1 .amp{color:var(--gold);font-style:italic;padding:0 2px}
  .sub{color:var(--muted);font-size:12.5px;max-width:60ch;line-height:1.45}
  .sub b{color:var(--p2)}
  .legend{margin-left:auto;display:flex;align-items:center;gap:14px;font-size:11.5px;color:var(--muted)}
  .key{display:flex;align-items:center;gap:5px}
  .sw{width:11px;height:11px;border-radius:3px;display:inline-block}
  .sub .p2t{color:var(--p2)} .sub .p1t{color:var(--p1)}
  .ramp{width:150px;height:10px;border-radius:5px;border:1px solid var(--gold-line)}
  .ramp-wrap{display:flex;flex-direction:column;gap:3px}
  .ramp-lab{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10px}
  #stage{position:relative;flex:1;min-height:0;cursor:grab}
  #stage.drag{cursor:grabbing}
  canvas{display:block;width:100%;height:100%}
  #tip{position:absolute;pointer-events:none;z-index:5;opacity:0;transition:opacity .09s;
    background:rgba(20,17,26,.92);border:1px solid var(--gold-line);border-radius:10px;
    padding:10px 12px;backdrop-filter:blur(7px);box-shadow:0 12px 34px rgba(0,0,0,.55);min-width:190px}
  #tip .cards{display:flex;gap:4px;margin-bottom:8px}
  #tip .cards img{height:96px;border-radius:5px;border:1px solid var(--gold-line);
    box-shadow:0 3px 10px rgba(0,0,0,.5)}
  #tip .banwrap{position:relative;display:inline-block;line-height:0}
  #tip .banwrap img{filter:grayscale(.85) brightness(.5)}
  #tip .banwrap .slash{position:absolute;inset:0}
  #tip .banwrap .slash::before{content:"";position:absolute;top:50%;left:-6%;width:112%;height:3px;
    background:rgba(226,74,64,.95);border-radius:2px;transform:rotate(-54deg);transform-origin:center}
  #tip .nm .no{color:#e24a40;font-style:italic}
  #tip .nm{font-family:var(--serif);font-size:14px;margin-bottom:6px;color:var(--ink);line-height:1.25}
  #tip .row{display:flex;justify-content:space-between;gap:16px;font-size:12px;padding:1px 0}
  #tip .row .k{color:var(--muted)} #tip .row .v{font-family:var(--mono);color:var(--ink)}
  #tip .big{font-size:20px;color:var(--p2);font-family:var(--mono);font-weight:600}
  .hint{position:absolute;left:16px;bottom:14px;z-index:3;color:var(--muted);font-size:11px;
    font-family:var(--mono);background:rgba(20,17,26,.6);padding:5px 9px;border-radius:7px;
    border:1px solid var(--line)}
  #body{flex:1;min-height:0;display:flex}
  #stage{flex:1}
  /* ---- combo forge sidebar ---- */
  #forge{width:280px;flex:none;border-left:1px solid var(--line);background:linear-gradient(180deg,
    var(--ground2),var(--ground));display:flex;flex-direction:column;overflow-y:auto;padding:14px 14px 18px}
  #forge h2{margin:0 0 2px;font-family:var(--serif);font-weight:600;font-size:16px;color:var(--ink)}
  #forge .fh{color:var(--muted);font-size:11px;margin-bottom:10px;line-height:1.4}
  #forge .fh b{color:var(--gold)}
  .rack{display:grid;grid-template-columns:repeat(7,1fr);gap:4px}
  .tile{position:relative;border-radius:5px;overflow:hidden;cursor:pointer;aspect-ratio:3/4;
    border:2px solid transparent;transition:border-color .1s,filter .1s;background:#0d0b12}
  .tile img{width:100%;height:100%;object-fit:cover;display:block;pointer-events:none}
  .tile.inc{border-color:var(--gold);box-shadow:0 0 8px rgba(212,178,74,.55)}
  .tile.exc{border-color:#e24a40}
  .tile.exc img{filter:grayscale(.85) brightness(.45)}   /* fade the ART only, not the slash */
  .tile.exc::after{content:"";position:absolute;top:50%;left:-8%;width:116%;height:3px;z-index:2;
    background:#ef5347;box-shadow:0 0 5px rgba(239,83,71,.9);transform:rotate(-52deg);transform-origin:center}
  .tile .cnt{position:absolute;right:1px;top:1px;background:var(--gold);color:#1a1206;font-family:var(--mono);
    font-size:10px;font-weight:700;border-radius:3px;padding:0 3px;line-height:14px}
  .tile.dim{opacity:.4;cursor:default}
  #forge .used{font-family:var(--mono);font-size:10.5px;color:var(--muted);text-align:right;margin:5px 1px 12px}
  #forge .out{border-top:1px solid var(--line);padding-top:12px;flex:1}
  .out .oc{display:flex;gap:4px;margin-bottom:9px;flex-wrap:wrap}
  .out .oc .banwrap{position:relative;display:inline-block;line-height:0}
  .out .oc .banwrap::after{content:"";position:absolute;top:50%;left:-6%;width:112%;height:2px;
    background:#e24a40;transform:rotate(-52deg);transform-origin:center}
  .out .oc img{height:74px;border-radius:4px;border:1px solid var(--gold-line)}
  .out .oc img.ban{filter:grayscale(.85) brightness(.5)}
  .out .onm{font-family:var(--serif);font-size:13.5px;margin-bottom:9px;color:var(--ink);line-height:1.25}
  .out .onm .no{color:#e24a40;font-style:italic}
  .out .bar{margin:7px 0}
  .out .bar .lab{display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px}
  .out .bar .lab .s{color:var(--muted)} .out .bar .lab .v{font-family:var(--mono)}
  .out .bar .track{height:8px;border-radius:4px;background:rgba(255,255,255,.06);overflow:hidden}
  .out .bar .fill{height:100%;border-radius:4px}
  .out .meta{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:10px;display:flex;
    justify-content:space-between}
  .out .note{margin-top:9px;font-size:10.5px;color:var(--muted);font-style:italic;line-height:1.4;
    border-top:1px dashed var(--line);padding-top:8px}
  .out .empty{color:var(--muted);font-size:12px;line-height:1.5;font-style:italic}
</style>
<div id="wrap">
  <header>
    <div>
      <h1>The Honeycomb <span class="amp">&middot;</span> Wall of Hands</h1>
      <div class="sub">Every recorded starting-hand combo, in columns by <b>summed card value</b>.
        <b class="p2t">2nd player</b> grows up, <b class="p1t">1st player</b> grows down &mdash; the same combos
        on both sides, so any up/down imbalance <em>is</em> the seat advantage. Each stack is sorted by that
        seat's winrate; the weaving lines are iso-winrate contours.</div>
    </div>
    <div class="legend">
      <span class="key"><i class="sw" style="background:#e0873a"></i>P2 &uarr;</span>
      <span class="key"><i class="sw" style="background:#5f93cc"></i>P1 &darr;</span>
      <div class="ramp-wrap">
        <div class="ramp" id="ramp"></div>
        <div class="ramp" id="ramp1"></div>
        <div class="ramp-lab"><span>20%</span><span>winrate (zoomed out)</span><span>90%</span></div>
      </div>
    </div>
  </header>
  <div id="body">
    <div id="stage">
      <canvas id="c"></canvas>
      <div id="tip"></div>
      <div class="hint">scroll to zoom &middot; drag to pan &middot; hover a cell</div>
    </div>
    <aside id="forge">
      <h2>Combo Forge</h2>
      <div class="fh"><b>left-click</b> to hold a card &middot; <b>right-click</b> to ban &middot; up to 4 total.
        The matching hex glows in both walls.</div>
      <div class="rack" id="rack"></div>
      <div class="used" id="used">0 / 4</div>
      <div class="out" id="forgeOut"></div>
    </aside>
  </div>
</div>
<script>
const COMBOS = /*__COMBOS__*/;
const ART = /*__ART__*/;
const META = /*__META__*/;

// a stable identity for each tile so the forge can look it up: sorted held cards, then sorted banned cards
const keyOf = c => [...c.cards].sort().join("+") + "¬" + [...(c.excl||[])].sort().join("+");
for(const c of COMBOS) c._key = keyOf(c);
const KEYMAP = new Map(COMBOS.map(c=>[c._key, c]));
let forgeKey = null;                        // the currently forged combo's key (glows in the mural)

// base (non-partition) tiles keyed by their held cards -> lets the forge compute ANY exclusion's stats
// on the fly by inclusion-exclusion, even when there is no drawn hex for it.
const BASEMAP = new Map();
for(const c of COMBOS) if(!c.excl) BASEMAP.set([...c.cards].sort().join("+"), c);
function ieStat(include, exclude){          // n(include, none of exclude) per seat, or null if uncomputable
  let n1=0,w1=0,n2=0,w2=0;
  for(let mask=0; mask<(1<<exclude.length); mask++){
    const T=[]; let bits=0;
    for(let i=0;i<exclude.length;i++) if(mask&(1<<i)){ T.push(exclude[i]); bits++; }
    const c = BASEMAP.get([...include,...T].sort().join("+"));
    if(!c) return null;                     // a required superset isn't in the table
    const s = bits%2 ? -1 : 1;
    n1+=s*c.n1; w1+=s*c.wr1*c.n1; n2+=s*c.n2; w2+=s*c.wr2*c.n2;
  }
  return (n1>0 && n2>0) ? {n1:Math.round(n1), wr1:w1/n1, n2:Math.round(n2), wr2:w2/n2} : null;
}

// ---- preload the 14 card arts ------------------------------------------------------------------
const IMG = {}; let ready = 0; const need = Object.keys(ART).length;
for(const k in ART){ const im = new Image(); im.onload = ()=>{ if(++ready===need) draw(); }; im.src = ART[k]; IMG[k]=im; }

// ---- diverging winrate ramp (about the seat average) -------------------------------------------
const STOPS_P2 = [[0.18,[74,46,34]],[0.40,[138,80,44]],[0.55,[190,112,50]],[0.72,[224,140,58]],[0.90,[242,198,78]]];
const STOPS_P1 = [[0.18,[40,43,66]],[0.40,[50,74,112]],[0.55,[70,108,158]],[0.72,[95,147,204]],[0.90,[156,202,236]]];
function ramp(wr, stops){
  const t = Math.max(stops[0][0], Math.min(stops[stops.length-1][0], wr));
  for(let i=1;i<stops.length;i++){ if(t<=stops[i][0]){
    const [a,ca]=stops[i-1], [b,cb]=stops[i], f=(t-a)/(b-a);
    return ca.map((v,j)=>Math.round(v+(cb[j]-v)*f)); } }
  return stops[stops.length-1][1];
}
const rgb=(c,al)=>`rgba(${c[0]},${c[1]},${c[2]},${al==null?1:al})`;
const rampCss=stops=>`linear-gradient(90deg,${stops.map(s=>rgb(ramp(s[0],stops))+" "+Math.round((s[0]-0.18)/(0.90-0.18)*100)+"%").join(",")})`;
{ const a=document.getElementById("ramp"); if(a) a.style.background=rampCss(STOPS_P2);
  const b=document.getElementById("ramp1"); if(b) b.style.background=rampCss(STOPS_P1); }

// ---- layout: flat-top hex stacks, dense summed-values wrapped into sub-columns ------------------
// A summed value with more than CAP combos is spread across ceil(N/CAP) side-by-side sub-columns to
// squash the bell's height into width. Combos are INTERLEAVED by winrate rank across the sub-columns
// (rank i -> sub-column i%w), so at any height every sub-column of a value sits at ~the same winrate --
// which keeps the winrate contour lines continuous as they cross the sub-columns.
const R = 46;                              // hex radius (world px)
const COLW = 1.5*R, HEXH = Math.sqrt(3)*R; // column pitch, vertical pitch
const CAP = META.cap;                      // max hexes per sub-column before wrapping
const MODE = META.mode;                    // "second" | "first" | "both"
const GAP = MODE==="both" ? HEXH*0.9 : 0;  // half-gap at the central axis (both-mode only)

// shared x-layout: sub-column count per summed value depends only on how many combos share it, which is
// identical for both seats -- so the two walls line up column-for-column.
const sums = [...new Set(COMBOS.map(c=>c.s))].sort((a,b)=>a-b);
const byS = new Map(sums.map(s=>[s,[]]));
for(const c of COMBOS) byS.get(c.s).push(c);
const groups = [], subX = new Map();
let colX = 0, maxRows = 0;
for(const s of sums){
  const w = Math.max(1, Math.ceil(byS.get(s).length/CAP));
  maxRows = Math.max(maxRows, Math.ceil(byS.get(s).length/w));
  subX.set(s, Array.from({length:w}, (_,j)=>(colX+j)*COLW));
  groups.push({s, x0:colX*COLW, x1:(colX+w-1)*COLW});
  colX += w;
}
const nCols = colX, STACK = maxRows*HEXH;

// a wall = one seat's mural. `sign` is the growth direction (-1 up, +1 down) and `base` the world-y it
// grows from; combos are sorted by THIS seat's winrate (worst near the axis, best at the tip) and
// interleaved across the shared sub-columns so the contours stay continuous.
function makeWall(wrKey, nKey, avg, seam, accent, stops, sign, base){
  const subcols = [];
  for(const s of sums){
    const xs = subX.get(s), w = xs.length;
    const arr = byS.get(s).slice().sort((a,b)=>a[wrKey]-b[wrKey]);
    const bk = Array.from({length:w}, ()=>[]);
    arr.forEach((c,i)=>bk[i%w].push(c));
    for(let j=0;j<w;j++) subcols.push({x:xs[j], cells:bk[j]});
  }
  const cells = [];
  for(let ci=0; ci<subcols.length; ci++){
    const sc = subcols[ci], off = (ci%2)? HEXH/2 : 0;
    sc.cells.forEach((c,j)=>cells.push({...c, wr:c[wrKey], n:c[nKey], seat:wrKey,
      cx:sc.x, cy: base + sign*(off + (j+0.5)*HEXH)}));
  }
  const contours = META.contours.map(t=>({t, avg: Math.abs(t-avg) < 0.006,
    pts: subcols.map(sc=>[sc.x, base + sign*sc.cells.reduce((n,c)=>n+(c[wrKey]<t?1:0),0)*HEXH])}));
  return {seam, accent, stops, cells, contours};
}

const AMBER = "rgba(190,112,52,.9)", BLUE = "rgba(95,147,204,.92)";
const GOLD = {line:"rgba(242,214,120,.98)", lab:"#f2d678"};   // P2 avg-contour accent
const BLU  = {line:"rgba(150,196,240,.98)", lab:"#9cc3f2"};   // P1 avg-contour accent
const WALLS = [];
if(MODE==="both"){
  WALLS.push(makeWall("wr2","n2",META.p2_avg,AMBER,GOLD,STOPS_P2,-1,-GAP));   // P2 grows UP  (amber)
  WALLS.push(makeWall("wr1","n1",META.p1_avg,BLUE ,BLU ,STOPS_P1,+1,+GAP));   // P1 grows DOWN (blue)
}else if(MODE==="first"){
  WALLS.push(makeWall("wr1","n1",META.p1_avg,BLUE ,BLU ,STOPS_P1,-1,STACK));  // single wall, up from the floor
}else{
  WALLS.push(makeWall("wr2","n2",META.p2_avg,AMBER,GOLD,STOPS_P2,-1,STACK));
}
const allCells = [].concat(...WALLS.map(w=>w.cells));
let cyMin = Infinity, cyMax = -Infinity;
for(const c of allCells){ if(c.cy<cyMin) cyMin=c.cy; if(c.cy>cyMax) cyMax=c.cy; }
const worldW = (nCols-1)*COLW, worldCx = worldW/2, worldCy = (cyMin+cyMax)/2, worldH = (cyMax-cyMin)+2*R;

// ---- camera (pan/zoom, hand-rolled) ------------------------------------------------------------
const stage = document.getElementById("stage"), cv = document.getElementById("c"), ctx = cv.getContext("2d");
let DPR = Math.min(2, window.devicePixelRatio||1);
let cam = {sc:1, tx:0, ty:0};              // screen = world*sc + t
function resize(){
  const r = stage.getBoundingClientRect();
  cv.width = Math.round(r.width*DPR); cv.height = Math.round(r.height*DPR);
  cv.style.width=r.width+"px"; cv.style.height=r.height+"px";
  if(!resize._init){ fit(); resize._init=true; } draw();
}
function fit(){
  const r = stage.getBoundingClientRect(), pad=40;
  const scW = (r.width-2*pad)/(worldW+2*R);
  cam.sc = Math.min(15/R, scW);            // open at ~readable hex size, never wider than the viewport
  cam.tx = r.width/2 - worldCx*cam.sc;
  cam.ty = (r.height-24)/2 - worldCy*cam.sc;
}
const S = (wx,wy)=>[wx*cam.sc+cam.tx, wy*cam.sc+cam.ty];
const W = (sx,sy)=>[(sx-cam.tx)/cam.sc, (sy-cam.ty)/cam.sc];

// ---- draw --------------------------------------------------------------------------------------
function hexPath(x,y,r){
  ctx.beginPath();
  for(let k=0;k<6;k++){ const a=Math.PI/180*(60*k); const px=x+r*Math.cos(a), py=y+r*Math.sin(a);
    k?ctx.lineTo(px,py):ctx.moveTo(px,py); }
  ctx.closePath();
}
function cover(img,dx,dy,dw,dh){           // object-fit:cover into the rect
  const ar=dw/dh, ia=img.width/img.height; let sx=0,sy=0,sw=img.width,sh=img.height;
  if(ia>ar){ sw=img.height*ar; sx=(img.width-sw)/2; } else { sh=img.width/ar; sy=(img.height-sh)/2; }
  ctx.drawImage(img,sx,sy,sw,sh,dx,dy,dw,dh);
}
function draw(){
  const w=cv.width, h=cv.height; ctx.setTransform(1,0,0,1,0,0);
  ctx.clearRect(0,0,w,h); ctx.scale(DPR,DPR);
  const rS = R*cam.sc;                      // on-screen hex radius
  const vw=cv.width/DPR, vh=cv.height/DPR, m=rS+4;
  const showArt = rS >= 13 && ready===need;
  for(const wall of WALLS) for(const c of wall.cells){
    const [sx,sy]=S(c.cx,c.cy);
    if(sx< -m||sx>vw+m||sy< -m||sy>vh+m) continue;   // cull offscreen
    const col = ramp(c.wr, wall.stops);
    if(!showArt){
      hexPath(sx,sy,rS-0.6); ctx.fillStyle=rgb(col); ctx.fill();
      continue;
    }
    // zoomed in: card art stays fully vivid -- winrate is carried by the sort + the contour lines,
    // NOT by a wash (a tint would fight the varied card colours).
    ctx.save(); hexPath(sx,sy,rS-0.6); ctx.clip();
    const excl=c.excl||[], arts=c.cards.concat(excl), k=arts.length, x0=sx-rS, y0=sy-rS, sw=(2*rS)/k;
    for(let i=0;i<k;i++){ const im=IMG[arts[i]]; if(im&&im.complete) cover(im,x0+i*sw,y0,sw+0.5,2*rS); }
    for(let e=0;e<excl.length;e++){ const bx=x0+(c.cards.length+e)*sw;      // strike each EXCLUDED card
      ctx.fillStyle="rgba(17,14,22,.58)"; ctx.fillRect(bx,y0,sw+0.5,2*rS);
      ctx.strokeStyle="rgba(226,74,64,.94)"; ctx.lineWidth=Math.max(1.4,rS*0.11); ctx.lineCap="round";
      ctx.beginPath(); ctx.moveTo(bx+rS*0.14,y0+rS*0.14); ctx.lineTo(bx+sw-rS*0.14,y0+2*rS-rS*0.14); ctx.stroke(); }
    ctx.restore();
    hexPath(sx,sy,rS-0.6); ctx.lineWidth=Math.max(1,rS*(excl.length?0.06:0.05));
    ctx.strokeStyle=excl.length?"rgba(226,74,64,.8)":wall.seam; ctx.stroke();  // seam colour = seat (red = partition)
  }
  drawContours();
  if(forgeKey){ for(const wall of WALLS) for(const c of wall.cells){ if(c._key!==forgeKey) continue;
    const [sx,sy]=S(c.cx,c.cy); if(sx< -m||sx>vw+m||sy< -m||sy>vh+m) continue;   // forged combo glows
    hexPath(sx,sy,rS+3); ctx.lineWidth=2; ctx.strokeStyle="rgba(255,240,150,.5)"; ctx.stroke();
    hexPath(sx,sy,rS-0.6); ctx.lineWidth=Math.max(3,rS*0.16); ctx.strokeStyle="rgba(255,240,150,.98)"; ctx.stroke(); } }
  if(hoverCell){ const [sx,sy]=S(hoverCell.cx,hoverCell.cy);      // hover ring
    hexPath(sx,sy,rS-0.6); ctx.lineWidth=Math.max(2,rS*0.11); ctx.strokeStyle="rgba(240,222,120,.95)"; ctx.stroke(); }
  drawAxis();
}
const MONO = "ui-monospace,Consolas,monospace";
function drawContours(){
  ctx.lineJoin="round"; ctx.lineCap="round";
  for(const wall of WALLS) for(const cn of wall.contours){
    ctx.beginPath();
    cn.pts.forEach((p,i)=>{ const [x,y]=S(p[0],p[1]); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
    ctx.strokeStyle="rgba(9,7,13,.72)"; ctx.lineWidth=cn.avg?8:6; ctx.stroke();          // dark halo
    ctx.strokeStyle=cn.avg?wall.accent.line:"rgba(233,225,212,.92)";       // avg line = seat accent
    ctx.lineWidth=cn.avg?3.4:2; ctx.stroke();
    const last=cn.pts[cn.pts.length-1], [lx,ly]=S(last[0],last[1]);       // label at the right edge
    ctx.font=(cn.avg?"bold ":"")+"12px "+MONO; ctx.textAlign="left"; ctx.textBaseline="middle";
    ctx.fillStyle="rgba(9,7,13,.82)"; ctx.fillRect(lx+5,ly-9,cn.avg?58:40,18);
    ctx.fillStyle=cn.avg?wall.accent.lab:"#e9e1d4";
    ctx.fillText((cn.t*100).toFixed(0)+"%"+(cn.avg?" avg":""), lx+8, ly+1);
  }
}
function drawAxis(){
  const vh=cv.height/DPR, atCenter = MODE==="both";
  if(atCenter){                                       // faint central axis line at world y=0
    const [x0,ay]=S(-R,0), [x1]=S(worldW+R,0);
    ctx.strokeStyle="rgba(212,178,74,.22)"; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(x0,ay); ctx.lineTo(x1,ay); ctx.stroke();
  }
  ctx.font="bold 16px "+MONO; ctx.textAlign="center"; ctx.textBaseline="middle";
  for(const g of groups){ const [sx,sy0]=S((g.x0+g.x1)/2,0);   // labels sit on the central axis (both) or bottom
    if(sx<10||sx>cv.width/DPR-10) continue;
    const y = atCenter ? sy0 : vh-14, tw=String(g.s).length*10+12;
    ctx.fillStyle="rgba(15,12,20,.72)"; ctx.fillRect(sx-tw/2,y-11,tw,22);
    ctx.fillStyle="rgba(202,194,216,.95)"; ctx.fillText(g.s, sx, y+1);
  }
}

// ---- interaction -------------------------------------------------------------------------------
let hoverCell=null;
function pick(sx,sy){                        // nearest hex center within radius, across both walls
  const [wx,wy]=W(sx,sy); let best=null,bd=(R*0.9)*(R*0.9);
  for(const c of allCells){ const dx=c.cx-wx, dy=c.cy-wy, d=dx*dx+dy*dy; if(d<bd){bd=d;best=c;} }
  return best;
}
const tip=document.getElementById("tip");
function showTip(c,cx,cy){
  const excl=c.excl||[], arts=c.cards.concat(excl), nInc=c.cards.length;
  const imgs=arts.map((n,i)=> (i>=nInc)
      ? `<span class="banwrap"><img src="${ART[n]}" alt="${n}"><i class="slash"></i></span>`
      : `<img src="${ART[n]}" alt="${n}">`).join("");
  const pretty=excl.length ? `${c.cards.join(" + ")} <span class="no">&mdash; no ${excl.join(", ")}</span>` : c.cards.join("  +  ");
  const row=(lab,wr,col,hot)=>`<div class="row"><span class="k">${lab}</span>`+
    `<span class="${hot?'big':'v'}" style="${hot?'color:'+col:''}">${(wr*100).toFixed(1)}%</span></div>`;
  tip.innerHTML=`<div class="cards">${imgs}</div><div class="nm">${pretty}</div>`+
    row("P2 &uarr; winrate", c.wr2, "var(--p2)", c.seat==="wr2")+
    row("P1 &darr; winrate", c.wr1, "var(--p1)", c.seat==="wr1")+
    `<div class="row"><span class="k">summed value</span><span class="v">${c.s}</span></div>`+
    `<div class="row"><span class="k">games</span><span class="v">P2 ${c.n2} &middot; P1 ${c.n1}</span></div>`;
  tip.style.opacity=1;
  const r=stage.getBoundingClientRect(), tw=tip.offsetWidth, th=tip.offsetHeight;
  let x=cx+16, y=cy+16; if(x+tw>r.width) x=cx-tw-16; if(y+th>r.height) y=cy-th-16;
  tip.style.left=Math.max(4,x)+"px"; tip.style.top=Math.max(4,y)+"px";
}
stage.addEventListener("pointermove",e=>{
  if(drag){ cam.tx+=e.clientX-drag.x; cam.ty+=e.clientY-drag.y; drag.x=e.clientX; drag.y=e.clientY; draw(); return; }
  const r=stage.getBoundingClientRect(), c=pick(e.clientX-r.left,e.clientY-r.top);
  if(c!==hoverCell){ hoverCell=c; draw(); }
  if(c) showTip(c,e.clientX-r.left,e.clientY-r.top); else tip.style.opacity=0;
});
let drag=null;
stage.addEventListener("pointerdown",e=>{ drag={x:e.clientX,y:e.clientY}; stage.classList.add("drag"); tip.style.opacity=0; });
addEventListener("pointerup",()=>{ drag=null; stage.classList.remove("drag"); });
stage.addEventListener("wheel",e=>{
  e.preventDefault();
  const r=stage.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const [wx,wy]=W(mx,my); const f=Math.exp(-e.deltaY*0.0016);
  cam.sc=Math.max(0.05,Math.min(6,cam.sc*f));
  cam.tx=mx-wx*cam.sc; cam.ty=my-wy*cam.sc; draw();
},{passive:false});
// ---- combo forge -------------------------------------------------------------------------------
const inc = new Map(), exc = new Set();     // held cards (name->count) and banned cards
const rack = document.getElementById("rack"), usedEl = document.getElementById("used"),
      outEl = document.getElementById("forgeOut"), CARDMAX = {}, VAL = {};
META.cards.forEach(cd=>{
  CARDMAX[cd.name] = cd.max; VAL[cd.name] = cd.val;
  const t = document.createElement("div"); t.className = "tile"; t.dataset.name = cd.name;
  t.innerHTML = `<img src="${ART[cd.name]}" alt="${cd.name}" title="${cd.name} (value ${cd.val})">`+
                `<span class="cnt" style="display:none"></span>`;
  t.addEventListener("click", ()=>leftClick(cd.name));
  t.addEventListener("contextmenu", e=>{ e.preventDefault(); rightClick(cd.name); });
  rack.appendChild(t);
});
function used(){ let t = exc.size; for(const v of inc.values()) t += v; return t; }
function leftClick(name){
  if(exc.has(name)){ exc.delete(name); inc.set(name,1); return update(); }   // ban -> hold
  const cur = inc.get(name)||0;
  if(cur >= CARDMAX[name]) inc.delete(name);                                 // cycle back to none at the cap
  else if(used() < 4) inc.set(name, cur+1);
  update();
}
function rightClick(name){
  if(inc.has(name)) inc.delete(name);
  if(exc.has(name)) exc.delete(name);
  else if(used() < 4) exc.add(name);
  update();
}
function selKey(){
  const cs = []; for(const [n,v] of inc) for(let i=0;i<v;i++) cs.push(n);
  return cs.sort().join("+") + "¬" + [...exc].sort().join("+");
}
function update(){
  const u = used();
  for(const t of rack.children){ const n = t.dataset.name, cnt = inc.get(n)||0;
    t.classList.toggle("inc", cnt>0); t.classList.toggle("exc", exc.has(n));
    const badge = t.querySelector(".cnt");
    if(cnt>1){ badge.textContent = cnt; badge.style.display = ""; } else badge.style.display = "none";
    t.classList.toggle("dim", !(cnt>0 || exc.has(n) || u<4));                // greyed = at the 4-card cap
  }
  usedEl.textContent = u + " / 4";
  const hasSel = inc.size>0 || exc.size>0;
  const m = hasSel ? KEYMAP.get(selKey()) : null;          // a drawn tile (glows in both walls)?
  forgeKey = m ? m._key : null;
  let stat = m, computed = false;
  if(!m && exc.size>0){                                     // no drawn hex, but the exclusion may be computable
    const incArr = []; for(const [n,v] of inc) for(let i=0;i<v;i++) incArr.push(n);
    stat = ieStat(incArr, [...exc]); computed = !!stat;
  }
  renderOut(hasSel, stat, computed);
  draw();
}
function renderOut(hasSel, stat, computed){
  if(!hasSel){ outEl.innerHTML = `<div class="empty">Left-click cards to hold them, right-click to ban. Any `+
    `held combo lights up its hex; a ban is computed from the table too (only the six curated bans have a hex).</div>`; return; }
  if(!stat){ outEl.innerHTML = `<div class="empty">${exc.size
    ? "Can't compute &mdash; held + banned cards exceed the table's 4-card range, or a piece is too rare."
    : "Not a recorded combo (too rare for stable stats)."}</div>`; return; }
  const inCards = []; for(const [n,v] of [...inc].sort()) for(let i=0;i<v;i++) inCards.push(n);
  const sval = inCards.reduce((a,n)=>a+VAL[n],0);
  const imgs = inCards.map(n=>`<img src="${ART[n]}">`).join("")
    + [...exc].sort().map(n=>`<span class="banwrap"><img class="ban" src="${ART[n]}"></span>`).join("");
  const nm = inCards.join(" + ") + (exc.size ? ` <span class="no">&mdash; no ${[...exc].sort().join(", ")}</span>` : "");
  const bar = (lab,wr,col)=>`<div class="bar"><div class="lab"><span class="s">${lab}</span>`+
    `<span class="v">${(wr*100).toFixed(1)}%</span></div><div class="track">`+
    `<div class="fill" style="width:${(wr*100).toFixed(0)}%;background:${col}"></div></div></div>`;
  outEl.innerHTML = `<div class="oc">${imgs}</div><div class="onm">${nm}</div>`+
    bar("P2 &uarr; win", stat.wr2, "var(--p2)") + bar("P1 &darr; win", stat.wr1, "var(--p1)")+
    `<div class="meta"><span>value ${sval}</span><span>n: P2 ${stat.n2} &middot; P1 ${stat.n1}</span></div>`+
    (computed ? `<div class="note">computed by inclusion-exclusion &mdash; no drawn hex for this ban</div>` : "");
}
update();

addEventListener("resize",resize);
resize();
</script>
"""


def build(csv_path: str, out_path: str, seat: str, assets_dir: str, art_width: int, cap: int) -> str:
    combos = _combos(csv_path) + _partition_tiles(csv_path)
    art = _art_data_uris(assets_dir, art_width)

    def seat_avg(wr_k, n_k):
        tot = sum(c[n_k] for c in combos)
        return sum(c[wr_k] * c[n_k] for c in combos) / max(1, tot)

    p1_avg, p2_avg = seat_avg("wr1", "n1"), seat_avg("wr2", "n2")
    levels = {0.40, 0.50, 0.70, 0.80, round(p2_avg, 3)}
    if seat in ("both", "first"):
        levels.add(round(p1_avg, 3))
    card_list = [{"name": cdef.name, "val": cdef.value, "max": cnt} for cdef, cnt in cards.DECK_SPEC]
    meta = {"seat": seat, "mode": seat, "p1_avg": round(p1_avg, 4), "p2_avg": round(p2_avg, 4),
            "n_combos": len(combos), "cap": cap, "contours": sorted(levels), "cards": card_list}
    html = (_PAGE
            .replace("/*__COMBOS__*/", json.dumps(combos, separators=(",", ":")))
            .replace("/*__ART__*/", json.dumps(art, separators=(",", ":")))
            .replace("/*__META__*/", json.dumps(meta, separators=(",", ":"))))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build the hand-combo honeycomb mural (self-contained HTML).")
    p.add_argument("--stats", default=os.path.join("results", "corpus_stats", "selfplay_gen25", "hand_combos.csv"))
    p.add_argument("--out", default=os.path.join("reports", "honeycomb_p2.html"))
    p.add_argument("--seat", choices=["second", "first", "both"], default="second")
    p.add_argument("--assets", default="assets")
    p.add_argument("--art-width", type=int, default=150, help="downscaled card-art width in px")
    p.add_argument("--cap", type=int, default=55, help="max combos per sub-column before wrapping (squash)")
    args = p.parse_args(argv)
    out = build(args.stats, args.out, args.seat, args.assets, args.art_width, args.cap)
    kb = os.path.getsize(out) / 1024
    print(f"wrote {out}  ({kb:.0f} KB)  seat={args.seat}")


if __name__ == "__main__":
    main()
