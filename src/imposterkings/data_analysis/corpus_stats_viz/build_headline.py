"""The Headline -- an engaging infographic of the turn-order edge and how it is won.

    python -m imposterkings.data_analysis.corpus_stats_viz.build_headline \
        --stats results/corpus_stats/gen25_5k/selfplay_gen25 \
        --assets assets --out reports/headline.html

Reads corpus_stats CSVs (turn_order, win_reasons_by_seat, setup_hide, setup_discard) and emits ONE
self-contained, CSP-safe HTML page: the ~59/41 second-player headline, then a pill toggle over three views,
each split into a First-player and a Second-player VERTICAL stacked bar. WIN REASON stacks by outcome; HIDE
and DISCARD stack by CARD (card-coloured) with a central portrait ladder whose zig-zag leaders start at each
segment. Card arts + colours are inlined; no external dependencies.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
from typing import Dict, List

from ... import cards  # type: ignore

NAME_VAL: Dict[str, int] = {}
NAME_ASSET: Dict[str, str] = {}
for _cdef, _cnt in cards.DECK_SPEC:
    NAME_VAL[_cdef.name] = _cdef.value
    NAME_ASSET[_cdef.name] = _cdef.asset


def _art_data_uris(assets_dir: str, width: int) -> Dict[str, str]:
    from PIL import Image
    out: Dict[str, str] = {}
    for name, fname in NAME_ASSET.items():
        with Image.open(os.path.join(assets_dir, fname)) as im:
            im = im.convert("RGB")
            h = round(im.height * width / im.width)
            im = im.resize((width, h), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=74, optimize=True)
        out[name] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return out


def _card_colors(assets_dir: str) -> Dict[str, str]:
    """name -> hex, from ImposterKings_colors.txt (the file spells it 'King's Hand')."""
    out: Dict[str, str] = {}
    with open(os.path.join(assets_dir, "ImposterKings_colors.txt"), encoding="utf-8") as fh:
        for line in fh:
            if ":" in line:
                nm, hx = line.split(":", 1)
                out[nm.strip().replace("King's Hand", "KingsHand")] = hx.strip()
    return out


def _read(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


REASONS = [   # display order + palette, matched to the four win_reason() codes
    ("assassin_kills_king", "Assassin instant-kill", "#d1504a"),
    ("stalled_out_of_cards", "Opponent runs out of cards", "#3fae9b"),
    ("stalled_king_spent", "Can't beat the throne, king already spent", "#cba24f"),
    ("kingshand_counter_stalls_them", "King's-Hand counter into an unbeatable throne", "#9a7ad0"),
]


def build(stats_dir: str, out_path: str, assets_dir: str, art_width: int) -> str:
    turn = {r["went"]: r for r in _read(os.path.join(stats_dir, "turn_order.csv"))}
    reason_by_seat = {"second": {}, "first": {}}
    for r in _read(os.path.join(stats_dir, "win_reasons_by_seat.csv")):
        reason_by_seat[r["winner"]][r["reason"]] = int(r["n"])
    colors = _card_colors(assets_dir)

    def setup_rows(fname: str) -> List[dict]:
        rows = []
        for r in _read(os.path.join(stats_dir, fname)):
            pf, ps = int(r["played_first"]), int(r["played_second"])
            if pf + ps <= 0:
                continue
            rows.append({
                "name": r["card"], "val": NAME_VAL.get(r["card"], 0), "color": colors.get(r["card"], "#888"),
                "pf": pf, "ps": ps,
                "rate_first": float(r["rate_first"]) if r["rate_first"] else 0.0,
                "rate_second": float(r["rate_second"]) if r["rate_second"] else 0.0,
                "p1_minus_p2": float(r["p1_minus_p2"]) if r["p1_minus_p2"] not in ("", None) else 0.0,
            })
        rows.sort(key=lambda x: -(x["pf"] + x["ps"]))     # most-chosen first (top of the bar + ladder)
        return rows

    data = {
        "games": int(turn["second"]["games"]),
        "second": {"winrate": float(turn["second"]["winrate"]), "wins": int(turn["second"]["wins"])},
        "first": {"winrate": float(turn["first"]["winrate"]), "wins": int(turn["first"]["wins"])},
        "reasons": [{"key": k, "label": lab, "color": col} for k, lab, col in REASONS],
        "reasonBySeat": reason_by_seat,
        "hide": setup_rows("setup_hide.csv"),
        "discard": setup_rows("setup_discard.csv"),
    }
    art = _art_data_uris(assets_dir, art_width)
    html = _TEMPLATE.replace("/*__DATA__*/", json.dumps(data)).replace("/*__ART__*/", json.dumps(art))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path


_TEMPLATE = r"""
<title>Imposter Kings — the second-player edge</title>
<style>
:root{ --ink:#0c0a06; --panel:#171208; --edge:#3a2f1c; --cream:#ece2cc; --muted:#a2916f; --faint:#6d5f45;
  --gold:#e8b44a; --blue:#6ba2dd; --barH:540px; --hudH:850px; }
*{box-sizing:border-box}
.stage{ max-width:1480px; margin:0 auto; padding:38px 24px 60px; color:var(--cream);
  font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  background:radial-gradient(120% 80% at 50% -10%,#241a0d 0%,transparent 55%),
    radial-gradient(90% 60% at 50% 120%,#1a1207 0%,transparent 60%),var(--ink);
  background-attachment:fixed; min-height:100%; }
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.eyebrow{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;letter-spacing:.28em;
  text-transform:uppercase;color:var(--muted);text-align:center}
h1{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;font-weight:600;text-align:center;
  line-height:1.06;margin:.42em 0 .1em;font-size:clamp(30px,6.2vw,58px);text-wrap:balance;letter-spacing:-.01em}
h1 em{font-style:normal;color:var(--gold)}
.big{font-size:1.18em;color:var(--gold);text-shadow:0 2px 26px rgba(232,180,74,.35)}
.big .pct{font-size:.5em;vertical-align:.35em}
.dek{text-align:center;color:var(--muted);max-width:54ch;margin:.5em auto 0;font-size:15px;line-height:1.5}
.dek b{color:var(--cream);font-weight:600}
.split{margin:26px auto 4px;max-width:720px}
.splitbar{position:relative;height:46px;border-radius:9px;overflow:hidden;display:flex;border:1px solid var(--edge);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.05),0 10px 30px rgba(0,0,0,.5)}
.splitbar .seg{display:flex;align-items:center;height:100%;width:0;transition:width 1.1s cubic-bezier(.2,.8,.2,1)}
.seg.p2{background:linear-gradient(180deg,#efbe5c,#c9902f);color:#2a1c05}
.seg.p1{background:linear-gradient(180deg,#5f93cc,#365a86);color:#eaf1fb;justify-content:flex-end}
.seg .lab{padding:0 14px;font-size:12px;letter-spacing:.16em;text-transform:uppercase;font-weight:700;white-space:nowrap}
.splitnote{display:flex;justify-content:space-between;margin-top:8px;color:var(--faint);font-size:11px;letter-spacing:.14em;text-transform:uppercase}

.pills{display:flex;gap:8px;justify-content:center;margin:34px auto 8px;flex-wrap:wrap}
.pill{appearance:none;border:1px solid var(--edge);background:var(--panel);color:var(--muted);
  font:600 12.5px/1 ui-sans-serif,system-ui;letter-spacing:.12em;text-transform:uppercase;padding:11px 18px;
  border-radius:999px;cursor:pointer;transition:.18s}
.pill:hover{color:var(--cream);border-color:#5a4a2c}
.pill[aria-selected="true"]{background:linear-gradient(180deg,#efbe5c,#cf9a34);color:#2a1c05;border-color:#efbe5c;box-shadow:0 6px 20px rgba(232,180,74,.28)}
.pill:focus-visible{outline:2px solid var(--gold);outline-offset:2px}

.frame{margin-top:22px;border:1px solid var(--edge);border-radius:14px;
  background:linear-gradient(180deg,rgba(31,24,16,.9),rgba(18,13,7,.9));padding:22px 18px 16px;position:relative}
.frame::before,.frame::after{content:"";position:absolute;width:14px;height:14px;border:2px solid #7a5a1f;opacity:.7}
.frame::before{top:9px;left:9px;border-right:0;border-bottom:0}
.frame::after{bottom:9px;right:9px;border-left:0;border-top:0}
.viewhead{font-size:12px;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);margin:0 0 14px;text-align:center}
.viewhead b{color:var(--cream)}
.caption{color:var(--muted);font-size:13.5px;line-height:1.55;text-align:center;max-width:64ch;margin:14px auto 0}
.caption b{color:var(--gold)}
.foot{margin-top:26px;text-align:center;color:var(--faint);font-size:11px}

/* two vertical stacked bars, plus optional central portrait ladder */
.duo{position:relative;display:flex;align-items:flex-start;justify-content:center;gap:clamp(40px,10vw,140px)}
.side{display:flex;flex-direction:column;align-items:center;z-index:2}
.vtitle{font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:12px;margin-bottom:2px}
.vtitle.p1{color:var(--blue)} .vtitle.p2{color:var(--gold)}
.vsub{color:var(--faint);font-family:ui-monospace,Menlo,monospace;font-size:11px;margin-bottom:10px}
.vbar{position:relative;width:78px;height:var(--barH);border-radius:8px;overflow:hidden;
  border:1px solid var(--edge);background:#0a0805;box-shadow:inset 0 0 26px rgba(0,0,0,.6)}
.vseg{width:100%;height:0;transition:height .95s cubic-bezier(.2,.8,.2,1);position:relative;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.13),inset 0 -1px 0 rgba(0,0,0,.28)}
.vseg.card{cursor:pointer}
.vseg.card:hover{filter:brightness(1.22) saturate(1.1);z-index:3}
.reasonseg{display:flex;align-items:center;justify-content:center}
.reasonseg span{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;font-weight:700;color:#0c0a06;
  text-shadow:0 1px 0 rgba(255,255,255,.28);opacity:0;transition:opacity .5s .7s}
.reasonseg.show span{opacity:1}

/* HUD unit: one seat's bar in the centre, its OWN cards fanning out to BOTH sides via zig-zag leaders */
.hud-wrap{display:flex;gap:22px;align-items:flex-start;justify-content:center}
.hud-unit{position:relative;flex:1 1 0;min-width:0;margin:2px 0 24px;padding:0 8px}
.hud-unit+.hud-unit{border-left:1px solid var(--edge)}
.utitle{text-align:center;font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:13px;margin-bottom:16px}
.utitle.p1{color:var(--blue)} .utitle.p2{color:var(--gold)}
.utitle .um{color:var(--faint);font-weight:500;letter-spacing:.04em;text-transform:none}
.hud-row{position:relative;display:flex;align-items:flex-start;justify-content:center;gap:0}
.hbar{position:relative;width:88px;height:var(--hudH);border-radius:9px;overflow:hidden;flex:none;
  border:1px solid var(--edge);background:#0a0805;box-shadow:inset 0 0 30px rgba(0,0,0,.6);z-index:2}
.hladder{width:36%;max-width:340px;height:var(--hudH);display:flex;flex-direction:column;
  justify-content:space-between;z-index:2}
.hladder.left{align-items:flex-start} .hladder.right{align-items:flex-end}
.hrung{display:flex;align-items:center;gap:11px;flex:1;min-height:0;transition:.18s}
.hrung img{height:84%;width:auto;max-height:150px;border-radius:6px;border:1px solid #000;
  box-shadow:0 4px 14px rgba(0,0,0,.6);display:block;flex:none}
.hrung .hn{font-size:12.5px;color:var(--muted);line-height:1.3}
.hrung .hn b{color:var(--cream);font-weight:600;display:block;font-size:13px}
.hrung .hn .hp{font-family:ui-monospace,Menlo,monospace;color:var(--faint);font-size:11px}
.hladder.left .hrung .hn{text-align:left}
.hladder.right .hrung .hn{text-align:right}
.hrung.dim{opacity:.3;filter:grayscale(.45)}
.vseg.card.hot{outline:2.5px solid var(--cream);outline-offset:-2px;z-index:6;
  box-shadow:0 0 16px rgba(236,226,204,.4),inset 0 0 0 1px rgba(0,0,0,.45)}
.leaders{position:absolute;inset:0;width:100%;height:100%;overflow:visible;pointer-events:none;z-index:1}

.legend{display:flex;flex-wrap:wrap;gap:8px 18px;justify-content:center;margin-top:16px}
.legend .it{display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--muted)}
.legend .sw{width:13px;height:13px;border-radius:3px;box-shadow:inset 0 0 0 1px rgba(0,0,0,.3)}
.tip{position:fixed;pointer-events:none;z-index:9;background:#0b0906;border:1px solid var(--edge);border-radius:8px;
  padding:9px 11px;font-size:12px;color:var(--cream);box-shadow:0 10px 30px rgba(0,0,0,.6);opacity:0;transition:opacity .12s;max-width:240px}
.tip .t-nm{font-weight:700;margin-bottom:3px}
@media (max-width:1080px){ .hud-wrap{flex-direction:column}
  .hud-unit+.hud-unit{border-left:0;border-top:1px solid var(--edge);padding-top:22px} }
@media (max-width:620px){ :root{--barH:430px;--hudH:540px} .vbar{width:52px} .hbar{width:58px}
  .hladder{width:40%} .hrung .hn{display:none} .hrung{gap:5px} h1{font-size:34px} }
</style>

<div class="stage">
  <div class="eyebrow" id="eyebrow"></div>
  <h1>The player who moves <em>second</em><br>wins <span class="big" id="hw"></span></h1>
  <p class="dek">Seats are symmetric and the deck is shuffled fresh each game, so this <b>18-point gap is pure
    turn order</b>. Below: <b>how</b> each side wins, and what the bot hides and discards to get there.</p>

  <div class="split">
    <div class="splitbar" id="splitbar">
      <div class="seg p2" id="segP2"><span class="lab">Second&nbsp;·&nbsp;<span id="p2n"></span></span></div>
      <div class="seg p1" id="segP1"><span class="lab"><span id="p1n"></span>&nbsp;·&nbsp;First</span></div>
    </div>
    <div class="splitnote"><span>Second player</span><span id="ngames"></span><span>First player</span></div>
  </div>

  <div class="pills" role="tablist" id="pills">
    <button class="pill" role="tab" data-v="reason" aria-selected="true">Win reason</button>
    <button class="pill" role="tab" data-v="hide">Hide setup</button>
    <button class="pill" role="tab" data-v="discard">Discard setup</button>
  </div>

  <div class="frame">
    <div class="viewhead" id="viewhead"></div>
    <div id="chart"></div>
    <p class="caption" id="caption"></p>
  </div>
  <div class="foot" id="foot"></div>
</div>
<div class="tip" id="tip"></div>

<script>
const DATA=/*__DATA__*/, ART=/*__ART__*/;
const NS="http://www.w3.org/2000/svg";
const $=(s,r=document)=>r.querySelector(s), el=(t,c)=>{const e=document.createElement(t);if(c)e.className=c;return e;};
const svgEl=(t,c)=>{const e=document.createElementNS(NS,t);if(c)e.setAttribute("class",c);return e;};
const pct=x=>(x*100).toFixed(1)+"%", int=n=>n.toLocaleString();
const tip=$("#tip");
const showTip=(h,x,y)=>{tip.innerHTML=h;tip.style.opacity=1;tip.style.left=Math.min(x+14,innerWidth-252)+"px";tip.style.top=(y+16)+"px";};
const hideTip=()=>tip.style.opacity=0;
let _resize=null;

$("#eyebrow").textContent="Imposter Kings · "+int(DATA.games)+" self-play games · hybrid k20·l3";
$("#hw").innerHTML=Math.round(DATA.second.winrate*100)+'<span class="pct">%</span>';
$("#p2n").textContent=pct(DATA.second.winrate); $("#p1n").textContent=pct(DATA.first.winrate);
$("#ngames").textContent=int(DATA.games)+" games";
$("#foot").textContent="Wins are per-game; a hand-row is one seat of one game. Second-player winrate 95% CI [57.8, 60.5]. Self-play — both seats share one policy, so the split is seat, not skill.";
requestAnimationFrame(()=>setTimeout(()=>{$("#segP2").style.width=(DATA.second.winrate*100)+"%";$("#segP1").style.width=(DATA.first.winrate*100)+"%";},80));

function renderReason(chart){
  const duo=el("div","duo");
  [["first","First player",DATA.first.wins,"p1"],["second","Second player",DATA.second.wins,"p2"]].forEach(([seat,who,tot,cl])=>{
    const side=el("div","side");
    side.innerHTML=`<div class="vtitle ${cl}">${who}</div><div class="vsub">${int(tot)} wins</div>`;
    const bar=el("div","vbar"); const by=DATA.reasonBySeat[seat];
    DATA.reasons.forEach(r=>{const n=by[r.key]||0; if(!n)return; const sh=n/tot;
      const seg=el("div","vseg reasonseg"); seg.style.background=r.color;
      seg.innerHTML=`<span>${(sh*100).toFixed(0)}%</span>`;
      seg.onmousemove=e=>showTip(`<div class="t-nm" style="color:${r.color}">${r.label}</div><div class="mono" style="color:var(--muted)">${int(n)} games · ${pct(sh)} of ${who.toLowerCase()} wins</div>`,e.clientX,e.clientY);
      seg.onmouseleave=hideTip; bar.append(seg);
      requestAnimationFrame(()=>setTimeout(()=>{seg.style.height=(sh*100)+"%";seg.classList.add("show");},110));
    });
    side.append(bar); duo.append(side);
  });
  chart.append(duo);
  const lg=el("div","legend");
  DATA.reasons.forEach(r=>{const it=el("div","it");const sw=el("span","sw");sw.style.background=r.color;it.append(sw,document.createTextNode(r.label));lg.append(it);});
  chart.append(lg);
  $("#caption").innerHTML="Both seats mostly win by the same stall, but the edge has a signature: <b>the second player wins far more by outlasting</b> — running the opponent out of cards is 32% of its wins versus just 14% of the first player's. Acting last, it is the one still holding a card to play.";
}

function renderSetup(chart, key){
  const verb=key==="hide"?"hidden":"discarded", single=verb.slice(0,-1)+"s";
  const draws=[]; const wrap=el("div","hud-wrap");
  ["first","second"].forEach(seat=>{
    const who=seat==="first"?"First player":"Second player", cls=seat==="first"?"p1":"p2";
    const fld=seat==="first"?"pf":"ps", rate=seat==="first"?"rate_first":"rate_second";
    const rows=DATA[key].slice().sort((a,b)=>b[fld]-a[fld]);   // each seat sorts by ITS own frequency
    const sum=rows.reduce((a,r)=>a+r[fld],0);
    const sideOf={};                                           // per-seat, local -> the deferred draw() can't
    rows.forEach((r,i)=>{sideOf[r.name]=(i%2===0)?"left":"right";});  // read another seat's assignment

    const unit=el("div","hud-unit");
    unit.innerHTML=`<div class="utitle ${cls}">${who} <span class="um">— what it ${single}</span></div>`;
    const row=el("div","hud-row"); const svg=svgEl("svg","leaders");
    const ladL=el("div","hladder left"), ladR=el("div","hladder right"), bar=el("div","hbar");
    const seg={}, img={};
    rows.filter(r=>sideOf[r.name]==="left").forEach(r=>mkRung(r,"left"));
    rows.filter(r=>sideOf[r.name]==="right").forEach(r=>mkRung(r,"right"));
    function mkRung(r,side){
      const rg=el("div","hrung");
      const im=`<img src="${ART[r.name]}" alt="${r.name}">`;
      const lab=`<span class="hn"><b>${r.name}</b><span class="hp">${(r[fld]/sum*100).toFixed(1)}% · val ${r.val}</span></span>`;
      rg.innerHTML=side==="left"?lab+im:im+lab;   // image on the bar-side, label outward -> leader clears the text
      (side==="left"?ladL:ladR).append(rg); img[r.name]=rg.querySelector("img"); img[r.name]._rg=rg;
    }
    rows.forEach(r=>{const sh=r[fld]/sum;
      const s=el("div","vseg card"); s.style.background=r.color; seg[r.name]=s;
      s.onmousemove=e=>showTip(`<div class="t-nm" style="color:${r.color}">${r.name} <span class="mono" style="color:var(--faint)">· val ${r.val}</span></div>`
        +`<div class="mono" style="color:var(--muted)">${who} ${verb} it ${int(r[fld])}× (${pct(sh)} of its ${verb})</div>`
        +`<div class="mono" style="color:var(--muted)">chosen ${pct(r[rate])} of when it held one</div>`,e.clientX,e.clientY);
      s.onmouseenter=()=>{s.classList.add("hot");Object.entries(img).forEach(([n,im])=>im._rg.classList.toggle("dim",n!==r.name));};
      s.onmouseleave=()=>{s.classList.remove("hot");hideTip();Object.values(img).forEach(im=>im._rg.classList.remove("dim"));};
      bar.append(s);
      requestAnimationFrame(()=>setTimeout(()=>s.style.height=(sh*100)+"%",110));
    });
    row.append(svg,ladL,bar,ladR); unit.append(row); wrap.append(unit);

    const draw=()=>{
      const dr=row.getBoundingClientRect();
      svg.setAttribute("viewBox",`0 0 ${dr.width} ${dr.height}`); svg.innerHTML="";
      rows.forEach(r=>{const s=seg[r.name].getBoundingClientRect(); if(s.height<=0)return;
        const im=img[r.name].getBoundingClientRect();
        const left=sideOf[r.name]==="left";
        const sx=(left?s.left:s.right)-dr.left, sy=s.top+s.height/2-dr.top;
        const tx=(left?im.right:im.left)-dr.left, ty=im.top+im.height/2-dr.top;
        const midx=(sx+tx)/2;
        const p=svgEl("polyline");
        p.setAttribute("points",`${sx},${sy} ${(sx+midx)/2},${sy} ${(midx+tx)/2},${ty} ${tx},${ty}`);
        p.setAttribute("fill","none");p.setAttribute("stroke",r.color);p.setAttribute("stroke-width","1.6");p.setAttribute("opacity",".62");
        svg.append(p);
        [[sx,sy],[tx,ty]].forEach(([x,y])=>{const d=svgEl("circle");d.setAttribute("cx",x);d.setAttribute("cy",y);d.setAttribute("r","2.6");d.setAttribute("fill",r.color);svg.append(d);});
      });
    };
    draws.push(draw);
  });
  chart.append(wrap);
  if(_resize) removeEventListener("resize",_resize);
  _resize=()=>draws.forEach(d=>d()); addEventListener("resize",_resize);
  requestAnimationFrame(()=>setTimeout(()=>draws.forEach(d=>d()),1080));

  const q=n=>DATA[key].find(r=>r.name===n)||{rate_first:0,rate_second:0};
  $("#caption").innerHTML= key==="hide"
    ? `Each bar is sorted by that seat's own hides; every card fans out to its portrait, and slice height is its share. <b>The first player hides royalty far more</b> — it conceals the Queen ${pct(q('Queen').rate_first)} of the time it holds one, versus the second player's ${pct(q('Queen').rate_second)}. Moving first, it protects its bomb; moving second, it plays it.`
    : `What each seat throws away at setup, each sorted its own way. Low-value cards dominate — you keep what threatens and shed what only stalls. Hover any slice for the exact split.`;
}

function render(v){
  const chart=$("#chart"); chart.innerHTML="";
  if(_resize){removeEventListener("resize",_resize);_resize=null;}
  const heads={reason:"How each seat <b>wins</b> — split by who won",
    hide:"What each seat <b>hides</b> at setup — height = share, colour = card",
    discard:"What each seat <b>discards</b> at setup — height = share, colour = card"};
  $("#viewhead").innerHTML=heads[v];
  v==="reason"?renderReason(chart):renderSetup(chart,v);
}
$("#pills").addEventListener("click",e=>{const b=e.target.closest(".pill");if(!b)return;
  [...$("#pills").children].forEach(p=>p.setAttribute("aria-selected",p===b));render(b.dataset.v);});
render("reason");
</script>
"""


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build the second-player-edge headline infographic (HTML).")
    p.add_argument("--stats", default=os.path.join("results", "corpus_stats", "gen25_5k", "selfplay_gen25"))
    p.add_argument("--assets", default="assets", help="card-art + colors directory")
    p.add_argument("--out", default=os.path.join("reports", "headline.html"))
    p.add_argument("--art-width", type=int, default=200)
    args = p.parse_args(argv)
    print(f"wrote {build(args.stats, args.out, args.assets, args.art_width)}")


if __name__ == "__main__":
    main()
