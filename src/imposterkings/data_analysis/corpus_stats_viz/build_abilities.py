"""The Trigger -- do you use a power when the game offers it? An activation funnel per optional ability.

    python -m imposterkings.data_analysis.corpus_stats_viz.build_abilities \
        --stats results/corpus_stats/gen25_5k/selfplay_gen25 --assets assets --out reports/abilities.html

Reads ability_usage.csv (the five MAY abilities: a window opens, the bot activates or declines; if activated
it either lands or is countered by a King's Hand; every terminal splits won/lost). Emits ONE self-contained,
CSP-safe page: a horizontal Sankey funnel per card, sorted by activation rate (Inquisitor 97% -> Sentry 21%).
Won/lost is drawn as a fill-level inside each terminal node. Card arts + colours inlined; no external deps.
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


def _art_data_uris(assets_dir: str, width: int, only: List[str]) -> Dict[str, str]:
    from PIL import Image
    out: Dict[str, str] = {}
    for name in only:
        with Image.open(os.path.join(assets_dir, NAME_ASSET[name])) as im:
            im = im.convert("RGB")
            h = round(im.height * width / im.width)
            im = im.resize((width, h), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=78, optimize=True)
        out[name] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return out


def _card_colors(assets_dir: str) -> Dict[str, str]:
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


# Per-ability wording. The five optional abilities use the defaults; the two reactions are reframed.
_LAB_DEFAULT = {"verb": "activated", "act": "Activated", "landed": "Landed",
                "countered": "Countered", "declined": "Declined"}
_LABELS = {
    "KingsHand": {"verb": "revealed", "act": "Revealed", "landed": "Counter landed",
                  "countered": "Countered", "declined": "Held back"},
    "Assassin": {"verb": "launched", "act": "Launched the kill", "landed": "Killed the king",
                 "countered": "Countered", "declined": "Never launched"},
}


def build(stats_dir: str, out_path: str, assets_dir: str, art_width: int) -> str:
    rows = list(csv.DictReader(open(os.path.join(stats_dir, "ability_usage.csv"), encoding="utf-8")))
    colors = _card_colors(assets_dir)
    cards = []
    for r in rows:
        g = lambda k: int(r[k]) if r[k] not in ("", None) else 0
        win = g("window_opened")
        cards.append({
            "name": r["card"], "val": NAME_VAL.get(r["card"], 0), "color": colors.get(r["card"], "#888"),
            "window": win, "activated": g("activated"), "declined": g("declined"),
            "landed": g("landed"), "countered": g("countered"),
            "landed_won": g("landed_won"), "landed_lost": g("landed_lost"),
            "countered_won": g("countered_won"), "countered_lost": g("countered_lost"),
            "declined_won": g("declined_won"), "declined_lost": g("declined_lost"),
            "forced": g("forced"), "act_rate": g("activated") / win if win else 0.0,
            "lab": _LABELS.get(r["card"], _LAB_DEFAULT),
            # per-seat won/lost for every terminal (P1 = first mover, P2 = second)
            **{f"{o}_{w}_{s}": g(f"{o}_{w}_{s}") for o in ("landed", "countered", "declined")
               for w in ("won", "lost") for s in ("first", "second")},
        })
    cards.sort(key=lambda c: -c["act_rate"])

    # the King's Hand's "Counter landed" arm, decomposed by what it stopped (freq only) -> a treemap
    kh_counters = [{"name": r["countered"], "n": int(r["n"]), "color": colors.get(r["countered"], "#888")}
                   for r in _read(os.path.join(stats_dir, "events_kingshand.csv"))]
    kh_counters.sort(key=lambda d: -d["n"])
    for c in cards:
        if c["name"] == "KingsHand":
            c["counters"] = kh_counters

    names = list(dict.fromkeys([c["name"] for c in cards] + [d["name"] for d in kh_counters]))
    art = _art_data_uris(assets_dir, art_width, names)
    data = {"cards": cards}
    html = _TEMPLATE.replace("/*__DATA__*/", json.dumps(data)).replace("/*__ART__*/", json.dumps(art))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path


_TEMPLATE = r"""
<title>Imposter Kings — do you pull the trigger?</title>
<style>
:root{ --ink:#0c0a06; --panel:#171208; --panel2:#1f1810; --edge:#3a2f1c; --cream:#ece2cc; --muted:#a2916f;
  --faint:#6d5f45; --gold:#e8b44a; --teal:#3fae9b; --crimson:#d1504a; --slate:#8a7a5c; }
*{box-sizing:border-box}
.stage{ max-width:1320px; margin:0 auto; padding:40px 24px 60px; color:var(--cream);
  font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  background:radial-gradient(120% 80% at 50% -10%,#241a0d 0%,transparent 55%),
    radial-gradient(90% 60% at 50% 120%,#1a1207 0%,transparent 60%),var(--ink);
  background-attachment:fixed; min-height:100%; }
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.eyebrow{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;letter-spacing:.28em;
  text-transform:uppercase;color:var(--muted);text-align:center}
h1{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;font-weight:600;text-align:center;
  line-height:1.07;margin:.4em 0 .12em;font-size:clamp(30px,5.6vw,54px);text-wrap:balance;letter-spacing:-.01em}
h1 em{font-style:normal;color:var(--gold)}
.dek{text-align:center;color:var(--muted);max-width:64ch;margin:.5em auto 0;font-size:15px;line-height:1.5}
.dek b{color:var(--cream);font-weight:600}

.key{display:flex;flex-wrap:wrap;gap:10px 22px;justify-content:center;align-items:center;margin:26px auto 6px;
  font-size:12.5px;color:var(--muted)}
.key .it{display:flex;align-items:center;gap:8px}
.key .sw{width:14px;height:14px;border-radius:3px;box-shadow:inset 0 0 0 1px rgba(0,0,0,.35)}
.key .fill{width:34px;height:14px;border-radius:3px;background:linear-gradient(180deg,#e9dcc0 0 45%,#4a4033 45%)}

.funnels{margin-top:14px;display:flex;flex-direction:column;gap:12px}
.fun{display:flex;align-items:stretch;gap:18px;border:1px solid var(--edge);border-radius:14px;
  background:linear-gradient(180deg,rgba(31,24,16,.9),rgba(18,13,7,.85));padding:14px 18px 14px 14px;position:relative}
.fun::before{content:"";position:absolute;top:8px;left:8px;width:13px;height:13px;border:2px solid #7a5a1f;
  border-right:0;border-bottom:0;opacity:.6}
.who{flex:0 0 172px;display:flex;flex-direction:column;align-items:center;text-align:center;justify-content:center;gap:2px}
.who img{width:78px;border-radius:7px;border:1px solid #000;box-shadow:0 4px 14px rgba(0,0,0,.6);margin-bottom:4px}
.who .nm{font-weight:700;letter-spacing:.02em;font-size:15px}
.who .rate{font-family:"Iowan Old Style",Palatino,Georgia,serif;font-size:34px;line-height:1;color:var(--gold);
  text-shadow:0 2px 20px rgba(232,180,74,.3)}
.who .rl{font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.who .win{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--faint);margin-top:3px}
.fun svg{flex:1;min-width:0;width:100%;height:auto;display:block;overflow:visible}
.fun{min-width:0}
.node{cursor:default} .node:hover{filter:brightness(1.15)}
.ntext{font-family:ui-sans-serif,system-ui;fill:var(--cream)}
.nsub{font-family:ui-monospace,Menlo,monospace;fill:var(--muted)}
.stagelab{font-family:ui-monospace,Menlo,monospace;fill:var(--faint);letter-spacing:.12em;text-transform:uppercase}

.caption{color:var(--muted);font-size:13.5px;line-height:1.6;max-width:74ch;margin:22px auto 0;text-align:center}
.caption b{color:var(--gold)}
.foot{margin-top:24px;text-align:center;color:var(--faint);font-size:11px}
.tip{position:fixed;pointer-events:none;z-index:9;background:#0b0906;border:1px solid var(--edge);border-radius:8px;
  padding:9px 11px;font-size:12px;color:var(--cream);box-shadow:0 10px 30px rgba(0,0,0,.6);opacity:0;transition:opacity .12s;max-width:250px}
.tip .tn{font-weight:700;margin-bottom:3px}
.rhs{flex:none;width:340px;display:flex;flex-direction:column;justify-content:center;align-items:center}
.rhs.filled{border-left:1px solid var(--edge);padding-left:16px}
.kh-tree{display:flex;flex-direction:column;gap:9px;align-items:center}
.kh-tt{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);text-align:center}
.kh-tt .mono{color:var(--faint)}
.kh-map{position:relative;border:1px solid var(--edge);border-radius:9px;overflow:hidden;background:#0a0805;
  box-shadow:inset 0 0 22px rgba(0,0,0,.55)}
.tile{position:absolute;overflow:hidden;box-shadow:inset 0 0 0 1px rgba(0,0,0,.55);display:flex;
  align-items:flex-end;justify-content:center;cursor:default;transition:filter .15s}
.tile:hover{filter:brightness(1.2)}
.tile img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;opacity:.92}
.tile .tlab{position:relative;z-index:1;font-size:10px;color:#fff;text-align:center;padding:2px 3px;line-height:1.12;
  text-shadow:0 1px 2px #000,0 0 5px #000,0 0 5px #000}
.tile .tlab b{display:block;font-size:11px}
@media (max-width:720px){ .fun{flex-direction:column} .who{flex-basis:auto;flex-direction:row;gap:12px;justify-content:flex-start}
  .rhs{width:auto} .rhs.filled{border-left:0;border-top:1px solid var(--edge);padding-left:0;padding-top:14px}
  .who img{width:54px;margin:0} .who .rate{font-size:28px} h1{font-size:32px} }
</style>

<div class="stage">
  <div class="eyebrow" id="eyebrow"></div>
  <h1>When the game offers you a power,<br><em>do you take it?</em></h1>
  <p class="dek">Seven abilities are a <b>choice</b> — use the power or hold it. Most can be foiled by a hidden
    <b>King's Hand</b>; the Assassin's kill and the King's Hand's own reveal are funnels here too. Each is sorted
    by how eagerly the bot uses it, and every ending splits into a <b style="color:#6ba2dd">P1</b> and a
    <b style="color:#e8b44a">P2</b> column, each filled to how often <b>that seat won</b> from there.</p>

  <div class="key">
    <div class="it"><span class="sw" style="background:var(--teal)"></span>Landed</div>
    <div class="it"><span class="sw" style="background:var(--crimson)"></span>Countered by King's Hand</div>
    <div class="it"><span class="sw" style="background:var(--slate)"></span>Declined</div>
    <div class="it"><span class="fill"></span>fuller = won more often</div>
    <div class="it"><span class="sw" style="background:transparent;box-shadow:inset 0 0 0 2px #6ba2dd"></span>P1
      <span class="sw" style="background:transparent;box-shadow:inset 0 0 0 2px #e8b44a;margin-left:6px"></span>P2 columns</div>
  </div>

  <div class="funnels" id="funnels"></div>
  <p class="caption" id="caption"></p>
  <div class="foot" id="foot"></div>
</div>
<div class="tip" id="tip"></div>

<script>
const DATA=/*__DATA__*/, ART=/*__ART__*/;
const NS="http://www.w3.org/2000/svg";
const $=(s,r=document)=>r.querySelector(s), el=(t,c)=>{const e=document.createElement(t);if(c)e.className=c;return e;};
const S=(t,a={})=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
const int=n=>n.toLocaleString(), pc=(a,b)=>b?Math.round(100*a/b)+"%":"0%";
const tip=$("#tip");
const showTip=(h,x,y)=>{tip.innerHTML=h;tip.style.opacity=1;tip.style.left=Math.min(x+14,innerWidth-262)+"px";tip.style.top=(y+16)+"px";};
const hideTip=()=>tip.style.opacity=0;
const dark=(hex,f)=>{const n=parseInt(hex.slice(1),16);const r=n>>16,g=n>>8&255,b=n&255;
  const m=v=>Math.round(v*(1-f));return `rgb(${m(r)},${m(g)},${m(b)})`;};

$("#eyebrow").textContent="Imposter Kings · 5,000 self-play games · seven abilities, each a choice";
$("#foot").textContent="ability_usage.csv — window_opened = activated + declined; activated = landed + countered (by a King's Hand). "
  +"Won/lost is the outcome of the game for the seat that faced the window.";

const COL={landed:"var(--teal)",countered:"var(--crimson)",declined:"var(--slate)"};
const RAW={landed:"#3fae9b",countered:"#d1504a",declined:"#8a7a5c"};

// geometry (viewBox units) -- two stages: the CHOICE (activate/decline) -> the OUTCOME, split P1|P2
const W=672,H=236,PY=20,IH=H-2*PY, NW=16, X0=8, X1=452, CW=17, CGAP=8, TW=CW*2+CGAP,
  LX=X1+TW+20, G1=20, G2=13, LBLH=48;
const SEAT={first:"#6ba2dd",second:"#e8b44a"};

function funnel(c){
  const svg=S("svg",{viewBox:`0 0 ${W} ${H}`,preserveAspectRatio:"xMinYMid meet"});
  const win=c.window;
  const actH=c.activated/win*(IH-G1), decH=c.declined/win*(IH-G1);
  const yAct=PY, yDec=PY+actH+G1;
  const hasCnt=c.countered>0, availA=Math.max(2,hasCnt?actH-G2:actH);    // no counter arm for the King's Hand
  const landH=c.landed/c.activated*availA, cntH=hasCnt?c.countered/c.activated*availA:0;
  const yLand=PY, yCnt=PY+landH+G2;

  const ribbon=(t0,b0,t1,b1,col)=>{const x0=X0+NW,x1=X1+TW,xm=(x0+x1)/2;   // reach behind BOTH seat columns
    svg.append(S("path",{d:`M${x0},${t0} C${xm},${t0} ${xm},${t1} ${x1},${t1} L${x1},${b1} C${xm},${b1} ${xm},${b0} ${x0},${b0} Z`,
      fill:col,opacity:.24}));};
  ribbon(yAct, yAct+landH, yLand, yLand+landH, RAW.landed);              // Activated -> Landed
  if(hasCnt) ribbon(yAct+landH+G2, yAct+actH, yCnt, yCnt+cntH, RAW.countered);   // Activated -> Countered (skip if none)
  ribbon(yDec, yDec+decH, yDec, yDec+decH, RAW.declined);               // Declined -> Declined

  // col0 source nodes: the choice
  const source=(y,h,fill,nm,ts)=>{const g=S("g",{class:"node"});
    g.append(S("rect",{x:X0,y,width:NW,height:Math.max(1,h),rx:2,fill,stroke:"rgba(0,0,0,.4)","stroke-width":.5}));
    g.onmousemove=e=>showTip(`<div class="tn" style="color:${fill}">${nm}</div><div class="mono" style="color:var(--muted)">${ts}</div>`,e.clientX,e.clientY);
    g.onmouseleave=hideTip; svg.append(g);};
  source(yAct,actH,"#c99a42",c.lab.act,`${int(c.activated)}× — ${pc(c.activated,win)} of ${int(win)} chances${c.forced?` · ${int(c.forced)} forced`:""}`);
  source(yDec,decH,"#6c6046",c.lab.declined,`${int(c.declined)}× — ${pc(c.declined,win)} of ${int(win)} chances`);

  // col1 terminals: each is TWO columns, P1 (blue border) and P2 (gold border), filled to that seat's winrate
  const tot4=o=>(c[o+"_won_first"]||0)+(c[o+"_lost_first"]||0)+(c[o+"_won_second"]||0)+(c[o+"_lost_second"]||0);
  const terms=[
    {y:yLand,h:landH,key:"landed",name:c.lab.landed,o:"landed"},
    {y:yCnt,h:cntH,key:"countered",name:c.lab.countered,o:"countered"},
    {y:yDec,h:decH,key:"declined",name:c.lab.declined,o:"declined"},
  ].filter(t=>tot4(t.o)>0);                                              // drop an empty arm (KH has no counter)
  const seatWL=(t,s)=>[c[`${t.o}_won_${s}`]||0, c[`${t.o}_lost_${s}`]||0];
  terms.forEach(t=>{
    const g=S("g",{class:"node"});
    [["first",X1],["second",X1+CW+CGAP]].forEach(([s,x])=>{
      const [won,lost]=seatWL(t,s), tot=won+lost, wonH=tot?t.h*won/tot:0;
      g.append(S("rect",{x,y:t.y,width:CW,height:Math.max(1,t.h),rx:2,fill:dark(RAW[t.key],.64)}));       // lost
      if(wonH>0.4) g.append(S("rect",{x,y:t.y+t.h-wonH,width:CW,height:wonH,rx:2,fill:RAW[t.key]}));        // won, rising from the floor
      g.append(S("rect",{x,y:t.y,width:CW,height:Math.max(1,t.h),rx:2,fill:"none",stroke:SEAT[s],"stroke-width":1.2,opacity:.85}));
    });
    const [w1,l1]=seatWL(t,"first"), [w2,l2]=seatWL(t,"second");
    g.onmousemove=e=>showTip(`<div class="tn" style="color:${RAW[t.key]}">${t.name}</div>`
      +`<div class="mono"><span style="color:${SEAT.first}">P1 ${pc(w1,w1+l1)}</span> won of ${int(w1+l1)}</div>`
      +`<div class="mono"><span style="color:${SEAT.second}">P2 ${pc(w2,w2+l2)}</span> won of ${int(w2+l2)}</div>`,e.clientX,e.clientY);
    g.onmouseleave=hideTip; svg.append(g); t.cy=t.y+t.h/2; t.w1=w1;t.l1=l1;t.w2=w2;t.l2=l2;});

  // de-overlap the three right-hand labels: desired y = node centre, then keep >= LBLH apart within bounds
  const minY=PY+14, maxY=H-18, L=terms.slice().sort((a,b)=>a.cy-b.cy);
  L[0].ly=Math.max(minY,L[0].cy);
  for(let i=1;i<L.length;i++) L[i].ly=Math.max(L[i].cy,L[i-1].ly+LBLH);
  L[L.length-1].ly=Math.min(L[L.length-1].ly,maxY);
  for(let i=L.length-2;i>=0;i--) L[i].ly=Math.min(L[i].ly,L[i+1].ly-LBLH);
  L[0].ly=Math.max(L[0].ly,minY);
  for(let i=1;i<L.length;i++) L[i].ly=Math.max(L[i].ly,L[i-1].ly+LBLH);
  terms.forEach(t=>{
    svg.append(S("polyline",{points:`${X1+TW},${t.cy} ${LX-12},${t.cy} ${LX-6},${t.ly}`,fill:"none",stroke:RAW[t.key],"stroke-width":1,opacity:.5}));
    const nm=S("text",{x:LX,y:t.ly-8,class:"ntext","font-size":15,"font-weight":600}); nm.textContent=t.name; svg.append(nm);
    const s1=S("text",{x:LX,y:t.ly+9,"font-size":13,"font-weight":600,fill:SEAT.first}); s1.textContent=`P1 ${pc(t.w1,t.w1+t.l1)}`; svg.append(s1);
    const s2=S("text",{x:LX+82,y:t.ly+9,"font-size":13,"font-weight":600,fill:SEAT.second}); s2.textContent=`P2 ${pc(t.w2,t.w2+t.l2)}`; svg.append(s2);
    const n1=S("text",{x:LX,y:t.ly+22,"font-size":10.5,fill:"var(--faint)"}); n1.textContent=`${int(t.w1+t.l1)} games`; svg.append(n1);
    const n2=S("text",{x:LX+82,y:t.ly+22,"font-size":10.5,fill:"var(--faint)"}); n2.textContent=`${int(t.w2+t.l2)} games`; svg.append(n2);});

  // tiny P1/P2 headers over the two sub-columns
  svg.append(Object.assign(S("text",{x:X1+CW/2,y:11,"font-size":9,"text-anchor":"middle",fill:SEAT.first}),{textContent:"P1"}));
  svg.append(Object.assign(S("text",{x:X1+CW+CGAP+CW/2,y:11,"font-size":9,"text-anchor":"middle",fill:SEAT.second}),{textContent:"P2"}));
  return svg;
}

// squarified treemap (Bruls et al.): partition a W×H box into cells with area ∝ n, kept near-square
function squarify(items, W, H){
  const nodes=items.map(d=>({d,v:d.n})), total=nodes.reduce((s,n)=>s+n.v,0)||1;
  nodes.forEach(n=>n.v=n.v/total*(W*H));
  const worst=(row,side)=>{const s=row.reduce((a,n)=>a+n.v,0);
    const mx=Math.max(...row.map(n=>n.v)),mn=Math.min(...row.map(n=>n.v));
    return Math.max(side*side*mx/(s*s), s*s/(side*side*mn));};
  const out=[]; let x=0,y=0,w=W,h=H,i=0;
  while(i<nodes.length){
    let row=[nodes[i]]; const side=Math.min(w,h);
    while(i+row.length<nodes.length){const nx=nodes[i+row.length];
      if(worst(row,side)>=worst(row.concat([nx]),side)) row.push(nx); else break;}
    const rs=row.reduce((a,n)=>a+n.v,0);
    if(w>=h){const cw=rs/h; let cy=y; row.forEach(n=>{const ch=n.v/cw; out.push({d:n.d,x,y:cy,w:cw,h:ch}); cy+=ch;}); x+=cw; w-=cw;}
    else    {const ch=rs/w; let cx=x; row.forEach(n=>{const cw=n.v/ch; out.push({d:n.d,x:cx,y,w:cw,h:ch}); cx+=cw;}); y+=ch; h-=ch;}
    i+=row.length;
  }
  return out;
}
function tree(c, boxH){
  const TW=320, TH=Math.max(90,Math.round(boxH)-32), tot=c.counters.reduce((a,d)=>a+d.n,0);
  const box=el("div","kh-tree");
  box.innerHTML=`<div class="kh-tt">Counter lands on <span class="mono">${int(tot)} cards</span></div>`;
  const map=el("div","kh-map"); map.style.width=TW+"px"; map.style.height=TH+"px";
  squarify(c.counters,TW,TH).forEach(r=>{
    const t=el("div","tile");
    t.style.cssText=`left:${r.x}px;top:${r.y}px;width:${r.w}px;height:${r.h}px;background:${r.d.color}`;
    t.innerHTML=`<img src="${ART[r.d.name]}" alt="${r.d.name}"><span class="tlab"><b>${r.d.name}</b>${int(r.d.n)}</span>`;
    t.onmousemove=e=>showTip(`<div class="tn" style="color:${r.d.color}">${r.d.name}</div><div class="mono" style="color:var(--muted)">countered ${int(r.d.n)}× · ${pc(r.d.n,tot)} of the King's Hand's reveals</div>`,e.clientX,e.clientY);
    t.onmouseleave=hideTip; map.append(t);
  });
  box.append(map); return box;
}

const wrap=$("#funnels");
const khRows=[];                           // King's Hand rows: size their treemap to the funnel height (below)
DATA.cards.forEach(c=>{
  const row=el("div","fun");
  const who=el("div","who");
  who.innerHTML=`<img src="${ART[c.name]}" alt="${c.name}"><div class="nm">${c.name}</div>`
    +`<div class="rate">${Math.round(c.act_rate*100)}%</div><div class="rl">${c.lab.verb}</div>`
    +`<div class="win">${int(c.window)} chances · v${c.val}</div>`;
  const fsvg=funnel(c);
  const rhs=el("div","rhs");               // a reserved right column on EVERY row keeps all funnels one width
  if(c.counters){ rhs.classList.add("filled"); khRows.push({rhs,c,svg:fsvg}); }   // KH: treemap built once laid out
  row.append(who, fsvg, rhs);
  wrap.append(row);
});

// The treemap is the only child tall enough to stretch its row. Build it to the funnel's own rendered
// height so the King's Hand row matches every other row instead of towering over them. (On mobile the
// row stacks vertically, so keep the treemap tall there.)
function fitTrees(){
  const wide=innerWidth>720;
  khRows.forEach(r=>{
    const h=wide?r.svg.getBoundingClientRect().height:300;
    if(!h) return;
    r.rhs.innerHTML="";
    r.rhs.appendChild(tree(r.c,h));
  });
}
requestAnimationFrame(fitTrees);
addEventListener("resize",fitTrees);

$("#caption").innerHTML="The spread is a lesson in discipline. <b>The Inquisitor fires on sight (97%)</b> — its guess is free "
  +"information, so there's no reason to hold it — while the <b>Sentry is hoarded (21%)</b>. The <b>King's Hand is revealed only "
  +"about half the times it could be</b>, kept back as a threat; and the <b>Assassin's kill wins outright whenever it lands</b> (both "
  +"columns full). But look down every funnel: <b>the gold P2 column is fuller than the blue P1 column almost everywhere</b> — the same "
  +"power, in the same spot, is simply worth more to whoever moves second. The ~59/41 seat edge, running through every ability.";
</script>
"""


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build the ability-activation funnel infographic (HTML).")
    p.add_argument("--stats", default=os.path.join("results", "corpus_stats", "gen25_5k", "selfplay_gen25"))
    p.add_argument("--assets", default="assets")
    p.add_argument("--out", default=os.path.join("reports", "abilities.html"))
    p.add_argument("--art-width", type=int, default=200)
    args = p.parse_args(argv)
    print(f"wrote {build(args.stats, args.out, args.assets, args.art_width)}")


if __name__ == "__main__":
    main()
