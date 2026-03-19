<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>InkCluster Hierarchical Ink Recognition</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#060608;
  --glass:rgba(255,255,255,0.03);
  --glass2:rgba(255,255,255,0.06);
  --border:rgba(255,255,255,0.07);
  --border2:rgba(255,255,255,0.12);
  --text:#f0ede8;
  --muted:#6b6878;
  --dim:#13121a;
  --ink:#e8e4ff;
  --c1:#64ffda;   /* char — mint */
  --c2:#ff9f43;   /* word — amber */
  --c3:#a29bfe;   /* phrase — lavender */
  --c1a:rgba(100,255,218,0.08);
  --c2a:rgba(255,159,67,0.08);
  --c3a:rgba(162,155,254,0.08);
  --mono:'Space Mono',monospace;
  --sans:'Space Grotesk',sans-serif;
}
body{
  background:var(--bg);color:var(--text);font-family:var(--sans);
  height:100vh;overflow:hidden;display:flex;flex-direction:column;
}

/* subtle noise texture */
body::before{
  content:'';position:fixed;inset:0;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
  pointer-events:none;z-index:0;opacity:.4;
}

/* ── TOPBAR ── */
.bar{
  position:relative;z-index:10;
  display:flex;align-items:center;padding:0 20px;height:52px;
  border-bottom:1px solid var(--border);
  background:rgba(6,6,8,0.9);
  backdrop-filter:blur(20px);
  flex-shrink:0;gap:0;user-select:none;
}

.logo{
  display:flex;align-items:center;gap:10px;margin-right:24px;flex-shrink:0;
}
.logo-mark{
  width:28px;height:28px;border-radius:6px;
  background:linear-gradient(135deg,var(--c1),var(--c3));
  display:flex;align-items:center;justify-content:center;
  font-size:13px;box-shadow:0 0 20px rgba(100,255,218,0.2);
}
.logo-text{
  font-family:var(--mono);font-size:12px;font-weight:700;
  letter-spacing:.08em;color:var(--text);
}
.logo-sub{
  font-family:var(--mono);font-size:9px;color:var(--muted);
  letter-spacing:.1em;text-transform:uppercase;margin-top:1px;
}

.sep{width:1px;height:24px;background:var(--border);margin:0 14px;flex-shrink:0}
.sp{flex:1}

/* tool buttons */
.tg{display:flex;gap:3px;flex-shrink:0}
.btn{
  font-family:var(--mono);font-size:10px;letter-spacing:.06em;
  padding:6px 13px;border:1px solid var(--border2);
  background:var(--glass);color:var(--muted);border-radius:5px;
  cursor:pointer;transition:all .18s;text-transform:uppercase;white-space:nowrap;
}
.btn:hover{border-color:rgba(255,255,255,0.25);color:var(--text);background:var(--glass2)}
.btn.on{background:var(--c1a);border-color:var(--c1);color:var(--c1)}
.btn.run{border-color:var(--c3);color:var(--c3)}
.btn.run:hover{background:var(--c3);color:var(--bg)}
.btn.clr:hover{border-color:#ff6b8a;color:#ff6b8a}
.btn.scatter{border-color:var(--c2);color:var(--c2)}
.btn.scatter:hover{background:var(--c2);color:var(--bg)}

/* slider controls */
.ctrl{display:flex;align-items:center;gap:7px;flex-shrink:0}
.clabel{font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:.08em;white-space:nowrap}
.cval{font-family:var(--mono);font-size:10px;color:var(--c3);min-width:28px}
input[type=range]{
  -webkit-appearance:none;width:70px;height:2px;
  background:var(--border2);border-radius:1px;outline:none;
}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;width:12px;height:12px;border-radius:50%;
  background:var(--c3);cursor:pointer;box-shadow:0 0 8px rgba(162,155,254,.5);
}

/* status chips */
.chip{
  font-family:var(--mono);font-size:10px;padding:4px 10px;
  border-radius:20px;background:var(--glass2);
  border:1px solid var(--border);color:var(--muted);
  letter-spacing:.05em;margin-left:6px;flex-shrink:0;
  transition:all .3s;
}
.chip span{color:var(--text);font-weight:700}
.chip.live{border-color:var(--c3);background:var(--c3a);color:var(--c3)}

/* ── CANVAS AREA ── */
.body{display:flex;flex:1;overflow:hidden;position:relative}
.canvas-wrap{
  flex:1;position:relative;overflow:hidden;
  cursor:crosshair;
}
canvas{position:absolute;inset:0;width:100%;height:100%;display:block}

/* ── RIGHT PANEL ── */
.panel{
  width:260px;border-left:1px solid var(--border);
  background:rgba(6,6,8,0.7);backdrop-filter:blur(20px);
  display:flex;flex-direction:column;overflow:hidden;flex-shrink:0;
  position:relative;z-index:5;
}

.psec{padding:16px;border-bottom:1px solid var(--border)}

.ptitle{
  font-family:var(--mono);font-size:9px;letter-spacing:.18em;
  text-transform:uppercase;color:var(--muted);margin-bottom:12px;
  display:flex;align-items:center;gap:6px;
}
.ptitle::after{content:'';flex:1;height:1px;background:var(--border)}

/* hierarchy levels display */
.levels{display:flex;flex-direction:column;gap:6px}
.level{
  display:flex;align-items:center;gap:10px;
  padding:9px 12px;border-radius:6px;
  border:1px solid transparent;transition:all .3s;
}
.level.l1{background:var(--c1a);border-color:rgba(100,255,218,.15)}
.level.l2{background:var(--c2a);border-color:rgba(255,159,67,.15)}
.level.l3{background:var(--c3a);border-color:rgba(162,155,254,.15)}
.level-icon{width:20px;height:20px;border-radius:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:10px}
.level.l1 .level-icon{background:rgba(100,255,218,.15);color:var(--c1)}
.level.l2 .level-icon{background:rgba(255,159,67,.15);color:var(--c2)}
.level.l3 .level-icon{background:rgba(162,155,254,.15);color:var(--c3)}
.level-info{flex:1}
.level-name{font-family:var(--mono);font-size:10px;font-weight:700;margin-bottom:1px}
.level.l1 .level-name{color:var(--c1)}
.level.l2 .level-name{color:var(--c2)}
.level.l3 .level-name{color:var(--c3)}
.level-desc{font-size:10px;color:var(--muted)}
.level-count{font-family:var(--mono);font-size:16px;font-weight:700}
.level.l1 .level-count{color:var(--c1)}
.level.l2 .level-count{color:var(--c2)}
.level.l3 .level-count{color:var(--c3)}

/* metrics */
.mrow{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.mkey{font-family:var(--mono);font-size:10px;color:var(--muted)}
.mval{font-family:var(--mono);font-size:11px;font-weight:600;color:var(--text)}
.mval.v3{color:var(--c3)}

/* pipeline steps */
.pipe{display:flex;flex-direction:column;gap:1px}
.pstep{
  display:flex;align-items:center;gap:8px;
  padding:6px 8px;border-radius:4px;
  font-family:var(--mono);font-size:10px;color:var(--muted);
  transition:all .25s;
}
.pstep.done{color:var(--text);background:var(--glass)}
.pstep.active{color:var(--c3);background:var(--c3a)}
.pdot{width:6px;height:6px;border-radius:50%;background:var(--border2);flex-shrink:0;transition:all .25s}
.pstep.done .pdot{background:var(--text)}
.pstep.active .pdot{background:var(--c3);box-shadow:0 0 6px var(--c3)}
.parr{font-size:8px;color:var(--border);padding-left:14px}

/* phrase list */
.plist{flex:1;overflow-y:auto;padding:10px}
.plist::-webkit-scrollbar{width:3px}
.plist::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
.pitem{
  display:flex;align-items:center;gap:8px;
  padding:7px 9px;border-radius:5px;
  margin-bottom:3px;cursor:pointer;
  border:1px solid transparent;transition:all .15s;
}
.pitem:hover{background:var(--glass2);border-color:var(--border)}
.pitem.sel{background:var(--c3a);border-color:rgba(162,155,254,.3)}
.pswatch{width:8px;height:8px;border-radius:2px;flex-shrink:0}
.pname{font-family:var(--mono);font-size:11px;flex:1}
.pmeta{font-family:var(--mono);font-size:9px;color:var(--muted)}

/* ── OVERLAYS ── */
.mode-badge{
  position:absolute;top:14px;left:14px;z-index:5;
  font-family:var(--mono);font-size:9px;letter-spacing:.12em;
  text-transform:uppercase;padding:4px 10px;border-radius:4px;
  pointer-events:none;transition:all .2s;
}
.mode-badge.draw{background:rgba(100,255,218,.08);color:var(--c1);border:1px solid rgba(100,255,218,.2)}
.mode-badge.pan{background:rgba(255,159,67,.08);color:var(--c2);border:1px solid rgba(255,159,67,.2)}

.help-overlay{
  position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  text-align:center;pointer-events:none;transition:opacity .5s;z-index:3;
}
.help-big{
  font-family:var(--mono);font-size:13px;color:rgba(255,255,255,.15);
  letter-spacing:.12em;margin-bottom:8px;
}
.help-small{font-family:var(--mono);font-size:10px;color:rgba(255,255,255,.07);letter-spacing:.08em}

/* alpha legend overlay on canvas */
.legend-overlay{
  position:absolute;bottom:16px;left:16px;z-index:5;
  display:flex;flex-direction:column;gap:5px;pointer-events:none;
}
.leg-item{
  display:flex;align-items:center;gap:7px;
  background:rgba(6,6,8,.7);backdrop-filter:blur(10px);
  padding:5px 10px;border-radius:4px;border:1px solid var(--border);
}
.leg-dot{width:8px;height:8px;border-radius:2px;flex-shrink:0}
.leg-txt{font-family:var(--mono);font-size:9px;letter-spacing:.08em}

/* status bar */
.sbar{
  height:28px;border-top:1px solid var(--border);
  background:rgba(6,6,8,.9);
  display:flex;align-items:center;padding:0 16px;gap:16px;flex-shrink:0;
  position:relative;z-index:10;
}
.sitem{font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:.06em}
.sitem b{color:var(--text);font-weight:600}

/* toast */
.toast{
  position:fixed;bottom:40px;left:50%;
  transform:translateX(-50%) translateY(12px);
  background:var(--c3);color:var(--bg);
  font-family:var(--mono);font-size:10px;font-weight:700;
  padding:7px 18px;border-radius:4px;letter-spacing:.05em;
  opacity:0;transition:all .22s;pointer-events:none;z-index:100;
}
.toast.on{opacity:1;transform:translateX(-50%) translateY(0)}

/* recognition text style — used in canvas drawRecogText */
.rec-badge{
  position:absolute;
  font-family:var(--sans);font-size:11px;
  background:rgba(6,6,8,.85);backdrop-filter:blur(8px);
  border:1px solid var(--c3);color:var(--c3);
  padding:3px 8px;border-radius:3px;pointer-events:none;z-index:6;
  white-space:nowrap;opacity:0;transition:opacity .2s;
  letter-spacing:.02em;
}
.rec-badge.show{opacity:1}
.chip.model-ok{border-color:var(--c1);background:var(--c1a);color:var(--c1)}
</style>
</head>
<body>

<!-- TOP BAR -->
<div class="bar">
  <div class="logo">
    <div class="logo-mark">✦</div>
    <div>
      <div class="logo-text">INKCLUSTER</div>
      <div class="logo-sub">Hierarchical</div>
    </div>
  </div>
  <div class="sep"></div>
  <div class="tg">
    <button class="btn on" id="bDraw" onclick="setMode('draw')">✏ Draw</button>
    <button class="btn"    id="bPan"  onclick="setMode('pan')">⊹ Pan</button>
  </div>
  <div class="sep"></div>
  <div class="tg">
    <button class="btn run" onclick="runHier()">⬡ Cluster</button>
    <button class="btn scatter" onclick="addScale()">⊕ Scale Demo</button>
    <button class="btn clr" onclick="clearAll()">⊘ Clear</button>
  </div>
  <div class="sep"></div>
  <div class="chip" id="chipModel">Server <span>not connected</span></div>
  <div class="sep"></div>
  <div class="ctrl">
    <span class="clabel">α</span>
    <input type="range" id="sA" min="8" max="30" value="15" oninput="updA()">
    <span class="cval" id="vA">1.5</span>
  </div>
  <div class="sp"></div>
  <div class="chip" id="chipStr">Strokes <span id="nStr">0</span></div>
  <div class="chip" id="chipResult" style="display:none"></div>
</div>

<!-- BODY -->
<div class="body">

  <!-- CANVAS -->
  <div class="canvas-wrap" id="cWrap">
    <canvas id="cv"></canvas>

    <div class="mode-badge draw" id="mBadge">Draw Mode</div>

    <div class="help-overlay" id="helpEl">
      <div class="help-big">Draw anything · then Cluster</div>
      <div class="help-small">Pan with middle mouse or Pan mode · Scroll to zoom</div>
    </div>

    <div class="legend-overlay">
      <div class="leg-item"><div class="leg-dot" style="background:var(--c3)"></div><span class="leg-txt" style="color:var(--c3)">Phrase</span></div>
      <div class="leg-item"><div class="leg-dot" style="background:var(--c2)"></div><span class="leg-txt" style="color:var(--c2)">Word</span></div>
      <div class="leg-item"><div class="leg-dot" style="background:var(--c1)"></div><span class="leg-txt" style="color:var(--c1)">Character</span></div>
    </div>
  </div>

  <!-- PANEL -->
  <div class="panel">

    <div class="psec">
      <div class="ptitle">Hierarchy</div>
      <div class="levels">
        <div class="level l3">
          <div class="level-icon">P</div>
          <div class="level-info">
            <div class="level-name">Phrases</div>
            <div class="level-desc">Sentences / lines</div>
          </div>
          <div class="level-count" id="nPhrases">—</div>
        </div>
        <div class="level l2">
          <div class="level-icon">W</div>
          <div class="level-info">
            <div class="level-name">Words</div>
            <div class="level-desc">Spatial word groups</div>
          </div>
          <div class="level-count" id="nWords">—</div>
        </div>
        <div class="level l1">
          <div class="level-icon">C</div>
          <div class="level-info">
            <div class="level-name">Characters</div>
            <div class="level-desc">Stroke clusters</div>
          </div>
          <div class="level-count" id="nChars">—</div>
        </div>
      </div>
    </div>

    <div class="psec">
      <div class="ptitle">Metrics</div>
      <div class="mrow"><span class="mkey">strokes</span><span class="mval" id="mStr">0</span></div>
      <div class="mrow"><span class="mkey">alpha (α)</span><span class="mval v3" id="mA">1.5</span></div>
      <div class="mrow"><span class="mkey">cluster time</span><span class="mval v3" id="mTime">—</span></div>
      <div class="mrow"><span class="mkey">zoom</span><span class="mval" id="mZoom">100%</span></div>
    </div>

    <div class="psec">
      <div class="ptitle">Pipeline</div>
      <div class="pipe">
        <div class="pstep" id="pRaw"><div class="pdot"></div>Raw Strokes</div>
        <div class="parr">↓</div>
        <div class="pstep" id="pPrep"><div class="pdot"></div>Preprocess</div>
        <div class="parr">↓</div>
        <div class="pstep" id="pKD"><div class="pdot"></div>KD-Tree Index</div>
        <div class="parr">↓</div>
        <div class="pstep" id="pH1"><div class="pdot"></div>L1: strokes → chars</div>
        <div class="parr">↓</div>
        <div class="pstep" id="pH2"><div class="pdot"></div>L2: chars → words</div>
        <div class="parr">↓</div>
        <div class="pstep" id="pH3"><div class="pdot"></div>L3: words → phrases</div>
      </div>
    </div>

    <div class="ptitle" style="padding:12px 16px 4px;font-family:var(--mono);font-size:9px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted)">Phrases</div>
    <div class="plist" id="pList">
      <div style="font-family:var(--mono);font-size:10px;color:var(--muted);padding:6px;text-align:center">Run Cluster first</div>
    </div>

  </div>
</div>

<!-- STATUS BAR -->
<div class="sbar">
  <div class="sitem">Canvas <b>∞ × ∞</b></div>
  <div class="sitem">Cursor <b id="sCursor">—</b></div>
  <div class="sitem" style="margin-left:auto">α₁ = α₂ = α &nbsp;·&nbsp; α₃ = α × 1.3 &nbsp;·&nbsp; k-NN sym=min</div>
</div>

<div class="toast" id="toast"></div>

<script>
// ── PALETTE ──
const PAL=['#64ffda','#ff9f43','#a29bfe','#ffd166','#ff6b9d','#4ecdc4','#a8ff78','#fd79a8','#0abde3','#f8b739','#26de81','#45aaf2'];
const C1='#64ffda',C2='#ff9f43',C3='#a29bfe';

// ── MODEL STATE ──
const SERVER = 'http://localhost:5001';
let serverReady = false;
let charText   = {};
let wordText   = {};
let phraseText = {};

// ── CHECK SERVER ON LOAD ──
async function checkServer(){
  try {
    const r = await fetch(`${SERVER}/health`, {signal: AbortSignal.timeout(2000)});
    const d = await r.json();
    if(d.model_loaded){
      serverReady = true;
      const chip = document.getElementById('chipModel');
      chip.className = 'chip model-ok';
      chip.innerHTML = `Model <span>ready</span> · vocab ${d.vocab_size}`;
      toast('Model server connected ✓');
    } else {
      setServerChip('server up, model not loaded');
    }
  } catch(e){
    setServerChip('not connected');
  }
}

function setServerChip(msg){
  const chip = document.getElementById('chipModel');
  chip.className = 'chip';
  chip.innerHTML = `Server <span>${msg}</span>`;
}

// Poll every 3s until connected
setInterval(()=>{ if(!serverReady) checkServer(); }, 3000);
setTimeout(checkServer, 500);

// ── STATE ──
let strokes=[],cur=null,drawing=false,mode='draw';
let alpha=1.5,sid=0;
let hC=null,hW=null,hP=null; // chars, words, phrases
let selPhrase=null;
let vpX=0,vpY=0,vpZ=1,pan=null;

const DPR=()=>window.devicePixelRatio||1;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const wrap=document.getElementById('cWrap');

function CW(){return cv.width/DPR()} function CH(){return cv.height/DPR()}

function resize(){
  const d=DPR();
  cv.width=wrap.clientWidth*d;cv.height=wrap.clientHeight*d;
  cv.style.width=wrap.clientWidth+'px';cv.style.height=wrap.clientHeight+'px';
  ctx.scale(d,d);draw();
}
window.addEventListener('resize',resize);

const s2c=(sx,sy)=>({x:(sx-vpX)/vpZ,y:(sy-vpY)/vpZ});
const c2s=(cx,cy)=>({x:cx*vpZ+vpX,y:cy*vpZ+vpY});

// ── INPUT ──
function gp(e){const r=wrap.getBoundingClientRect(),s=e.touches?e.touches[0]:e;return{sx:s.clientX-r.left,sy:s.clientY-r.top}}

wrap.addEventListener('mousedown',e=>dn(e));
wrap.addEventListener('mousemove',e=>mv(e));
wrap.addEventListener('mouseup',up);
wrap.addEventListener('mouseleave',up);
wrap.addEventListener('touchstart',e=>{e.preventDefault();dn(e)},{passive:false});
wrap.addEventListener('touchmove', e=>{e.preventDefault();mv(e)},{passive:false});
wrap.addEventListener('touchend',  e=>{e.preventDefault();up()},{passive:false});
wrap.addEventListener('wheel',e=>{
  e.preventDefault();
  const{sx,sy}=gp(e),f=e.deltaY<0?1.12:0.9;
  vpX=sx-(sx-vpX)*f;vpY=sy-(sy-vpY)*f;
  vpZ=Math.max(.04,Math.min(12,vpZ*f));
  document.getElementById('mZoom').textContent=Math.round(vpZ*100)+'%';
  draw();
},{passive:false});

// Middle mouse pan
wrap.addEventListener('mousedown',e=>{if(e.button===1){e.preventDefault();pan={sx:e.clientX,sy:e.clientY,vpX,vpY};}});
window.addEventListener('mouseup',e=>{if(e.button===1)pan=null;});

function dn(e){
  const{sx,sy}=gp(e);
  if(mode==='pan'){pan={sx,sy,vpX,vpY};return}
  if(e.button===1)return;
  drawing=true;hC=hW=hP=null;selPhrase=null;
  cur={points:[s2c(sx,sy)]};
  document.getElementById('helpEl').style.opacity='0';
}
function mv(e){
  const{sx,sy}=gp(e),cp=s2c(sx,sy);
  document.getElementById('sCursor').textContent=Math.round(cp.x)+', '+Math.round(cp.y);
  if(pan){vpX=pan.vpX+(sx-pan.sx);vpY=pan.vpY+(sy-pan.sy);draw();return}
  if(!drawing||!cur)return;
  cur.points.push(cp);draw();
}
function up(){
  pan=null;
  if(!drawing)return;drawing=false;
  if(cur&&cur.points.length>1){
    strokes.push({points:cur.points,id:sid++,col:C1,level:'raw'});
    updCount();pipe(['raw']);
  }
  cur=null;draw();
}

// ── ALGORITHMS ──
function resamp(pts,n=32){
  if(pts.length<2)return pts;
  const d=[0];for(let i=1;i<pts.length;i++)d.push(d[i-1]+Math.hypot(pts[i].x-pts[i-1].x,pts[i].y-pts[i-1].y));
  const T=d[d.length-1];if(!T)return pts;
  return Array.from({length:n},(_,k)=>{
    const t=(k/(n-1))*T,i=Math.max(1,d.findIndex(v=>v>=t));
    const u=(t-d[i-1])/(d[i]-d[i-1]+1e-9);
    return{x:pts[i-1].x+u*(pts[i].x-pts[i-1].x),y:pts[i-1].y+u*(pts[i].y-pts[i-1].y)};
  });
}
function smth(pts,w=2){return pts.map((_,i)=>{let sx=0,sy=0,c=0;for(let k=Math.max(0,i-w);k<=Math.min(pts.length-1,i+w);k++){sx+=pts[k].x;sy+=pts[k].y;c++;}return{x:sx/c,y:sy/c};})}
function ctr(pts){const xs=pts.map(p=>p.x),ys=pts.map(p=>p.y);return{x:(Math.min(...xs)+Math.max(...xs))/2,y:(Math.min(...ys)+Math.max(...ys))/2}}
function prep(pts){return smth(resamp(pts,32),2)}

function nnEps(cs,a,k=1){
  const n=cs.length;if(n<=k)return new Array(n).fill(12);
  return cs.map((c,i)=>{
    const ds=cs.map((o,j)=>j!==i?Math.hypot(o.x-c.x,o.y-c.y):Infinity).sort((a,b)=>a-b).slice(0,k);
    return Math.max(8,a*ds.reduce((s,v)=>s+v,0)/ds.length);
  });
}
function dbscan(cs,ea){
  const n=cs.length,L=new Array(n).fill(-1),vis=new Array(n).fill(false);let cid=0;
  const nb=i=>{const r=[];for(let j=0;j<n;j++){if(j===i)continue;if(Math.hypot(cs[j].x-cs[i].x,cs[j].y-cs[i].y)<=Math.min(ea[i],ea[j]))r.push(j);}return r;};
  for(let i=0;i<n;i++){
    if(vis[i])continue;vis[i]=true;L[i]=cid;
    const S=[...nb(i)];
    for(let si=0;si<S.length;si++){const q=S[si];if(!vis[q]){vis[q]=true;nb(q).forEach(j=>{if(!S.includes(j)&&j!==q)S.push(j);});}if(L[q]===-1)L[q]=cid;}
    cid++;
  }
  return L;
}

// ── MYSCRIPT API CONFIG ──
// Get free keys at developer.myscript.com
const MS_APP_KEY  = 'bf786719-2152-4641-bf5f-7ad282e8d585';
const MS_HMAC_KEY = '92b040fc-3c51-46ce-8870-202dfd888d7d';
const MS_URL      = 'https://cloud.myscript.com/api/v4.0/iink/batch';

let msReady = MS_APP_KEY !== 'YOUR_APPLICATION_KEY';

function updateModelChip2(){
  const chip = document.getElementById('chipModel');
  if(msReady){
    chip.className = 'chip model-ok';
    chip.innerHTML = 'MyScript <span>ready</span>';
  } else {
    chip.className = 'chip';
    chip.innerHTML = 'MyScript <span>add keys</span>';
  }
}
updateModelChip2();

// Convert stroke points to MyScript stroke format
function toMSStrokes(strokePtsArrays){
  return strokePtsArrays.map(pts => ({
    x: pts.map(p => Math.round(p.x)),
    y: pts.map(p => Math.round(p.y)),
    t: pts.map((_,i) => i * 20),  // fake timestamps 20ms apart
  }));
}

// Call MyScript REST API for one cluster
async function recogniseClusterMS(strokePtsArrays, type='Text'){
  if(!msReady) return '';
  if(!strokePtsArrays || !strokePtsArrays.length) return '';

  const body = {
    contentType: type,  // 'Text' or 'Math'
    strokeGroups: [{
      strokes: toMSStrokes(strokePtsArrays)
    }]
  };

  try {
    const r = await fetch(MS_URL, {
      method: 'POST',
      headers: {
        'Accept':           'application/json,application/vnd.myscript.jiix',
        'Content-Type':     'application/json',
        'applicationKey':   MS_APP_KEY,
        'hmac':             await computeHMAC(MS_APP_KEY, MS_HMAC_KEY, JSON.stringify(body)),
      },
      body: JSON.stringify(body),
    });
    if(!r.ok){ console.warn('MyScript error:', r.status, await r.text()); return ''; }
    const d = await r.json();
    // Math API returns LaTeX in .label or .expressions[0].label
    return d?.label
        || d?.expressions?.[0]?.label
        || d?.exports?.['application/x-latex']
        || '';
  } catch(e){
    console.warn('MyScript fetch error:', e);
    return '';
  }
}

// HMAC-SHA512 using Web Crypto API
async function computeHMAC(appKey, hmacKey, message){
  const key = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(appKey + hmacKey),
    {name:'HMAC', hash:'SHA-512'},
    false, ['sign']
  );
  const sig = await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(message));
  return Array.from(new Uint8Array(sig)).map(b=>b.toString(16).padStart(2,'0')).join('');
}

async function recogniseAll(){
  if(!msReady){ toast('Add MyScript API keys in the HTML file'); return; }
  charText={}; wordText={}; phraseText={};
  toast('Recognising with MyScript...');

  // Recognise words (most important level)
  for(let wi=0; wi<hW.length; wi++){
    const pts = hW[wi].idxs.map(i=>strokes[i].points);
    wordText[wi] = await recogniseClusterMS(pts, 'Math');
  }

  // Assemble phrase text from word results
  hP.forEach(ph=>{
    const wids = hW
      .map((_,wi)=>wi)
      .filter(wi=>hW[wi].idxs.some(i=>ph.idxs.includes(i)));
    phraseText[ph.id] = wids.map(wi=>wordText[wi]).filter(Boolean).join(' ');
  });

  // Recognise chars
  for(const ch of hC){
    const pts = ch.idxs.map(i=>strokes[i].points);
    charText[ch.id] = await recogniseClusterMS(pts, 'Math');
  }

  draw();
  buildPList();
  toast('Recognition complete ✓');
}

// ── RUN HIER ──
async function runHier(){
  if(!strokes.length){toast('Draw some strokes first!');return}

  pipe(['raw','prep']);await dl(80);
  const ppts=strokes.map(s=>prep(s.points));
  const cs=ppts.map(ctr);

  pipe(['raw','prep','kd']);await dl(60);
  const t0=performance.now();

  // L1
  const l1=dbscan(cs,nnEps(cs,alpha,1));
  const charMap={};l1.forEach((lbl,i)=>(charMap[lbl]||(charMap[lbl]=[])).push(i));
  const charIds=Object.keys(charMap).map(Number).sort((a,b)=>a-b);
  const charCtrs=charIds.map(id=>{const ix=charMap[id];const px=ix.map(i=>cs[i].x),py=ix.map(i=>cs[i].y);return{x:(Math.min(...px)+Math.max(...px))/2,y:(Math.min(...py)+Math.max(...py))/2};});

  pipe(['raw','prep','kd','h1']);await dl(60);

  // L2
  const l2=dbscan(charCtrs,nnEps(charCtrs,alpha,1));
  const wordMap={};l2.forEach((lbl,ci)=>(wordMap[lbl]||(wordMap[lbl]=[])).push(...charMap[charIds[ci]]));
  const wordIds=Object.keys(wordMap).map(Number).sort((a,b)=>a-b);
  const wordCtrs=wordIds.map(id=>{const ix=wordMap[id];const px=ix.map(i=>cs[i].x),py=ix.map(i=>cs[i].y);return{x:(Math.min(...px)+Math.max(...px))/2,y:(Math.min(...py)+Math.max(...py))/2};});

  pipe(['raw','prep','kd','h1','h2']);await dl(60);

  // L3
  const l3=dbscan(wordCtrs,nnEps(wordCtrs,alpha*1.3,1));
  const phraseMap={};l3.forEach((lbl,wi)=>(phraseMap[lbl]||(phraseMap[lbl]=[])).push(...wordMap[wordIds[wi]]));
  const phraseIds=Object.keys(phraseMap).map(Number).sort((a,b)=>a-b);

  const tf=performance.now()-t0;

  // Assign colors
  strokes.forEach((s,i)=>{
    const cIdx=l1[i];
    const wIdx=wordIds.indexOf(l2[charIds.indexOf(cIdx)]);
    const pIdx=wIdx>=0?phraseIds.indexOf(l3[wIdx]):-1;
    s.charId=cIdx;
    s.wordId=wIdx>=0?wordIds[wIdx]:-1;
    s.phraseId=pIdx>=0?phraseIds[pIdx]:-1;
    s.col=pIdx>=0?PAL[phraseIds[pIdx]%PAL.length]:C1;
  });

  hC=charIds.map(id=>({id,idxs:charMap[id]}));
  hW=wordIds.map((id,wi)=>({id,idxs:wordMap[id],phraseId:phraseIds[phraseIds.indexOf(l3[wi])]}));
  hP=phraseIds.map(id=>({id,idxs:phraseMap[id]}));

  // Update panel
  document.getElementById('nChars').textContent=hC.length;
  document.getElementById('nWords').textContent=hW.length;
  document.getElementById('nPhrases').textContent=hP.length;
  document.getElementById('mTime').textContent=tf.toFixed(1)+'ms';

  const chip=document.getElementById('chipResult');
  chip.style.display='';chip.className='chip live';
  chip.innerHTML=`<span>${hP.length}</span> phrases · <span>${hW.length}</span> words · <span>${hC.length}</span> chars`;

  buildPList();
  pipe(['raw','prep','kd','h1','h2','h3']);
  draw();
  toast(`${hC.length} chars → ${hW.length} words → ${hP.length} phrases · ${tf.toFixed(1)}ms`);

  // Auto-run recognition if MyScript is configured
  if(msReady) await recogniseAll();
}

function buildPList(){
  const list=document.getElementById('pList');
  if(!hP||!hP.length){list.innerHTML='<div style="font-family:var(--mono);font-size:10px;color:var(--muted);padding:6px;text-align:center">No phrases found</div>';return}
  list.innerHTML=hP.map(ph=>{
    const wc=hW.filter(w=>w.idxs.some(i=>ph.idxs.includes(i))).length;
    const cc=hC.filter(c=>c.idxs.some(i=>ph.idxs.includes(i))).length;
    const col=PAL[ph.id%PAL.length];
    const txt=phraseText[ph.id];
    return`<div class="pitem${selPhrase===ph.id?' sel':''}" onclick="selP(${ph.id})">
      <div style="flex:1;min-width:0">
        <div style="display:flex;align-items:center;gap:8px">
          <div class="pswatch" style="background:${col};flex-shrink:0"></div>
          <span class="pname" style="color:${col}">Phrase ${ph.id+1}</span>
          <span class="pmeta">${wc}w · ${cc}c</span>
        </div>
        ${txt?`<div style="font-family:var(--sans);font-size:11px;color:${col};margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;opacity:.85">"${txt}"</div>`:''}
      </div>
    </div>`;
  }).join('');
}

function selP(id){
  selPhrase=selPhrase===id?null:id;
  buildPList();draw();
}

// ── DRAWING ──
function grid(){
  ctx.save();ctx.strokeStyle='rgba(255,255,255,.018)';ctx.lineWidth=1;
  const gs=80*vpZ,sx=((vpX%gs)+gs)%gs,sy=((vpY%gs)+gs)%gs;
  for(let x=sx;x<CW();x+=gs){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,CH());ctx.stroke();}
  for(let y=sy;y<CH();y+=gs){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(CW(),y);ctx.stroke();}
  ctx.restore();
}

function drawS(pts,col,alpha_=1,lw=2.2){
  if(!pts||pts.length<2)return;
  ctx.save();ctx.globalAlpha=alpha_;ctx.strokeStyle=col;
  ctx.lineWidth=lw*vpZ;ctx.lineCap='round';ctx.lineJoin='round';
  ctx.beginPath();const s0=c2s(pts[0].x,pts[0].y);ctx.moveTo(s0.x,s0.y);
  for(let i=1;i<pts.length;i++){const sp=c2s(pts[i].x,pts[i].y);ctx.lineTo(sp.x,sp.y);}
  ctx.stroke();ctx.restore();
}

function bbox(pts){const xs=pts.map(p=>p.x),ys=pts.map(p=>p.y);return{x:Math.min(...xs),y:Math.min(...ys),w:Math.max(...xs)-Math.min(...xs),h:Math.max(...ys)-Math.min(...ys)};}

function drawBox(pts,col,pad,lw,alpha_,dash=[]){
  if(!pts.length)return;
  const b=bbox(pts);
  // pad in world coords, corners transformed to screen
  const p0=c2s(b.x-pad,b.y-pad),p1=c2s(b.x+b.w+pad,b.y+b.h+pad);
  ctx.save();ctx.strokeStyle=col;ctx.globalAlpha=alpha_;
  ctx.lineWidth=lw; // screen pixels, not world
  ctx.setLineDash(dash);
  ctx.strokeRect(p0.x,p0.y,p1.x-p0.x,p1.y-p0.y);
  ctx.restore();
}

function drawLbl(pts,col,txt,pad,size=10){
  if(!pts.length)return;
  const b=bbox(pts);
  const p0=c2s(b.x-pad,b.y-pad);
  ctx.save();
  ctx.font=`700 ${size}px 'Space Mono'`;
  ctx.fillStyle=col;ctx.globalAlpha=1.0;
  ctx.fillText(txt,p0.x+4,p0.y-4);
  ctx.restore();
}

// Draw recognised text above a bounding box
function drawRecogText(pts, col, text, pad, maxWidth){
  if(!pts.length || !text) return;
  const b = bbox(pts);
  const p0 = c2s(b.x - pad, b.y - pad);  // top-left of box in screen coords
  const p1 = c2s(b.x + b.w + pad, b.y - pad);  // top-right

  const boxW = p1.x - p0.x;
  const fontSize = Math.max(10, Math.min(18, boxW / (text.length * 0.65)));

  ctx.save();
  ctx.font = `600 ${fontSize}px 'Space Grotesk'`;
  ctx.fillStyle = col;
  ctx.globalAlpha = 0.95;

  // Background pill behind text for readability
  const tw = ctx.measureText(text).width;
  const padX = 5, padY = 3;
  const tx = p0.x + 4;
  const ty = p0.y - fontSize - 6;

  ctx.globalAlpha = 0.6;
  ctx.fillStyle = '#060608';
  ctx.beginPath();
  ctx.roundRect(tx - padX, ty - padY, tw + padX*2, fontSize + padY*2, 3);
  ctx.fill();

  ctx.globalAlpha = 1.0;
  ctx.fillStyle = col;
  ctx.fillText(text, tx, ty + fontSize - 2);
  ctx.restore();
}

function draw(){
  ctx.clearRect(0,0,CW(),CH());
  grid();

  strokes.forEach(s=>{
    const inSel=selPhrase===null||s.phraseId===selPhrase;
    drawS(s.points,s.col,inSel?1:0.1);
  });
  if(cur)drawS(cur.points,C1);

  if(hC&&hW&&hP){
    // L1: char boxes — mint, tight dashes + recognised char above
    hC.forEach((ch,ci)=>{
      const pts=ch.idxs.flatMap(i=>strokes[i].points);
      const inSel=selPhrase===null||strokes[ch.idxs[0]]?.phraseId===selPhrase;
      drawBox(pts,C1,8,1.5,inSel?0.7:0.15,[3,3]);
      if(inSel && charText[ch.id]) drawRecogText(pts, C1, charText[ch.id], 8);
    });
    // L2: word boxes — amber, medium dashes + recognised word above
    hW.forEach((w,wi)=>{
      const pts=w.idxs.flatMap(i=>strokes[i].points);
      const inSel=selPhrase===null||strokes[w.idxs[0]]?.phraseId===selPhrase;
      drawBox(pts,C2,18,2,inSel?0.85:0.15,[6,4]);
      if(inSel){
        if(wordText[wi]) drawRecogText(pts, C2, wordText[wi], 18);
        else drawLbl(pts,C2,'W',18,9);
      }
    });
    // L3: phrase boxes — lavender, solid + recognised phrase above
    hP.forEach(ph=>{
      const pts=ph.idxs.flatMap(i=>strokes[i].points);
      const isSel=selPhrase===ph.id;
      const isNone=selPhrase===null;
      drawBox(pts,C3,30,2.5,isNone?0.9:isSel?1.0:0.15,[]);
      if(phraseText[ph.id]) drawRecogText(pts, C3, phraseText[ph.id], 30);
      else drawLbl(pts,C3,`P${ph.id+1}`,30,11);
    });
  }
}

// ── SCATTER ──
function addScale(){
  const W=CW(),H=CH();
  const groups=[
    {x:.10,y:.20,sc:45/vpZ,nw:3},{x:.10,y:.55,sc:16/vpZ,nw:4},{x:.58,y:.35,sc:30/vpZ,nw:3}
  ];
  groups.forEach(g=>{
    for(let w=0;w<g.nw;w++){
      const wx=g.x*W+w*g.sc*5.5;const base=s2c(wx,g.y*H);
      for(let c=0;c<3+Math.floor(Math.random()*2);c++){
        const cx=base.x+c*g.sc*1.4;
        for(let t=0;t<1+Math.floor(Math.random()*2);t++){
          const pts=[];
          for(let p=0;p<8+Math.floor(Math.random()*6);p++)
            pts.push({x:cx+(Math.random()-.5)*g.sc,y:base.y+(Math.random()-.5)*g.sc*.7});
          strokes.push({points:pts,id:sid++,col:C1,phraseId:-1});
        }
      }
    }
  });
  hC=hW=hP=null;selPhrase=null;
  document.getElementById('helpEl').style.opacity='0';
  updCount();pipe(['raw']);draw();
  toast('Scale-variant demo added — hit ⬡ Cluster');
}

function clearAll(){
  strokes=[];hC=hW=hP=null;cur=null;selPhrase=null;
  charText={};wordText={};phraseText={};
  document.getElementById('helpEl').style.opacity='1';
  document.getElementById('nChars').textContent='—';
  document.getElementById('nWords').textContent='—';
  document.getElementById('nPhrases').textContent='—';
  document.getElementById('mTime').textContent='—';
  document.getElementById('mStr').textContent='0';
  document.getElementById('nStr').textContent='0';
  document.getElementById('chipResult').style.display='none';
  document.getElementById('pList').innerHTML='<div style="font-family:var(--mono);font-size:10px;color:var(--muted);padding:6px;text-align:center">Run Cluster first</div>';
  pipe([]);draw();
}

// ── CONTROLS ──
function setMode(m){
  mode=m;
  document.getElementById('bDraw').classList.toggle('on',m==='draw');
  document.getElementById('bPan').classList.toggle('on',m==='pan');
  const b=document.getElementById('mBadge');b.className='mode-badge '+m;
  b.textContent=m==='draw'?'Draw Mode':'Pan Mode';
  wrap.style.cursor=m==='pan'?'grab':'crosshair';
}
function updA(){
  alpha=+document.getElementById('sA').value/10;
  document.getElementById('vA').textContent=alpha.toFixed(1);
  document.getElementById('mA').textContent=alpha.toFixed(1);
}
function updCount(){const n=strokes.length;document.getElementById('nStr').textContent=n;document.getElementById('mStr').textContent=n;}

// ── PIPE ──
const PM={raw:'pRaw',prep:'pPrep',kd:'pKD',h1:'pH1',h2:'pH2',h3:'pH3'};
function pipe(active){
  Object.keys(PM).forEach(k=>{
    const el=document.getElementById(PM[k]);el.className='pstep';
    if(active.includes(k))el.classList.add(k===active[active.length-1]?'active':'done');
  });
}

function dl(ms){return new Promise(r=>setTimeout(r,ms))}
function toast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('on');setTimeout(()=>t.classList.remove('on'),2500)}

resize();updA();
</script>
</body>
</html>