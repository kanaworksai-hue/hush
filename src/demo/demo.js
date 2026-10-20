// Comparison demo: one 1 spp path-traced frame per display frame, denoised by SVGF and Hush; drag the divider to
// compare any two of 1 spp / SVGF / Hush. Works with touch; the render resolution is a fraction of the screen's device
// pixels. "Benchmark" times every scene on this device (the result is also in window.__bench).
//
//   ?scene=box|living|courtyard&left=noisy|svgf|hush&right=...&scale=0.5[&all]
//   &brand=0: without the KanaWorks_AI artwork (src/demo/brand.js)
//   &all: run every denoiser each frame (default on desktop; on phones only the ones on screen run)
//   &t=<seconds>&pause&split=<0..1>: start time, frozen scene, divider position (screenshots)
//   &clean: no HUD / controls / grip (README images)
import * as THREE from 'three';
import { SCENES } from '../scenes.js';
import { SceneGPU, PathTracer, JITTER } from '../pt/index.js';
import { Hush, SVGF } from '../index.js';
import { initDevice, GpuTimer, Composer } from '../gpu.js';
import { addBrand } from './brand.js';
import { Post } from './post.js';

const q = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const METHODS = { noisy: '1 spp', svgf: 'SVGF', hush: 'Hush' };
const SCENE_NAMES = { box: 'Box', living: 'Living room', courtyard: 'Courtyard' };
const SCALES = [0.25, 0.35, 0.5, 0.75, 1];
const mobile = matchMedia('(pointer: coarse)').matches;
const dt = 1 / 60;  // fixed scene step per frame (the motion per frame the denoiser was trained on)

const state = {
  scene: SCENES[q.get('scene')] ? q.get('scene') : 'box',
  left: METHODS[q.get('left')] ? q.get('left') : 'noisy',
  right: METHODS[q.get('right')] ? q.get('right') : 'hush',
  scale: +(q.get('scale') ?? 0),
  split: +(q.get('split') ?? 0.5),
  paused: q.has('pause'),
};
const runAll = q.has('all') || !mobile;
if (q.has('clean')) for (const id of ['hud', 'bar', 'grip']) $(id).style.display = 'none';

function message(html) {
  const m = $('msg');
  m.hidden = !html;
  if (html) m.innerHTML = html;
}

let device, info, timer;
try {
  ({ device, info } = await initDevice());
} catch (e) {
  message(`This demo needs WebGPU (Chrome / Edge 113+, Safari 26+).<br><span style="color:#9a978f">${e.message}</span>`);
  throw e;
}
device.lost.then((i) => message(`GPU device lost: ${i.message}`));
timer = new GpuTimer(device);

const canvas = $('c');
const ctx = canvas.getContext('webgpu');
const format = navigator.gpu.getPreferredCanvasFormat();
ctx.configure({ device, format, alphaMode: 'opaque' });

// ------------------------------------------------------------------ display: two images, a divider, bloom, AgX (post.js)
const post = new Post(device, format);

// ------------------------------------------------------------------ scene / renderer state
let S = null, sg = null, pt = null, svgf = null, hush = null, comp = null;
let W = 0, H = 0, frame = 0, time = +(q.get('t') ?? 0), prevVP = null, loading = null;
const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 200);
const ms = { pt: null, svgf: null, hush: null, cpu: null };
const ema = (k, v) => { ms[k] = ms[k] == null ? v : ms[k] * 0.9 + v * 0.1; };

function setCamera(t) {
  const c = S.camera(t);
  camera.fov = c.fov ?? 50;
  camera.aspect = W / H;
  camera.position.set(...c.pos);
  camera.lookAt(new THREE.Vector3(...c.target));
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
}

function defaultScale() {
  const px = innerWidth * innerHeight * devicePixelRatio ** 2;
  if (mobile) return 0.5;
  // desktop: about 0.6 MPixel of path tracing (1080p window -> 0.5)
  return SCALES.filter((s) => s * s * px <= 0.65e6).pop() ?? SCALES[0];
}

function resize() {
  const cw = Math.round(innerWidth * devicePixelRatio), ch = Math.round(innerHeight * devicePixelRatio);
  if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
  const s = state.scale || defaultScale();
  const w = Math.max(32, Math.round(cw * s)), h = Math.max(32, Math.round(ch * s));
  if (w === W && h === H && pt) return;
  W = w; H = h;
  if (pt) pt.resize(W, H); else if (sg) pt = new PathTracer(device, sg, W, H);
  if (svgf) svgf.resize(W, H); else svgf = new SVGF(device, W, H);
  if (hush) hush.resize(W, H); else hush = new Hush(device, { width: W, height: H });
  if (comp) { comp.noisy.destroy(); comp.ref.destroy(); }
  comp = new Composer(device, W, H);
  prevVP = null;
}

async function loadScene(name) {
  state.scene = name;
  updateButtons();
  const job = (async () => {
    message(`Loading ${SCENE_NAMES[name]}…`);
    const t0 = performance.now();
    const next = await SCENES[name]();
    if (q.get('brand') !== '0') await addBrand(name, next.scene);
    next.animate(time, dt);
    const g = new SceneGPU(device);
    await g.build(next.scene);
    pt?.destroy(); sg?.destroy();
    S = next; sg = g; pt = null;
    W = H = 0;
    resize();
    hush.exposure = S.exposure ?? 1;
    svgf.reset(); hush.reset();
    message('');
    console.log(`scene ${name}: ${sg.triTotal} tris, ${(performance.now() - t0).toFixed(0)} ms`);
  })();
  loading = job;
  await job;
  if (loading === job) loading = null;
}

// ------------------------------------------------------------------ frame
const hushTimer = { desc: (l) => timer.desc(`hush.${l}`) };
let lastT = performance.now();
let ran = { svgf: false, hush: false };

function renderFrame() {
  const now = performance.now();
  ema('cpu', now - lastT);
  lastT = now;
  if (loading || !pt) return;
  if (!state.paused) time += dt;
  S.animate(time, dt);
  sg.update();
  setCamera(time);
  const env = S.env(time, dt);
  const slot = frame & 1;
  const enc = device.createCommandEncoder();
  timer.begin();
  const vp = pt.writeUniforms({ camera, prevViewProj: prevVP, frame, mode: 0, jitter: JITTER[frame % 16], env });
  pt.encode(enc, slot, timer);
  const g = pt.gbuf[slot], gp = pt.gbuf[slot ^ 1];
  comp.encode(enc, g, pt.accum);
  const shown = new Set([state.left, state.right]);
  const want = { svgf: runAll || shown.has('svgf'), hush: runAll || shown.has('hush') };
  if (want.svgf) {
    if (!ran.svgf) svgf.reset();
    svgf.encode(enc, { ...g, prevNormalDepth: gp.normalDepth, prevMotion: gp.motion }, timer);
  }
  if (want.hush) {
    if (!ran.hush) hush.reset();
    hush.encode(enc, g, hushTimer);
  }
  ran = want;
  const tex = (m) => (m === 'svgf' ? svgf.output : m === 'hush' ? hush.output : comp.noisy);
  post.encode(enc, {
    left: tex(state.left), right: tex(state.right), split: state.split, exposure: (S.exposure ?? 1) * (S.display ?? 1),
    target: ctx.getCurrentTexture().createView(), width: canvas.width, height: canvas.height,
  });
  const resolved = timer.resolve(enc);
  device.queue.submit([enc.finish()]);
  prevVP = vp.clone();
  frame++;
  if (resolved) {
    const r0 = { ...ran };
    timer.read().then((r) => {
      if (!r) return;
      let sv = 0, hu = 0;
      for (const [k, v] of Object.entries(r)) {
        if (k === 'pathtrace') ema('pt', v);
        else if (k.startsWith('hush.')) hu += v;
        else sv += v;
      }
      if (r0.svgf) ema('svgf', sv);
      if (r0.hush) ema('hush', hu);
      onTimings?.(r, r0);
    });
  }
}

let onTimings = null;
let hudN = 0;
function hudText() {
  const f = (v) => (v == null ? '—' : `${v.toFixed(2)} ms`);
  const gpu = timer.enabled;
  const rows = [
    ['Path trace (1 spp)', gpu ? f(ms.pt) : 'n/a'],
    ['SVGF', gpu ? (ran.svgf ? f(ms.svgf) : 'off') : 'n/a'],
    ['Hush', gpu ? (ran.hush ? f(ms.hush) : 'off') : 'n/a'],
    ['Frame', ms.cpu ? `${ms.cpu.toFixed(1)} ms · ${(1000 / ms.cpu).toFixed(0)} fps` : '—'],
  ];
  return rows.map(([a, b]) => `<div class="row"><span>${a}</span><span>${b}</span></div>`).join('')
    + `<div class="dim">${W}×${H} · ${SCENE_NAMES[state.scene]}${gpu ? '' : ' · no GPU timestamps'}</div>`;
}

function loop() {
  renderFrame();
  if (++hudN % 15 === 0) $('hudBody').innerHTML = hudText();
  requestAnimationFrame(loop);
}

// ------------------------------------------------------------------ UI
function seg(id, items, get, set) {
  const el = $(id);
  for (const [k, label] of Object.entries(items)) {
    const b = document.createElement('button');
    b.textContent = label;
    b.dataset.k = k;
    b.onclick = () => set(k);
    el.appendChild(b);
  }
  el.update = () => { for (const b of el.children) b.classList.toggle('on', b.dataset.k === get()); };
}
seg('scene', SCENE_NAMES, () => state.scene, (k) => { if (k !== state.scene && !bench) loadScene(k); });
seg('left', METHODS, () => state.left, (k) => { state.left = k; updateButtons(); });
seg('right', METHODS, () => state.right, (k) => { state.right = k; updateButtons(); });
const sel = $('scale');
for (const s of SCALES) sel.add(new Option(`${Math.round(s * 100)}%`, s));
sel.onchange = () => { state.scale = +sel.value; resize(); updateButtons(); };
$('pause').onclick = () => { state.paused = !state.paused; updateButtons(); };
function updateButtons() {
  for (const id of ['scene', 'left', 'right']) $(id).update();
  sel.value = String(state.scale || defaultScale());
  $('pause').textContent = state.paused ? 'Play' : 'Pause';
  $('tagL').textContent = METHODS[state.left];
  $('tagR').textContent = METHODS[state.right];
  placeSplit();
}
function placeSplit() {
  const x = state.split * innerWidth;
  $('grip').style.left = `${x}px`;
  $('tagL').style.left = `${Math.max(x / 2, 40)}px`;
  $('tagR').style.left = `${Math.min(x + (innerWidth - x) / 2, innerWidth - 40)}px`;
}
let drag = false;
canvas.addEventListener('pointerdown', (e) => { drag = true; canvas.setPointerCapture(e.pointerId); state.split = e.clientX / innerWidth; placeSplit(); });
canvas.addEventListener('pointermove', (e) => { if (drag) { state.split = Math.min(Math.max(e.clientX / innerWidth, 0), 1); placeSplit(); } });
canvas.addEventListener('pointerup', () => { drag = false; });
addEventListener('resize', () => { resize(); placeSplit(); });

// ------------------------------------------------------------------ benchmark (every scene, current resolution)
let bench = null;
$('benchBtn').onclick = async () => {
  if (bench) return;
  const box = $('bench');
  box.hidden = false;
  const out = { date: new Date().toISOString(), ua: navigator.userAgent, adapter: info, dpr: devicePixelRatio,
    screen: [innerWidth, innerHeight], timestamps: timer.enabled, runAll, scenes: {} };
  const keep = { scene: state.scene, paused: state.paused };
  bench = out;
  state.paused = false;
  for (const name of Object.keys(SCENES)) {
    box.textContent = `Benchmark: loading ${name}…`;
    await loadScene(name);
    const acc = { frames: 0, cpu: 0, gpu: {} , n: 0 };
    let warm = 60;
    const t0 = performance.now();
    await new Promise((resolve) => {
      onTimings = (r) => {
        if (warm > 0) return;
        for (const [k, v] of Object.entries(r)) acc.gpu[k] = (acc.gpu[k] ?? 0) + v;
        acc.n++;
      };
      let last = performance.now();
      const tick = () => {
        const now = performance.now();
        if (warm > 0) warm--; else { acc.frames++; acc.cpu += now - last; }
        last = now;
        box.textContent = `Benchmark: ${name} ${acc.frames}/180`;
        if (acc.frames >= 180 || now - t0 > 60000) resolve(); else requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
    onTimings = null;
    const g = Object.fromEntries(Object.entries(acc.gpu).map(([k, v]) => [k, +(v / Math.max(acc.n, 1)).toFixed(3)]));
    const sum = (f) => +Object.entries(g).filter(([k]) => f(k)).reduce((a, [, v]) => a + v, 0).toFixed(3);
    out.scenes[name] = {
      width: W, height: H, frameMs: +(acc.cpu / Math.max(acc.frames, 1)).toFixed(2), fps: +(1000 * acc.frames / Math.max(acc.cpu, 1)).toFixed(1),
      ptMs: g.pathtrace ?? null, svgfMs: sum((k) => k !== 'pathtrace' && !k.startsWith('hush.')) || null,
      hushMs: sum((k) => k.startsWith('hush.')) || null, gpuMs: g, tris: sg.triTotal,
    };
  }
  const lines = [`${info}`, `${W}×${H} (scale ${state.scale || defaultScale()})`, ''];
  for (const [k, v] of Object.entries(out.scenes)) {
    const f = (x) => (x == null ? '  n/a' : x.toFixed(2).padStart(6));
    lines.push(`${k.padEnd(10)} frame ${v.frameMs.toFixed(1).padStart(6)} ms  PT ${f(v.ptMs)}  SVGF ${f(v.svgfMs)}  Hush ${f(v.hushMs)}`);
  }
  lines.push('', 'tap to close');
  box.textContent = lines.join('\n');
  box.onclick = () => { box.hidden = true; };
  window.__bench = out;
  bench = null;
  state.paused = keep.paused;
  await loadScene(keep.scene);
};

// ------------------------------------------------------------------ start
resize();
await loadScene(state.scene);
updateButtons();
requestAnimationFrame(loop);
if (q.has('autobench')) $('benchBtn').click();
