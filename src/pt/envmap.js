// Equirectangular environment map for the path tracer (CPU side; PathTracer uploads it when env.map changes).
//
// The sun is cut out of the map and returned as an analytic cone light, so it keeps crisp shadows and the
// sun-light NEE + MIS the tracer already has; what is left (sky, clouds, landscape) is importance sampled through
// an alias table over a coarse ENV_GRID of cells (pmf ~ mean luminance * sin(theta), uniform inside a cell).
//
// Convention (same as three.js equirect): row 0 = up, u = atan2(z, x) / 2pi + 0.5, v = acos(y) / pi.
import { DataUtils, FloatType } from 'three';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';

export const ENV_GRID = [256, 128];  // must match ENV_GW / ENV_GH in pathtracer.wgsl.js

const lum = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

function dirOf(u, v) {
  const phi = (u - 0.5) * 2 * Math.PI, th = v * Math.PI;
  return [Math.cos(phi) * Math.sin(th), Math.cos(th), Math.sin(phi) * Math.sin(th)];
}

// Cuts the sun (the pixels around the brightest one that are much brighter than the ring around them) out of `rgba`
// in place and returns { dir, irradiance } of what was removed, or null.
function cutSun(rgba, W, H, coneDeg) {
  let best = 0, bi = 0;
  for (let i = 0; i < W * H; i++) {
    const l = lum(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]);
    if (l > best) { best = l; bi = i; }
  }
  const s = dirOf(((bi % W) + 0.5) / W, (Math.floor(bi / W) + 0.5) / H);
  const cosIn = Math.cos((coneDeg * Math.PI) / 180), cosOut = Math.cos((coneDeg * 1.6 * Math.PI) / 180);
  const r0 = Math.max(0, Math.floor(bi / W) - Math.ceil((coneDeg * 1.7 * H) / 180));
  const r1 = Math.min(H - 1, Math.floor(bi / W) + Math.ceil((coneDeg * 1.7 * H) / 180));
  // the ring between coneDeg and 1.6 coneDeg is what the sky around the sun looks like
  const ring = [0, 0, 0];
  let n = 0;
  const inner = [];
  for (let y = r0; y <= r1; y++) {
    for (let x = 0; x < W; x++) {
      const d = dirOf((x + 0.5) / W, (y + 0.5) / H);
      const c = d[0] * s[0] + d[1] * s[1] + d[2] * s[2];
      const i = (y * W + x) * 4;
      if (c >= cosIn) inner.push([i, d, y]);
      else if (c >= cosOut) { ring[0] += rgba[i]; ring[1] += rgba[i + 1]; ring[2] += rgba[i + 2]; n++; }
    }
  }
  if (!n) return null;
  for (let k = 0; k < 3; k++) ring[k] /= n;
  const ringL = lum(...ring);
  if (best < 50 * ringL) return null;  // no distinct sun (overcast)
  const E = [0, 0, 0], c = [0, 0, 0];
  const dA = (2 * Math.PI / W) * (Math.PI / H);
  for (const [i, d, y] of inner) {
    if (lum(rgba[i], rgba[i + 1], rgba[i + 2]) < 3 * ringL) continue;
    const w = dA * Math.sin(((y + 0.5) / H) * Math.PI);
    const ex = [0, 1, 2].map((k) => Math.max(0, rgba[i + k] - ring[k]) * w);
    for (let k = 0; k < 3; k++) { E[k] += ex[k]; rgba[i + k] = ring[k]; }
    const wl = lum(...ex);
    for (let k = 0; k < 3; k++) c[k] += d[k] * wl;
  }
  const cl = Math.hypot(...c) || 1;
  return { dir: c.map((x) => x / cl), irradiance: E };
}

// Vose alias table over the cells; entry = [threshold, alias (u32 bits), pmf, 0].
function aliasTable(weights) {
  const N = weights.length;
  const sum = weights.reduce((a, b) => a + b, 0);
  const table = new Float32Array(N * 4);
  const u32 = new Uint32Array(table.buffer);
  const p = weights.map((w) => (w / sum) * N);
  const small = [], large = [];
  p.forEach((x, i) => (x < 1 ? small : large).push(i));
  while (small.length && large.length) {
    const s = small.pop(), l = large[large.length - 1];
    table[s * 4] = p[s];
    u32[s * 4 + 1] = l;
    p[l] -= 1 - p[s];
    if (p[l] < 1) { large.pop(); small.push(l); }
  }
  for (const i of [...small, ...large]) { table[i * 4] = 1; u32[i * 4 + 1] = i; }
  for (let i = 0; i < N; i++) table[i * 4 + 2] = weights[i] / sum;
  return table;
}

// image: { data: Float32Array RGBA, width, height } (row 0 = up). Returns the map the path tracer takes as env.map.
export function prepareEnvMap(image, { sun = true, sunConeDeg = 3 } = {}) {
  const { width: W, height: H } = image;
  const rgba = new Float32Array(image.data);
  const sunInfo = sun ? cutSun(rgba, W, H, sunConeDeg) : null;
  const [GW, GH] = ENV_GRID;
  const weights = new Float64Array(GW * GH);
  for (let y = 0; y < H; y++) {
    const gy = Math.min(GH - 1, Math.floor((y * GH) / H));
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      weights[gy * GW + Math.min(GW - 1, Math.floor((x * GW) / W))] += lum(rgba[i], rgba[i + 1], rgba[i + 2]);
    }
  }
  for (let gy = 0; gy < GH; gy++) {
    const st = Math.sin(((gy + 0.5) / GH) * Math.PI);
    for (let gx = 0; gx < GW; gx++) weights[gy * GW + gx] = weights[gy * GW + gx] * st + 1e-6;
  }
  const half = new Uint16Array(W * H * 4);
  for (let i = 0; i < rgba.length; i++) half[i] = DataUtils.toHalfFloat(Math.min(rgba[i], 65000));
  return { width: W, height: H, half, table: aliasTable(Array.from(weights)), sun: sunInfo };
}

export async function loadEnvMap(url, opts) {
  const tex = await new HDRLoader().setDataType(FloatType).loadAsync(url);
  return prepareEnvMap(tex.image, opts);
}

// The cut-out sun as a path tracer light (pathtracer.js writeUniforms), rotated like the map (env.mapRotation, radians
// about +y). scale / tint: art direction on top of the measured irradiance.
export function envSunLight(map, rotation = 0, { angularDeg = 0.53, scale = 1, tint = [1, 1, 1] } = {}) {
  if (!map.sun) return null;
  const [x, y, z] = map.sun.dir;
  const c = Math.cos(rotation), s = Math.sin(rotation);
  const dir = [c * x + s * z, y, -s * x + c * z];
  const cosMax = Math.cos(((angularDeg / 2) * Math.PI) / 180);
  const omega = 2 * Math.PI * (1 - cosMax);
  return { kind: 0, dir, cosMax, radiance: map.sun.irradiance.map((e, k) => (e * scale * tint[k]) / omega) };
}
