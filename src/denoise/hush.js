// Hush: neural-parameter denoiser for 1 spp path tracing (WGSL compute).
//
// SVGF-style temporal accumulation + a tiny per-pixel MLP at half resolution that predicts the filter parameters
// (temporal alpha and anti-lag clamp for the next frame, a-trous luminance strength / centre weight, coarse blend)
// + 3 edge-aware 3x3 a-trous iterations + a 1/4-res coarse level + TAA. The network never outputs pixels.
// The weights were trained on a differentiable PyTorch copy of these passes; a golden test checks that the browser
// output matches it, so every pass here has an exact twin in the training code.
//
// Inputs per frame (same G-buffer as the SVGF baseline):
//   radiance rgba16float (rgb), normalDepth rgba16float (world normal, linear depth; < 0 = background),
//   motion rgba16float (xy = previous - current pixel, object id < 2048, previous linear depth),
//   material rgba16float (albedo, roughness), emission rgba16float (primary emission).
//   History is kept internally (no previous G-buffer needed).
// Output: `this.output` rgba16float, linear HDR, antialiased.

import DEFAULT_WEIGHTS from './hush-weights.js';

export const N_FEAT = 20;
export const N_OUT = 12;
// f16 weights, row-major (out, in) like torch.nn.Linear: W1 b1 W2 b2 W3 b3 (feature normalisation folded into W1/b1).
// The hidden width h follows from the count: n = h*h + (N_FEAT + N_OUT + 2) h + N_OUT (h = 16: 812 weights).
export function hiddenOf(n) {
  for (let h = 1; h <= 256; h++) if (h * h + (N_FEAT + N_OUT + 2) * h + N_OUT === n) return h;
  return 0;
}
const COARSE_S = 0.25;

const COMMON = /* wgsl */ `
struct P { size: vec2u, flags: u32, _a: u32, exposure: f32, s: f32, size2: vec2u };
@group(0) @binding(0) var<uniform> prm: P;
fn lum(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
fn inside(p: vec2i) -> bool { return p.x >= 0 && p.y >= 0 && p.x < i32(prm.size.x) && p.y < i32(prm.size.y); }
fn clampP(p: vec2i) -> vec2i { return clamp(p, vec2i(0), vec2i(prm.size) - 1); }
fn octEnc(n: vec3f) -> vec2f {
  let v = n / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-20);
  let s = vec2f(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
  if (v.z < 0.0) { return (1.0 - abs(v.yx)) * s; }
  return v.xy;
}
fn octDec(e: vec2f) -> vec3f {
  var v = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
  let t = max(-v.z, 0.0);
  v.x += select(t, -t, v.x >= 0.0);
  v.y += select(t, -t, v.y >= 0.0);
  return v / max(length(v), 1e-20);
}
fn log1p(x: f32) -> f32 { return log(1.0 + x); }
const HMAX = 32.0;
`;

// Workgroups are TW x TH pixels; passes with neighbourhoods first stage a tile (+ apron) in workgroup memory.
const TW = 16, TH = 8;
const tileDecl = (apron, decls) => /* wgsl */ `
const AP = ${apron};
const SW = ${TW + 2 * apron};
const SH = ${TH + 2 * apron};
${decls}
fn sIdx(l: vec2i) -> i32 { return (l.y + AP) * SW + l.x + AP; }
`;

const lit = (v) => {
  const t = String(v);
  return /[.e]/.test(t) ? t : `${t}.0`;
};
const v4 = (a) => `vec4f(${a.map(lit).join(', ')})`;

const TILE_FNS = /* wgsl */ `
fn qf16(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(clamp(x, -65504.0, 65504.0)) & 0xffffe000u); }
fn q4(v: vec4f) -> vec4f { return vec4f(qf16(v.x), qf16(v.y), qf16(v.z), qf16(v.w)); }
fn pow128(x: f32) -> f32 {
  let x2 = x * x; let x4 = x2 * x2; let x8 = x4 * x4; let x16 = x8 * x8; let x32 = x16 * x16; let x64 = x32 * x32;
  return x64 * x64;
}
`;

// 1. demodulate, reproject the history (own key of the previous frame), anti-lag clamp + temporal blend; then per
// 2x2 block the 20 network features and the half-res coarse source (packed f16, see FEAT_LAYOUT)
//   F0 = f0..f7, F1 = f8..f15, F2 = f16..f19 + Ah (mean colour, mean variance), F3 = Gh (oct normal, depth, count)
const TEMPORAL = COMMON + /* wgsl */ `
@group(0) @binding(1) var radiance: texture_2d<f32>;
@group(0) @binding(2) var normalDepth: texture_2d<f32>;
@group(0) @binding(3) var motion: texture_2d<f32>;
@group(0) @binding(4) var material: texture_2d<f32>;
@group(0) @binding(5) var prevKey: texture_2d<f32>;
@group(0) @binding(6) var prevColor: texture_2d<f32>;
@group(0) @binding(7) var prevMom: texture_2d<f32>;
@group(0) @binding(8) var prevPar0: texture_2d<f32>;
@group(0) @binding(9) var outA: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var outMom: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var outKey: texture_storage_2d<rgba16float, write>;
@group(0) @binding(12) var outF0: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(13) var outF1: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(14) var outF2: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(15) var outF3: texture_storage_2d<rgba32uint, write>;
${tileDecl(1, `var<workgroup> sI: array<vec4f, SW * SH>;  // illum rgb, valid
var<workgroup> sP: array<array<vec4f, 5>, ${TW * TH}>;  // per-pixel feature inputs (f16-rounded like the torch side)`)}

// f16 rounding toward zero, like a rgba16float storage write (normal range)
fn qf16(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(clamp(x, -65504.0, 65504.0)) & 0xffffe000u); }
fn q4(v: vec4f) -> vec4f { return vec4f(qf16(v.x), qf16(v.y), qf16(v.z), qf16(v.w)); }
fn pk(a: f32, b: f32) -> u32 { return pack2x16float(vec2f(qf16(a), qf16(b))); }

fn check(k: vec4f, n: vec3f, zPrev: f32, obj: f32) -> bool {
  return k.z >= 0.0 && k.w == obj && abs(k.z - zPrev) <= 0.05 * zPrev + 0.01 && dot(octDec(k.xy), n) >= 0.9;
}
fn consistent(q: vec2i, n: vec3f, zPrev: f32, obj: f32) -> bool {
  return inside(q) && check(textureLoad(prevKey, q, 0), n, zPrev, obj);
}

// temporal accumulation of one pixel; returns the feature inputs (P0..P4, see sP)
fn pixel(p: vec2i, lp: vec2i) -> array<vec4f, 5> {
  var P = array<vec4f, 5>(vec4f(0.0), vec4f(0.0), vec4f(0.0, 0.0, -1.0, 0.0), vec4f(0.0), vec4f(0.0));
  let nd = textureLoad(normalDepth, p, 0);
  let mv = textureLoad(motion, p, 0);
  if (nd.w < 0.0) {
    textureStore(outA, p, vec4f(0.0)); textureStore(outMom, p, vec4f(0.0));
    textureStore(outKey, p, vec4f(0.0, 0.0, -1.0, mv.z));
    return P;
  }
  let n = nd.xyz;
  let mr = textureLoad(material, p, 0);
  let illum = sI[sIdx(lp)].rgb;
  let l = lum(illum);
  // 3x3 stats of the current illumination (valid neighbours), centred (two passes over the tile, no local array)
  var s1 = vec3f(0.0); var cnt = 0.0;
  for (var j = 0; j < 9; j++) {
    let c = sI[sIdx(lp + vec2i(j % 3 - 1, j / 3 - 1))];
    s1 += c.rgb * c.a; cnt += c.a;
  }
  cnt = max(cnt, 1.0);
  let mu = s1 / cnt;
  let slm = lum(mu);
  var sv3 = vec3f(0.0); var svar = 0.0;
  for (var j = 0; j < 9; j++) {
    let c = sI[sIdx(lp + vec2i(j % 3 - 1, j / 3 - 1))];
    let d = c.rgb - mu;
    let dl = lum(c.rgb) - slm;
    sv3 += c.a * d * d;
    svar += c.a * dl * dl;
  }
  let sd = sqrt(max(sv3 / cnt, vec3f(1e-12)));
  svar /= cnt;
  // reprojection
  var has = false;
  var hc = vec3f(0.0);
  var hm = vec4f(0.0);
  var v = 0.0;
  var alpha = 1.0; var gamma = 1.0;
  if ((prm.flags & 2u) == 0u) {
    let pp = vec2f(p) + mv.xy;
    let b = floor(pp);
    let f = pp - b;
    let bi = vec2i(b);
    var sumW = 0.0;
    let wts = array<f32, 4>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    // issue all 12 loads up front (clamped), then test: the dependent load chain is the cost here
    var k4: array<vec4f, 4>; var c4: array<vec3f, 4>; var m4: array<vec4f, 4>;
    for (var i = 0; i < 4; i++) {
      let q = clampP(bi + vec2i(i & 1, i >> 1));
      k4[i] = textureLoad(prevKey, q, 0); c4[i] = textureLoad(prevColor, q, 0).rgb; m4[i] = textureLoad(prevMom, q, 0);
    }
    let c = vec2i(floor(pp + 0.5));
    let pq = textureLoad(prevPar0, clampP(c) >> vec2u(1u), 0);
    for (var i = 0; i < 4; i++) {
      if (wts[i] > 0.0 && inside(bi + vec2i(i & 1, i >> 1)) && check(k4[i], n, mv.w, mv.z)) {
        sumW += wts[i];
        hc += wts[i] * c4[i];
        hm += wts[i] * m4[i];
      }
    }
    if (sumW >= 0.01) {
      hc /= sumW; hm /= sumW; v = sumW; has = true;
    } else {
      var n3 = 0.0;
      hc = vec3f(0.0); hm = vec4f(0.0);
      for (var j = 0; j < 9; j++) {
        let q = c + vec2i(j % 3 - 1, j / 3 - 1);
        if (consistent(q, n, mv.w, mv.z)) {
          n3 += 1.0;
          hc += textureLoad(prevColor, q, 0).rgb;
          hm += textureLoad(prevMom, q, 0);
        }
      }
      if (n3 > 0.0) { hc /= n3; hm /= n3; v = 0.1; has = true; }
    }
    alpha = pq.x; gamma = pq.y;
  }
  has = has && hm.w > 0.0;
  var color = illum;
  var nm = vec2f(l, l * l);
  var nfast = l;
  var len1 = 1.0;
  var lag = 0.0;
  if (has) {
    let hcl = min(max(hc, mu - gamma * sd), mu + gamma * sd);
    len1 = min(hm.w + 1.0, HMAX);
    let a = max(alpha, 1.0 / len1);
    color = hcl + (illum - hcl) * a;
    nm = vec2f(hm.x + (l - hm.x) * a, hm.y + (l * l - hm.y) * a);
    nfast = hm.z + (l - hm.z) * max(1.0 / 6.0, 1.0 / len1);
    let lh = lum(hc);
    lag = abs(slm - lh) / (sqrt(max(svar / cnt, 1e-12)) + 0.02 * lh + 1e-3);
  } else {
    v = 0.0;
  }
  let tvar = max(nm.y - nm.x * nm.x, 0.0);
  let A = vec4f(color, svar + (tvar - svar) * min(len1 / 4.0, 1.0));
  let oc = octEnc(n);
  textureStore(outA, p, A);
  textureStore(outMom, p, vec4f(nm, nfast, len1));
  textureStore(outKey, p, vec4f(oc, nd.w, mv.z));
  // feature inputs, from the values as stored (f16)
  let E = prm.exposure;
  let Aq = q4(A);
  let m1 = E * qf16(nm.x);
  let m2 = E * E * qf16(nm.y);
  P[0] = vec4f(E * lum(Aq.rgb), E * qf16(l), max(m2 - m1 * m1, 0.0), len1);
  P[1] = vec4f(qf16(v), qf16(lag), E * qf16(nfast), m1);
  P[2] = vec4f(qf16(oc.x), qf16(oc.y), qf16(nd.w), qf16(length(mv.xy)));
  P[3] = mr;
  P[4] = Aq;
  return P;
}

// 20 features of half-res block X (local block lb) -> F0..F3
fn block(X: vec2i, lb: vec2i) {
  let E = prm.exposure;
  var nv = 0.0;
  var S0 = vec4f(0.0); var S1 = vec4f(0.0); var N = vec3f(0.0); var S3 = vec4f(0.0); var AC = vec4f(0.0);
  var MV = 0.0; var Z = 0.0;
  var la: array<f32, 4>; var ns: array<vec3f, 4>; var zs: array<f32, 4>; var ok: array<bool, 4>;
  for (var j = 0; j < 4; j++) {
    let l = lb * 2 + vec2i(j & 1, j >> 1);
    let P = sP[l.y * ${TW} + l.x];
    ok[j] = P[2].z >= 0.0;
    if (!ok[j]) { continue; }
    nv += 1.0;
    la[j] = P[0].x; ns[j] = octDec(P[2].xy); zs[j] = P[2].z;
    S0 += P[0]; S1 += P[1]; N += ns[j]; S3 += P[3]; AC += P[4]; MV += P[2].w; Z += P[2].z;
  }
  if (nv == 0.0) {
    textureStore(outF0, X, vec4u(0u)); textureStore(outF1, X, vec4u(0u)); textureStore(outF2, X, vec4u(0u));
    textureStore(outF3, X, vec4u(pk(0.0, 0.0), pk(-1.0, 0.0), 0u, 0u));
    return;
  }
  let k = 1.0 / nv;
  S0 *= k; S1 *= k; N *= k; S3 *= k; AC *= k; MV *= k; Z *= k;
  let Lacc = S0.x;
  var VA = 0.0;
  for (var j = 0; j < 4; j++) { if (ok[j]) { VA += (la[j] - Lacc) * (la[j] - Lacc); } }
  VA *= k;
  // 4x4 neighbourhood (block +-1) stats of the current luminance, centred
  var s1 = 0.0; var n4 = 0.0;
  for (var j = 0; j < 16; j++) {
    let t = sI[sIdx(lb * 2 + vec2i((j & 3) - 1, (j >> 2) - 1))];
    s1 += t.a * E * qf16(lum(t.rgb)); n4 += t.a;
  }
  let mu4 = s1 / max(n4, 1.0);
  var var4 = 0.0;
  for (var j = 0; j < 16; j++) {
    let t = sI[sIdx(lb * 2 + vec2i((j & 3) - 1, (j >> 2) - 1))];
    let d = E * qf16(lum(t.rgb)) - mu4;
    var4 += t.a * d * d;
  }
  var4 /= max(n4, 1.0);
  let Nn = N / max(length(N), 1e-6);
  var zmax = -1e30; var zmin = 1e30; var mind = 1e30;
  for (var j = 0; j < 4; j++) {
    if (ok[j]) { zmax = max(zmax, zs[j]); zmin = min(zmin, zs[j]); mind = min(mind, dot(ns[j], Nn)); }
  }
  let dz = clamp((zmax - zmin) / max(zmin, 1e-6), 0.0, 1.0);
  let dn = clamp(1.0 - mind, 0.0, 2.0);
  let oc = octEnc(Nn);
  let f = array<f32, 20>(
    log1p(Lacc), log1p(S0.y), log1p(S0.z), log1p(var4), S0.w / HMAX, S1.x, log1p(S1.y),
    log1p(abs(S1.z - S1.w) / (S1.w + 1e-3)), oc.x, oc.y, dz, dn, S3.x, S3.y, S3.z, S3.w,
    1.0 - nv / 4.0, log1p(MV), log1p(mu4), log1p(VA));
  textureStore(outF0, X, vec4u(pk(f[0], f[1]), pk(f[2], f[3]), pk(f[4], f[5]), pk(f[6], f[7])));
  textureStore(outF1, X, vec4u(pk(f[8], f[9]), pk(f[10], f[11]), pk(f[12], f[13]), pk(f[14], f[15])));
  textureStore(outF2, X, vec4u(pk(f[16], f[17]), pk(f[18], f[19]), pk(AC.x, AC.y), pk(AC.z, AC.w)));
  textureStore(outF3, X, vec4u(pk(oc.x, oc.y), pk(Z, nv), 0u, 0u));
}

@compute @workgroup_size(${TW}, ${TH})
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  let origin = vec2i(wid.xy * vec2u(${TW}u, ${TH}u)) - AP;
  for (var i = i32(li); i < SW * SH; i += ${TW * TH}) {
    let q = origin + vec2i(i % SW, i / SW);
    var e = vec4f(0.0);
    if (inside(q) && textureLoad(normalDepth, q, 0).w >= 0.0) {
      e = vec4f(textureLoad(radiance, q, 0).rgb / max(textureLoad(material, q, 0).rgb, vec3f(1e-3)), 1.0);
    }
    sI[i] = e;
  }
  workgroupBarrier();
  let p = vec2i(wid.xy * vec2u(${TW}u, ${TH}u) + lid.xy);
  var P = array<vec4f, 5>(vec4f(0.0), vec4f(0.0), vec4f(0.0, 0.0, -1.0, 0.0), vec4f(0.0), vec4f(0.0));
  if (inside(p)) { P = pixel(p, vec2i(lid.xy)); }
  sP[li] = P;
  workgroupBarrier();
  if (li < ${(TW / 2) * (TH / 2)}u) {
    let lb = vec2i(i32(li) % ${TW / 2}, i32(li) / ${TW / 2});
    let X = vec2i(wid.xy * vec2u(${TW / 2}u, ${TH / 2}u)) + lb;
    let hs = vec2i((prm.size + 1u) / 2u);
    if (X.x < hs.x && X.y < hs.y) { block(X, lb); }
  }
}
`;

// 2. MLP on the features -> filter parameters (half res, prm.size = half size); then the first mid-level a-trous
// (3x3, step 1, variance prefilter) of the block means with this pixel's s_1 / c_1
const INFER = COMMON + TILE_FNS + /* wgsl */ `
@group(0) @binding(1) var F0: texture_2d<u32>;
@group(0) @binding(2) var F1: texture_2d<u32>;
@group(0) @binding(3) var F2: texture_2d<u32>;
@group(0) @binding(4) var F3: texture_2d<u32>;
@group(0) @binding(5) var out0: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var out1: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var out2: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var outMid: texture_storage_2d<rgba16float, write>;
const AP = 1;
const SW = ${TW + 2};
var<workgroup> sA: array<vec4f, ${(TW + 2) * (TH + 2)}>;  // block means
var<workgroup> sG: array<vec4f, ${(TW + 2) * (TH + 2)}>;  // decoded normal, depth
fn ti(l: vec2i) -> i32 { return (l.y + AP) * SW + l.x + AP; }
${atrousFn({ R: 1, step: 1, prefilter: true })}
fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

@compute @workgroup_size(${TW}, ${TH})
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  let base = vec2i(wid.xy * vec2u(${TW}u, ${TH}u));
  for (var i = i32(li); i < ${(TW + 2) * (TH + 2)}; i += ${TW * TH}) {
    let q = clampP(base - AP + vec2i(i % SW, i / SW));
    let a2 = textureLoad(F2, q, 0);
    sA[i] = vec4f(unpack2x16float(a2.z), unpack2x16float(a2.w));
    let g3 = textureLoad(F3, q, 0);
    sG[i] = vec4f(octDec(unpack2x16float(g3.x)), unpack2x16float(g3.y).x);
  }
  workgroupBarrier();
  let l = vec2i(lid.xy);
  let X = base + l;
  if (!inside(X)) { return; }
  if (sG[ti(l)].w < 0.0) {
    textureStore(out0, X, vec4f(0.0)); textureStore(out1, X, vec4f(0.0)); textureStore(out2, X, vec4f(0.0));
    textureStore(outMid, X, vec4f(0.0));
    return;
  }
  let a = textureLoad(F0, X, 0); let b = textureLoad(F1, X, 0); let c2 = textureLoad(F2, X, 0);
  let u = array<vec2f, 10>(unpack2x16float(a.x), unpack2x16float(a.y), unpack2x16float(a.z), unpack2x16float(a.w),
    unpack2x16float(b.x), unpack2x16float(b.y), unpack2x16float(b.z), unpack2x16float(b.w),
    unpack2x16float(c2.x), unpack2x16float(c2.y));
  let f = array<f32, ${N_FEAT}>(u[0].x, u[0].y, u[1].x, u[1].y, u[2].x, u[2].y, u[3].x, u[3].y, u[4].x, u[4].y,
    u[5].x, u[5].y, u[6].x, u[6].y, u[7].x, u[7].y, u[8].x, u[8].y, u[9].x, u[9].y);
  // MLP ${N_FEAT}-h-h-${N_OUT}, unrolled, weights baked in as literals (mlpCode)
  MLP_CODE
  let o = array<f32, 12>(o_0.x, o_0.y, o_0.z, o_0.w, o_1.x, o_1.y, o_1.z, o_1.w, o_2.x, o_2.y, o_2.z, o_2.w);
  let alpha = 1.0 / 32.0 + (31.0 / 32.0) * sigmoid(o[0]);
  let gamma = 0.5 + 7.5 * sigmoid(o[1]);
  let beta = sigmoid(o[2]);
  let kappa = sigmoid(o[3]);
  let s = 0.25 * exp(clamp(vec4f(o[4], o[5], o[6], o[7]), vec4f(-5.0), vec4f(5.0)));
  let c = exp(clamp(vec4f(o[8], o[9], o[10], o[11]), vec4f(-4.0), vec4f(4.0)));
  textureStore(out0, X, vec4f(alpha, gamma, beta, kappa));
  textureStore(out1, X, s);
  textureStore(out2, X, c);
  textureStore(outMid, X, filt(l, X, qf16(s.y), qf16(c.y)));
}
`;

// 3. coarse level, first pass (prm.size = quarter size, prm.size2 = half size): 2x2 half -> quarter for the tile +-2
// (in workgroup memory), then the first fixed 5x5 a-trous (step 1); also writes the coarse guide
const COARSE1 = COMMON + TILE_FNS + /* wgsl */ `
@group(0) @binding(1) var F2: texture_2d<u32>;
@group(0) @binding(2) var F3: texture_2d<u32>;
@group(0) @binding(3) var outC: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var outG: texture_storage_2d<rgba16float, write>;
const AP = 2;
const SW = ${TW + 4};
var<workgroup> sA: array<vec4f, ${(TW + 4) * (TH + 4)}>;
var<workgroup> sG: array<vec4f, ${(TW + 4) * (TH + 4)}>;  // decoded normal, depth
fn ti(l: vec2i) -> i32 { return (l.y + AP) * SW + l.x + AP; }
${atrousFn({ R: 2, step: 1, prefilter: true })}

struct Down { c: vec4f, g: vec4f };  // colour + var, (oct normal, depth, 0) as stored
fn down(Q: vec2i) -> Down {
  let hs = vec2i(prm.size2);
  var N = 0.0; var col = vec3f(0.0); var vr = 0.0; var nrm = vec3f(0.0); var z = 0.0;
  for (var j = 0; j < 4; j++) {
    let h = Q * 2 + vec2i(j & 1, j >> 1);
    if (h.x >= hs.x || h.y >= hs.y) { continue; }
    let g3 = textureLoad(F3, h, 0);
    let oz = unpack2x16float(g3.y);
    if (oz.y <= 0.0) { continue; }
    let g2 = textureLoad(F2, h, 0);
    let a = vec4f(unpack2x16float(g2.z), unpack2x16float(g2.w));
    let w = oz.y;
    N += w; col += w * a.rgb; vr += w * a.a; nrm += w * octDec(unpack2x16float(g3.x)); z += w * oz.x;
  }
  if (N == 0.0) { return Down(vec4f(0.0), vec4f(0.0, 0.0, -1.0, 0.0)); }
  let o = octEnc(nrm / max(length(nrm), 1e-6));
  return Down(q4(vec4f(col / N, vr / N / N)), vec4f(qf16(o.x), qf16(o.y), qf16(z / N), 0.0));
}

@compute @workgroup_size(${TW}, ${TH})
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  let base = vec2i(wid.xy * vec2u(${TW}u, ${TH}u));
  for (var i = i32(li); i < ${(TW + 4) * (TH + 4)}; i += ${TW * TH}) {
    let q = clampP(base - AP + vec2i(i % SW, i / SW));
    let d = down(q);
    sA[i] = d.c;
    sG[i] = vec4f(octDec(d.g.xy), d.g.z);
    let lq = base - AP + vec2i(i % SW, i / SW);
    if (all(lq == q) && all(lq >= base) && all(lq < base + vec2i(${TW}, ${TH}))) { textureStore(outG, q, d.g); }
  }
  workgroupBarrier();
  let l = vec2i(lid.xy);
  let p = base + l;
  if (!inside(p)) { return; }
  if (sG[ti(l)].w < 0.0) { textureStore(outC, p, vec4f(0.0)); return; }
  textureStore(outC, p, filt(l, p, prm.s, 1.0));
}
`;



// 4-6. edge-aware a-trous, kernel (2R+1)^2, as a WGSL function over a workgroup tile:
//   fn filt(l: vec2i, q: vec2i, s: f32, c: f32) -> vec4f   one iteration at pixel q (tile coords l)
// colour from sA, decoded guide (normal, depth) from sG. The tile holds clamp-to-edge loads, so the variance
// prefilter / depth gradient match textureLoad(clampP(.)); taps outside the image are skipped (as in
// training). Taps are unrolled with constant offsets / weights; divisions are hoisted.
function atrousFn({ R, step, prefilter, name = 'filt' }) {
  const h = R === 1 ? [0.25, 0.5, 0.25] : [1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16];
  const dists = [...new Set([...Array(2 * R + 1).keys()].flatMap((y) => [...Array(2 * R + 1).keys()]
    .map((x) => (x - R) ** 2 + (y - R) ** 2)))].filter((d) => d > 0).sort((a, b) => a - b);
  const taps = [];
  for (let dy = -R; dy <= R; dy++) {
    for (let dx = -R; dx <= R; dx++) {
      if (dx === 0 && dy === 0) continue;
      const o = `vec2i(${dx * step}, ${dy * step})`;
      taps.push(`  {
    let gq = sG[ti(l + ${o})];
    if (inside(q + ${o}) && gq.w >= 0.0) {
      let cq = sA[ti(l + ${o})];
      let w = ${lit(h[dx + R] * h[dy + R])} * pow128(max(dot(gq.xyz, gp.xyz), 0.0))
        * exp(-abs(gq.w - gp.w) * iz${dists.indexOf(dx * dx + dy * dy)} - sl * abs(lum(cq.rgb) - lc));
      sw += w; sc += w * cq.rgb; sv += w * w * cq.a;
    }
  }`);
    }
  }
  return /* wgsl */ `
fn ${name}(l: vec2i, q: vec2i, s: f32, c: f32) -> vec4f {
  let gp = sG[ti(l)];
  let center = sA[ti(l)];
  ${prefilter ? `let gv = 0.0625 * (sA[ti(l + vec2i(-1, -1))].a + sA[ti(l + vec2i(1, -1))].a + sA[ti(l + vec2i(-1, 1))].a + sA[ti(l + vec2i(1, 1))].a)
    + 0.125 * (sA[ti(l + vec2i(0, -1))].a + sA[ti(l + vec2i(-1, 0))].a + sA[ti(l + vec2i(1, 0))].a + sA[ti(l + vec2i(0, 1))].a)
    + 0.25 * center.a;` : 'let gv = center.a;'}
  let lc = lum(center.rgb);
  let sl = s / (sqrt(max(gv, 1e-12)) + 1e-6);
  let zl = sG[ti(l - vec2i(1, 0))].w;
  let zr = sG[ti(l + vec2i(1, 0))].w;
  let zu = sG[ti(l - vec2i(0, 1))].w;
  let zd = sG[ti(l + vec2i(0, 1))].w;
  let gx = min(select(1e9, abs(zr - gp.w), zr > 0.0), select(1e9, abs(zl - gp.w), zl > 0.0));
  let gy = min(select(1e9, abs(zd - gp.w), zd > 0.0), select(1e9, abs(zu - gp.w), zu > 0.0));
  let zg = min(max(gx, gy), gp.w);
  ${dists.map((d, k) => `let iz${k} = 1.0 / (zg * ${lit(Math.sqrt(d) * step)} + 1e-3);`).join('\n  ')}
  let cw = ${lit(h[R] * h[R])} * c;
  var sw = cw;
  var sc = center.rgb * cw;
  var sv = center.a * cw * cw;
${taps.join('\n')}
  return vec4f(sc / sw, sv / (sw * sw));
}
`;
}


// a-trous pass: tile load (+ apron) then one iteration. kind:
//   'coarse' quarter res, fixed s;  'mid1' / 'mid2' half res (source: packed block means / previous mid output,
//   guide: packed block guide), learned s_1,2 c_1,2;  'fine' full res, learned s_0 c_0 (output = colour history)
function atrousShader({ R, step, kind, prefilter = false }) {
  const ap = Math.max(R * step, 1), sw = TW + 2 * ap, sh = TH + 2 * ap;
  const b = (i) => `@group(0) @binding(${i})`;
  const half = kind === 'mid1' || kind === 'mid2' || kind === 'mid3';
  const binds = [
    `${b(1)} var inA: texture_2d<${kind === 'mid1' ? 'u32' : 'f32'}>;`,
    `${b(2)} var guide: texture_2d<${half ? 'u32' : 'f32'}>;`,
    `${b(3)} var outA: texture_storage_2d<rgba16float, write>;`,
    ...(kind === 'coarse' ? [] : [`${b(4)} var par1: texture_2d<f32>;`, `${b(5)} var par2: texture_2d<f32>;`]),
  ];
  const loadA = kind === 'mid1'
    ? 'let a2 = textureLoad(inA, q, 0); sA[i] = vec4f(unpack2x16float(a2.z), unpack2x16float(a2.w));'
    : 'sA[i] = textureLoad(inA, q, 0);';
  const loadG = half
    ? 'let g3 = textureLoad(guide, q, 0); sG[i] = vec4f(octDec(unpack2x16float(g3.x)), unpack2x16float(g3.y).x);'
    : 'let g = textureLoad(guide, q, 0); sG[i] = vec4f(octDec(g.xy), g.z);';
  const pp = kind === 'fine' ? 'p >> vec2u(1u)' : 'p';
  const comp = { fine: 'x', mid1: 'y', mid2: 'z', mid3: 'w' }[kind];
  const params = kind === 'coarse' ? 'prm.s, 1.0' : `textureLoad(par1, ${pp}, 0).${comp}, textureLoad(par2, ${pp}, 0).${comp}`;
  return COMMON + TILE_FNS + `
${binds.join('\n')}
const AP = ${ap};
const SW = ${sw};
var<workgroup> sA: array<vec4f, ${sw * sh}>;
var<workgroup> sG: array<vec4f, ${sw * sh}>;  // decoded normal, depth
fn ti(l: vec2i) -> i32 { return (l.y + AP) * SW + l.x + AP; }
${atrousFn({ R, step, prefilter })}
@compute @workgroup_size(${TW}, ${TH})
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  let base = vec2i(wid.xy * vec2u(${TW}u, ${TH}u));
  for (var i = i32(li); i < ${sw * sh}; i += ${TW * TH}) {
    let q = clampP(base - AP + vec2i(i % SW, i / SW));
    ${loadA}
    ${loadG}
  }
  workgroupBarrier();
  let l = vec2i(lid.xy);
  let p = base + l;
  if (!inside(p)) { return; }
  if (sG[ti(l)].w < 0.0) { textureStore(outA, p, vec4f(0.0)); return; }
  textureStore(outA, p, filt(l, p, ${params}));
}
`;
}

// 5+6a. fine level (one 3x3 a-trous on the temporal output, learned s_0 / c_0; also the colour history), then
// recombine the levels at full res and remodulate:
//   ill = mid + kappa (fine - mid source)  [half-res terms joint-bilateral upsampled], lerp to the coarse level by beta
const FINE_FINAL = COMMON + TILE_FNS + /* wgsl */ `
@group(0) @binding(1) var inA: texture_2d<f32>;
@group(0) @binding(2) var guide: texture_2d<f32>;
@group(0) @binding(3) var par0: texture_2d<f32>;
@group(0) @binding(4) var par1: texture_2d<f32>;
@group(0) @binding(5) var par2: texture_2d<f32>;
@group(0) @binding(6) var F2: texture_2d<u32>;
@group(0) @binding(7) var F3: texture_2d<u32>;
@group(0) @binding(8) var mid: texture_2d<f32>;
@group(0) @binding(9) var coarse: texture_2d<f32>;
@group(0) @binding(10) var cguide: texture_2d<f32>;
@group(0) @binding(11) var material: texture_2d<f32>;
@group(0) @binding(12) var outHist: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var outX: texture_storage_2d<rgba16float, write>;
@group(0) @binding(14) var emission: texture_2d<f32>;
const AP = 1;
const SW = ${TW + 2};
var<workgroup> sA: array<vec4f, ${(TW + 2) * (TH + 2)}>;
var<workgroup> sG: array<vec4f, ${(TW + 2) * (TH + 2)}>;  // decoded normal, depth
fn ti(l: vec2i) -> i32 { return (l.y + AP) * SW + l.x + AP; }
${atrousFn({ R: 1, step: 1, prefilter: true })}

fn jbw(fr: vec2f, j: i32, nq: vec3f, zq: f32, n: vec3f, z: f32) -> f32 {
  let wb = select(1.0 - fr.x, fr.x, (j & 1) == 1) * select(1.0 - fr.y, fr.y, (j >> 1) == 1);
  let dn = max(dot(nq, n), 0.0);
  let dn2 = dn * dn;
  let dn4 = dn2 * dn2;
  return wb * exp(-abs(zq - z) / (0.05 * z + 1e-3)) * dn4 * dn4;
}

@compute @workgroup_size(${TW}, ${TH})
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  let base = vec2i(wid.xy * vec2u(${TW}u, ${TH}u));
  for (var i = i32(li); i < ${(TW + 2) * (TH + 2)}; i += ${TW * TH}) {
    let q = clampP(base - AP + vec2i(i % SW, i / SW));
    sA[i] = textureLoad(inA, q, 0);
    let g = textureLoad(guide, q, 0);
    sG[i] = vec4f(octDec(g.xy), g.z);
  }
  workgroupBarrier();
  let l = vec2i(lid.xy);
  let p = base + l;
  if (!inside(p)) { return; }
  let alb = textureLoad(material, p, 0).rgb;
  let emi = textureLoad(emission, p, 0).rgb;
  let gp = sG[ti(l)];
  if (gp.w < 0.0) { textureStore(outHist, p, vec4f(0.0)); textureStore(outX, p, vec4f(emi, 0.0)); return; }
  let hp = p >> vec2u(1u);
  let A1 = filt(l, p, textureLoad(par1, hp, 0).x, textureLoad(par2, hp, 0).x);
  textureStore(outHist, p, A1);
  let f = vec3f(qf16(A1.r), qf16(A1.g), qf16(A1.b));
  let n = gp.xyz;
  let z = gp.w;
  let pr = textureLoad(par0, hp, 0);
  // half-res mid level
  let hs = vec2i((prm.size + 1u) / 2u);
  let hpos = (vec2f(p) + 0.5) / 2.0 - 0.5;
  let hb = floor(hpos);
  var uB = vec3f(0.0); var uA = vec3f(0.0); var swh = 0.0;
  for (var j = 0; j < 4; j++) {
    let q = vec2i(hb) + vec2i(j & 1, j >> 1);
    if (q.x < 0 || q.y < 0 || q.x >= hs.x || q.y >= hs.y) { continue; }
    let g3 = textureLoad(F3, q, 0);
    let oz = unpack2x16float(g3.y);
    if (oz.x < 0.0) { continue; }
    let w = jbw(hpos - hb, j, octDec(unpack2x16float(g3.x)), oz.x, n, z);
    let a2 = textureLoad(F2, q, 0);
    swh += w;
    uB += w * textureLoad(mid, q, 0).rgb;
    uA += w * vec3f(unpack2x16float(a2.z), unpack2x16float(a2.w).x);
  }
  var ill = f;
  if (swh > 1e-4) { ill = uB / swh + pr.w * (f - uA / swh); }
  // quarter-res coarse level
  let qs = vec2i((prm.size + 3u) / 4u);
  let cpos = (vec2f(p) + 0.5) / 4.0 - 0.5;
  let cb = floor(cpos);
  var uC = vec3f(0.0); var swq = 0.0;
  for (var j = 0; j < 4; j++) {
    let q = vec2i(cb) + vec2i(j & 1, j >> 1);
    if (q.x < 0 || q.y < 0 || q.x >= qs.x || q.y >= qs.y) { continue; }
    let cg = textureLoad(cguide, q, 0);
    if (cg.z < 0.0) { continue; }
    let w = jbw(cpos - cb, j, octDec(cg.xy), cg.z, n, z);
    swq += w;
    uC += w * textureLoad(coarse, q, 0).rgb;
  }
  var up = ill;
  if (swq > 1e-4) { up = uC / swq; }
  ill = ill + (up - ill) * pr.z;
  textureStore(outX, p, vec4f(ill * alb + emi, 0.0));
}
`;

// 6. TAA (YCoCg variance clip, Karis tonemapped blend), manual bilinear history
const TAA = COMMON + /* wgsl */ `
@group(0) @binding(1) var X: texture_2d<f32>;
@group(0) @binding(2) var motion: texture_2d<f32>;
@group(0) @binding(3) var prevTaa: texture_2d<f32>;
@group(0) @binding(4) var outTaa: texture_storage_2d<rgba16float, write>;

fn toYCoCg(c: vec3f) -> vec3f {
  return vec3f(c.r * 0.25 + c.g * 0.5 + c.b * 0.25, c.r * 0.5 - c.b * 0.5, -c.r * 0.25 + c.g * 0.5 - c.b * 0.25);
}
fn fromYCoCg(c: vec3f) -> vec3f { return vec3f(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }
fn tm(c: vec3f) -> vec3f { return c / (1.0 + lum(c)); }
fn itm(c: vec3f) -> vec3f { return c / max(1.0 - lum(c), 1e-4); }
${tileDecl(1, 'var<workgroup> sX: array<vec3f, SW * SH>;  // YCoCg(tm(X)), clamp-to-edge')}

@compute @workgroup_size(${TW}, ${TH})
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  let origin = vec2i(wid.xy * vec2u(${TW}u, ${TH}u)) - AP;
  for (var i = i32(li); i < SW * SH; i += ${TW * TH}) {
    sX[i] = toYCoCg(tm(textureLoad(X, clampP(origin + vec2i(i % SW, i / SW)), 0).rgb));
  }
  workgroupBarrier();
  let p = vec2i(wid.xy * vec2u(${TW}u, ${TH}u) + lid.xy);
  if (p.x >= i32(prm.size.x) || p.y >= i32(prm.size.y)) { return; }
  let lp = vec2i(lid.xy);
  let cur = tm(textureLoad(X, p, 0).rgb);
  if ((prm.flags & 2u) != 0u) { textureStore(outTaa, p, vec4f(itm(cur), 1.0)); return; }
  var m1 = vec3f(0.0);
  for (var j = 0; j < 9; j++) { m1 += sX[sIdx(lp + vec2i(j % 3 - 1, j / 3 - 1))]; }
  m1 /= 9.0;
  var v3 = vec3f(0.0);
  for (var j = 0; j < 9; j++) { let d = sX[sIdx(lp + vec2i(j % 3 - 1, j / 3 - 1))] - m1; v3 += d * d; }
  let sd = sqrt(max(v3 / 9.0, vec3f(1e-12)));
  let mv = textureLoad(motion, p, 0).xy;
  let sz = vec2f(prm.size);
  let uv = (vec2f(p) + 0.5 + mv) / sz;
  var res = cur;
  if (all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0))) {
    let pp = uv * sz - 0.5;
    let b = floor(pp);
    let f = pp - b;
    var hist = vec3f(0.0);
    for (var j = 0; j < 4; j++) {
      let o = vec2i(j & 1, j >> 1);
      let wb = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
      hist += wb * textureLoad(prevTaa, clampP(vec2i(b) + o), 0).rgb;
    }
    var hh = toYCoCg(tm(hist));
    hh = clamp(hh, m1 - 1.25 * sd, m1 + 1.25 * sd);
    let hr = fromYCoCg(hh);
    res = hr + (cur - hr) * 0.1;
  }
  textureStore(outTaa, p, vec4f(itm(res), 1.0));
}
`;


// Unrolled MLP: f (array<f32, N_FEAT>) -> o_0..o_k (vec4f; outputs >= N_OUT are 0). w: Float32Array (hidden width h).
// f16: half-precision arithmetic (shader-f16; the weights are f16 anyway), twice the FMA rate on most GPUs.
export function mlpCode(w, HIDDEN, f16 = false) {
  const W1 = 0, B1 = HIDDEN * N_FEAT, W2 = B1 + HIDDEN, B2 = W2 + HIDDEN * HIDDEN, W3 = B2 + HIDDEN, B3 = W3 + N_OUT * HIDDEN;
  const layer = (name, nOut, nIn, wOff, bOff, input, relu) => {
    const lines = [];
    for (let j = 0; j < nOut; j += 4) {
      const col = (i) => [0, 1, 2, 3].map((r) => (j + r < nOut ? w[wOff + (j + r) * nIn + i] : 0));
      const bias = [0, 1, 2, 3].map((r) => (j + r < nOut ? w[bOff + j + r] : 0));
      const vec = f16 ? (a) => `vec4h(${a.map((x) => `${lit(x)}h`).join(', ')})` : v4;
      const terms = Array.from({ length: nIn }, (_, i) => `${input(i)} * ${vec(col(i))}`);
      const e = `${vec(bias)} + ${terms.join(' + ')}`;
      lines.push(`let ${name}_${j / 4} = ${relu ? `max(${e}, ${f16 ? 'vec4h(0.0h)' : 'vec4f(0.0)'})` : e};`);
    }
    return lines;
  };
  const comp = (n) => (i) => `${n}_${i >> 2}.${'xyzw'[i & 3]}`;
  return [
    ...layer('h1', HIDDEN, N_FEAT, W1, B1, (i) => (f16 ? `f16(f[${i}])` : `f[${i}]`), true),
    ...layer('h2', HIDDEN, HIDDEN, W2, B2, comp('h1'), true),
    ...layer(f16 ? 'oh' : 'o', N_OUT, HIDDEN, W3, B3, comp('h2'), false),
    ...(f16 ? Array.from({ length: Math.ceil(N_OUT / 4) }, (_, k) => `let o_${k} = vec4f(oh_${k});`) : []),
  ].join('\n  ');
}

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

function decodeWeights(w) {
  if (typeof w === 'string') return Uint8Array.from(atob(w), (c) => c.charCodeAt(0)).buffer;
  if (ArrayBuffer.isView(w)) return w.buffer.slice(w.byteOffset, w.byteOffset + w.byteLength);
  return w;
}

// Device limits Hush needs beyond the WebGPU defaults; request them when creating the device, e.g.
// new WebGPURenderer({ requiredLimits: Hush.requiredLimits }). The temporal pass writes 7 storage textures.
const REQUIRED_LIMITS = Object.freeze({ maxStorageTexturesPerShaderStage: 8 });

const INPUTS = ['radiance', 'normalDepth', 'motion', 'material', 'emission'];

export class Hush {
  static requiredLimits = REQUIRED_LIMITS;

  // options: { width, height, exposure = 1, weights = built-in (ArrayBuffer, typed array or base64 string of the f16
  // values the export script writes; the MLP width follows from the count), f16 = false (MLP in shader-f16; no gain on
  // RTX 3060) }
  constructor(device, options = {}) {
    const { width, height, exposure = 1, weights = DEFAULT_WEIGHTS, f16 = false } = options;
    for (const [k, v] of Object.entries(REQUIRED_LIMITS)) {
      if (!(device.limits[k] >= v)) {
        throw new Error(`Hush needs the device limit ${k} >= ${v} (this device: ${device.limits[k]}); create the device with requiredLimits: Hush.requiredLimits`);
      }
    }
    this.device = device;
    this.exposure = exposure;
    this.f16 = f16 && device.features.has('shader-f16');
    const u16 = new Uint16Array(decodeWeights(weights));
    this.hidden = hiddenOf(u16.length);
    if (!this.hidden) throw new Error(`hush weights: ${u16.length} f16 values fit no MLP size`);
    const wf = Float32Array.from(u16, halfToFloat);
    const pipe = (code) => device.createComputePipeline({
      layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });
    this.p = {
      temporal: pipe(TEMPORAL),
      infer: pipe((this.f16 ? 'enable f16;\n' : '') + INFER.replace('MLP_CODE', mlpCode(wf, this.hidden, this.f16))),
      coarse1: pipe(COARSE1),
      coarse2: pipe(atrousShader({ R: 2, step: 2, kind: 'coarse', prefilter: true })),
      mid2: pipe(atrousShader({ R: 1, step: 2, kind: 'mid2' })),
      mid3: pipe(atrousShader({ R: 1, step: 4, kind: 'mid3' })),
      fineFinal: pipe(FINE_FINAL),
      taa: pipe(TAA),
    };
    this.ubos = Array.from({ length: 10 }, () => device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    this.textures = [];
    this.resize(width, height);
  }

  // Reallocates the internal textures for a new input size (no-op if unchanged) and resets the history.
  resize(width, height) {
    if (!(width > 0 && height > 0)) throw new Error(`Hush: bad size ${width}x${height}`);
    if (width === this.width && height === this.height) return;
    for (const t of this.textures) t.destroy();
    const device = this.device;
    this.width = width;
    this.height = height;
    this.frame = 0;
    this.resetHistory = true;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    const all = this.textures = [];
    const tex = (w, h, format = 'rgba16float') => {
      const t = device.createTexture({ size: [w, h], format, usage });
      all.push(t);
      return t;
    };
    const W = width, H = height, W2 = Math.ceil(W / 2), H2 = Math.ceil(H / 2), W4 = Math.ceil(W2 / 2), H4 = Math.ceil(H2 / 2);
    this.half = [W2, H2];
    this.quarter = [W4, H4];
    this.A0 = tex(W, H); this.X = tex(W, H);
    this.Bh = [tex(W2, H2), tex(W2, H2)];
    this.mom = [tex(W, H), tex(W, H)];
    this.key = [tex(W, H), tex(W, H)];
    this.histColor = [tex(W, H), tex(W, H)];
    this.taa = [tex(W, H), tex(W, H)];
    this.par0 = [tex(W2, H2), tex(W2, H2)];
    this.par1 = tex(W2, H2); this.par2 = tex(W2, H2);
    this.F = [0, 1, 2, 3].map(() => tex(W2, H2, 'rgba32uint'));
    this.C = [tex(W4, H4), tex(W4, H4)];
    this.CG = tex(W4, H4);
  }

  // Output of the last denoise(): rgba16float, linear HDR, antialiased (TAA), input size. Two textures alternate
  // between frames, so fetch it after every call.
  get output() { return this.taa[(this.frame + 1) & 1]; }

  // Drops the history (camera cut, scene switch); the next frame starts from scratch.
  reset() { this.resetHistory = true; }

  destroy() {
    for (const t of this.textures) t.destroy();
    for (const b of this.ubos) b.destroy();
    this.textures = [];
  }

  // Records one frame into `encoder` and returns the output texture.
  // inputs = { radiance, normalDepth, motion, material, emission }: rgba16float GPUTextures with TEXTURE_BINDING usage,
  // the denoiser's size, layout as in src/index.d.ts (extra fields are ignored). Uniforms go through queue.writeBuffer,
  // so submit the encoder before the next denoise().
  denoise(encoder, inputs, timer) {
    for (const k of INPUTS) {
      const t = inputs[k];
      if (!t) throw new Error(`Hush.denoise: missing input '${k}'`);
      if (t.width !== this.width || t.height !== this.height) {
        throw new Error(`Hush.denoise: '${k}' is ${t.width}x${t.height}, the denoiser ${this.width}x${this.height} (call resize)`);
      }
    }
    return this.encode(encoder, inputs, timer);
  }

  writeParams(i, { size = [this.width, this.height], s = 0, size2 = [0, 0] } = {}) {
    const b = new ArrayBuffer(32);
    const u = new Uint32Array(b), f = new Float32Array(b);
    u[0] = size[0]; u[1] = size[1]; u[2] = this.resetHistory ? 2 : 0;
    f[4] = this.exposure; f[5] = s; u[6] = size2[0]; u[7] = size2[1];
    this.device.queue.writeBuffer(this.ubos[i], 0, b);
    return this.ubos[i];
  }

  // denoise() without the input checks. timer (optional, src/gpu.js GpuTimer): { desc(label) } times every pass;
  // { span: true, spanBegin, spanEnd } times the whole denoiser as one span
  encode(encoder, g, timer) {
    const d = this.device;
    const cur = this.frame & 1, prev = cur ^ 1;
    const v = (t) => t.createView();
    const tiles = (s) => [Math.ceil(s[0] / TW), Math.ceil(s[1] / TH)];
    const full = tiles([this.width, this.height]), qt = tiles(this.quarter);
    let nPass = 0;
    const run = (pipeline, ubo, entries, n, label) => {
      const bg = d.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: ubo } }, ...entries.map((r, i) => ({ binding: i + 1, resource: v(r) }))],
      });
      let desc;
      if (timer?.span) desc = nPass === 0 ? timer.spanBegin('total') : label === 'taa' ? timer.spanEnd() : undefined;
      else if (timer) desc = timer.desc(label);
      nPass++;
      const pass = encoder.beginComputePass(desc);
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(n[0], n[1]);
      pass.end();
    };
    const P = this.p;
    run(P.temporal, this.writeParams(0), [g.radiance, g.normalDepth, g.motion, g.material, this.key[prev], this.histColor[prev],
      this.mom[prev], this.par0[prev], this.A0, this.mom[cur], this.key[cur], ...this.F], full, 'temporal');
    const ht = [Math.ceil(this.half[0] / TW), Math.ceil(this.half[1] / TH)];
    run(P.infer, this.writeParams(1, { size: this.half }), [...this.F, this.par0[cur], this.par1, this.par2, this.Bh[0]], ht, 'infer+mid');
    run(P.mid2, this.writeParams(5, { size: this.half }), [this.Bh[0], this.F[3], this.Bh[1], this.par1, this.par2], ht, 'infer+mid');
    run(P.mid3, this.writeParams(9, { size: this.half }), [this.Bh[1], this.F[3], this.Bh[0], this.par1, this.par2], ht, 'infer+mid');
    const qd = { size: this.quarter, s: COARSE_S, size2: this.half };
    run(P.coarse1, this.writeParams(3, qd), [this.F[2], this.F[3], this.C[1], this.CG], qt, 'coarse');
    run(P.coarse2, this.writeParams(4, qd), [this.C[1], this.CG, this.C[0]], qt, 'coarse');
    const k = this.key[cur];
    run(P.fineFinal, this.writeParams(6), [this.A0, k, this.par0[cur], this.par1, this.par2, this.F[2], this.F[3], this.Bh[0],
      this.C[0], this.CG, g.material, this.histColor[cur], this.X, g.emission], full, 'fine+final');
    run(P.taa, this.writeParams(8), [this.X, g.motion, this.taa[prev], this.taa[cur]], full, 'taa');
    this.resetHistory = false;
    this.frame++;
    return this.taa[cur];
  }
}
