// WGSL source of the path tracer (a JS module so the package needs no bundler plugin).
export default /* wgsl */ `
// Minimal WebGPU path tracer for Hush: BVH over world-space triangles (several BLAS, dynamic ones
// refit on the CPU), GGX metal/rough + smooth glass, analytic sky or an importance-sampled equirect environment map
// (envmap.js), NEE + MIS on cone lights (sun + sphere lights) and on the environment map.
//
// mode 0 (noisy): 1 path per pixel with the per-frame subpixel jitter; writes the G-buffer.
// mode 1 (reference): "spp" paths per pixel with box-filter jitter, accumulated into "accum"
//   (full radiance incl. primary emission). The G-buffer is not touched.

const PI = 3.14159265358979;
const MAX_BLAS = 8u;
const MAX_LIGHTS = 8u;
const INF = 1e30;
const ENV_GW = 256u;  // environment importance grid (envmap.js ENV_GRID)
const ENV_GH = 128u;

struct Blas { nodeOffset: u32, triOffset: u32, nodeCount: u32, flags: u32 };  // flags: 1 = dynamic, 2 = no shadows
// kind 0: sun / directional cone (dir in pos.xyz), 1: sphere light
struct Light { pos: vec4f, prevPos: vec4f, radiance: vec4f, params: vec4f }; // pos.w = radius, params.x = kind, params.y = cosMax (sun)

struct Frame {
  camPos: vec4f,
  camRight: vec4f,     // xyz * tanHalfFov * aspect
  camUp: vec4f,        // xyz * tanHalfFov
  camFwd: vec4f,
  viewProj: mat4x4f,
  prevViewProj: mat4x4f,
  size: vec2u,
  frame: u32,
  mode: u32,
  jitter: vec2f,       // pixels, [-0.5, 0.5]
  spp: u32,
  maxBounces: u32,
  skyZenith: vec4f,
  skyHorizon: vec4f,   // w = sky intensity
  groundColor: vec4f,
  counts: vec4u,       // x = blas count, y = light count, z = sample offset (reference)
  clampLum: f32,
  pixelAngle: f32,
  shadowTau: f32,      // > 0: secondary bounces cast one shadow ray (light picked by unoccluded contribution, RR below tau)
  clampDirect: f32,    // > 0: clamp on direct light at the primary hit too (glints on small glossy parts)
  env: vec4f,          // environment map: x = intensity, y / z = cos / sin of its rotation about +y, w = 1 if bound
  blas: array<Blas, MAX_BLAS>,
  lights: array<Light, MAX_LIGHTS>,
};

struct Vertex { pos: vec4f, prev: vec4f, nrm: vec4f }; // pos.w = u, nrm.w = v
struct Material {
  base: vec4f,       // rgb, w = alpha cutoff (0 = opaque)
  emissive: vec4f,   // rgb * strength
  pbr: vec4f,        // roughness, metalness, transmission, ior
  tex: vec4i,        // base layer, normal layer, arm layer, flags
  uv: vec4f,         // uv scale xy, normal scale z
};

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> nodes: array<vec4f>;
@group(0) @binding(2) var<storage, read> tris: array<vec4f>;   // per triangle: [v0, alpha-tested], [e1], [e2]
@group(0) @binding(3) var<storage, read> triIdx: array<vec4u>;
@group(0) @binding(4) var<storage, read> verts: array<Vertex>;
@group(0) @binding(5) var<storage, read> mats: array<Material>;
@group(0) @binding(6) var<storage, read_write> accum: array<vec4f>;
@group(0) @binding(7) var texArr: texture_2d_array<f32>;
@group(0) @binding(8) var samp: sampler;
@group(0) @binding(9) var outRadiance: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var outNormalDepth: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var outMotion: texture_storage_2d<rgba16float, write>;
@group(0) @binding(12) var outMaterial: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var outEmission: texture_storage_2d<rgba16float, write>;
@group(0) @binding(14) var envTex: texture_2d<f32>;
@group(0) @binding(15) var<storage, read> envTable: array<vec4f>;  // alias table: threshold, alias (bits), pmf, 0
@group(0) @binding(16) var envSamp: sampler;

// ---------------------------------------------------------------- random
var<private> rngState: u32;
fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd() -> f32 {
  rngState = pcg(rngState);
  return f32(rngState >> 8u) * (1.0 / 16777216.0);
}

fn lum(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

// ---------------------------------------------------------------- traversal
struct Hit { t: f32, tri: u32, u: f32, v: f32, light: i32 };


fn slab(o: vec3f, inv: vec3f, bmin: vec3f, bmax: vec3f, tmax: f32) -> f32 {
  let t0 = (bmin - o) * inv;
  let t1 = (bmax - o) * inv;
  let tn = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
  let tf = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  if (tn <= tf && tf > 0.0 && tn < tmax) { return max(tn, 0.0); }
  return INF;
}

fn alphaPass(tri: u32, u: f32, v: f32) -> bool {
  let ti = triIdx[tri];
  let m = mats[ti.w & 0xffffu];
  if (m.base.w <= 0.0 || m.tex.x < 0) { return true; }
  let uv = (verts[ti.x].pos.w * (1.0 - u - v) + verts[ti.y].pos.w * u + verts[ti.z].pos.w * v);
  let uv2 = vec2f(uv, verts[ti.x].nrm.w * (1.0 - u - v) + verts[ti.y].nrm.w * u + verts[ti.z].nrm.w * v);
  let a = textureSampleLevel(texArr, samp, uv2 * m.uv.xy, m.tex.x, 2.0).a;
  return a >= m.base.w;
}

fn intersectTri(o: vec3f, d: vec3f, tri: u32, tmax: f32) -> vec3f {
  let a = tris[tri * 3u];
  let v0 = a.xyz;
  let e1 = tris[tri * 3u + 1u].xyz;
  let e2 = tris[tri * 3u + 2u].xyz;
  let p = cross(d, e2);
  let det = dot(e1, p);
  if (abs(det) < 1e-12) { return vec3f(INF, 0.0, 0.0); }
  let inv = 1.0 / det;
  let s = o - v0;
  let u = dot(s, p) * inv;
  if (u < 0.0 || u > 1.0) { return vec3f(INF, 0.0, 0.0); }
  let q = cross(s, e1);
  let v = dot(d, q) * inv;
  if (v < 0.0 || u + v > 1.0) { return vec3f(INF, 0.0, 0.0); }
  let t = dot(e2, q) * inv;
  if (t <= 1e-4 || t >= tmax) { return vec3f(INF, 0.0, 0.0); }
  if (a.w > 0.0 && !alphaPass(tri, u, v)) { return vec3f(INF, 0.0, 0.0); }
  return vec3f(t, u, v);
}

// Packed BVH2 (see bvh.js packBVH): every node holds both children's boxes; leaves are tested from the parent.
const STACK = 32u;  // >= BVH depth (scene.js warns otherwise)
var<workgroup> stack: array<u32, 64u * STACK>;  // traversal stacks of the 8x8 workgroup (cheaper than private arrays)
var<private> stackBase: u32;
fn traceBVH(o: vec3f, d: vec3f, tmaxIn: f32, shadow: bool) -> Hit {
  var h: Hit;
  h.t = tmaxIn;
  h.tri = 0xffffffffu;
  h.light = -1;
  let inv = 1.0 / select(d, vec3f(1e-9), abs(d) < vec3f(1e-9));
  for (var b = 0u; b < F.counts.x; b++) {
    let bl = F.blas[b];
    if (shadow && (bl.flags & 2u) != 0u) { continue; }
    var sp = 0u;
    var node = 0u;
    loop {
      let k = (bl.nodeOffset + node) * 4u;
      let lmin = nodes[k];
      let lmax = nodes[k + 1u];
      let rmin = nodes[k + 2u];
      let rmax = nodes[k + 3u];
      var tl = slab(o, inv, lmin.xyz, lmax.xyz, h.t);
      var tr = slab(o, inv, rmin.xyz, rmax.xyz, h.t);
      let lc = bitcast<u32>(lmax.w);
      let rc = bitcast<u32>(rmax.w);
      if (lc > 0u && tl < INF) {
        let first = bl.triOffset + bitcast<u32>(lmin.w);
        for (var i = 0u; i < lc; i++) {
          let r = intersectTri(o, d, first + i, h.t);
          if (r.x < h.t) {
            h.t = r.x; h.u = r.y; h.v = r.z; h.tri = first + i;
            if (shadow) { return h; }
          }
        }
        tl = INF;
      }
      if (rc > 0u && tr < h.t) {
        let first = bl.triOffset + bitcast<u32>(rmin.w);
        for (var i = 0u; i < rc; i++) {
          let r = intersectTri(o, d, first + i, h.t);
          if (r.x < h.t) {
            h.t = r.x; h.u = r.y; h.v = r.z; h.tri = first + i;
            if (shadow) { return h; }
          }
        }
        tr = INF;
      }
      let l = bitcast<u32>(lmin.w);
      let r = bitcast<u32>(rmin.w);
      if (tl < h.t && tr < h.t) {
        if (tl <= tr) { node = l; stack[stackBase + sp] = r; } else { node = r; stack[stackBase + sp] = l; }
        sp = min(sp + 1u, STACK - 1u);
      } else if (tl < h.t) {
        node = l;
      } else if (tr < h.t) {
        node = r;
      } else {
        if (sp == 0u) { break; }
        sp--;
        node = stack[stackBase + sp];
      }
    }
  }
  return h;
}

fn sphereT(o: vec3f, d: vec3f, c: vec3f, r: f32) -> f32 {
  let oc = o - c;
  let b = dot(oc, d);
  let cc = dot(oc, oc) - r * r;
  let disc = b * b - cc;
  if (disc < 0.0) { return INF; }
  let s = sqrt(disc);
  var t = -b - s;
  if (t < 1e-4) { t = -b + s; }
  if (t < 1e-4) { return INF; }
  return t;
}

fn traceScene(o: vec3f, d: vec3f) -> Hit {
  var h = traceBVH(o, d, INF, false);
  for (var i = 0u; i < F.counts.y; i++) {
    let L = F.lights[i];
    if (L.params.x != 1.0) { continue; }
    let t = sphereT(o, d, L.pos.xyz, L.pos.w);
    if (t < h.t) { h.t = t; h.light = i32(i); h.tri = 0xffffffffu; }
  }
  return h;
}

fn visible(o: vec3f, d: vec3f, tmax: f32) -> bool {
  let h = traceBVH(o, d, tmax, true);
  return h.tri == 0xffffffffu;
}

// ---------------------------------------------------------------- environment
fn envLocal(d: vec3f) -> vec3f { return vec3f(F.env.y * d.x - F.env.z * d.z, d.y, F.env.z * d.x + F.env.y * d.z); }
fn envUV(l: vec3f) -> vec2f { return vec2f(atan2(l.z, l.x) * (0.5 / PI) + 0.5, acos(clamp(l.y, -1.0, 1.0)) / PI); }

// solid-angle pdf of sampleEnv for direction d (world)
fn envPdf(d: vec3f) -> f32 {
  let l = envLocal(d);
  let uv = envUV(l);
  let st = sqrt(max(0.0, 1.0 - l.y * l.y));
  if (st < 1e-5) { return 0.0; }
  let c = min(vec2u(uv * vec2f(f32(ENV_GW), f32(ENV_GH))), vec2u(ENV_GW - 1u, ENV_GH - 1u));
  return envTable[c.y * ENV_GW + c.x].z * f32(ENV_GW * ENV_GH) / (2.0 * PI * PI * st);
}

// importance-sampled direction (xyz, world) and its pdf (w; 0 = reject)
fn sampleEnv() -> vec4f {
  let N = ENV_GW * ENV_GH;
  var i = min(u32(rnd() * f32(N)), N - 1u);
  let a = envTable[i];
  if (rnd() >= a.x) { i = bitcast<u32>(a.y); }
  let u = (f32(i % ENV_GW) + rnd()) / f32(ENV_GW);
  let v = (f32(i / ENV_GW) + rnd()) / f32(ENV_GH);
  let phi = (u - 0.5) * 2.0 * PI;
  let th = v * PI;
  let st = sin(th);
  let l = vec3f(cos(phi) * st, cos(th), sin(phi) * st);
  let d = vec3f(F.env.y * l.x + F.env.z * l.z, l.y, -F.env.z * l.x + F.env.y * l.z);
  return vec4f(d, select(0.0, envTable[i].z * f32(N) / (2.0 * PI * PI * st), st > 1e-5));
}

fn sky(d: vec3f) -> vec3f {
  if (F.env.w > 0.5) { return textureSampleLevel(envTex, envSamp, envUV(envLocal(d)), 0.0).rgb * F.env.x; }
  let y = d.y;
  var c: vec3f;
  if (y >= 0.0) {
    c = mix(F.skyHorizon.rgb, F.skyZenith.rgb, pow(clamp(y, 0.0, 1.0), 0.6));
  } else {
    c = mix(F.skyHorizon.rgb, F.groundColor.rgb, clamp(-y * 4.0, 0.0, 1.0));
  }
  return c * F.skyHorizon.w;
}

// ---------------------------------------------------------------- BSDF
fn onb(n: vec3f) -> mat3x3f {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;
  let t = vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
  let bb = vec3f(b, s + n.y * n.y * a, -n.y);
  return mat3x3f(t, bb, n);
}

fn D_ggx(nh: f32, a2: f32) -> f32 {
  let d = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}
fn G1(nx: f32, a2: f32) -> f32 {
  return 2.0 * nx / (nx + sqrt(a2 + (1.0 - a2) * nx * nx));
}
fn fresnel(f0: vec3f, c: f32) -> vec3f {
  return f0 + (1.0 - f0) * pow(1.0 - clamp(c, 0.0, 1.0), 5.0);
}
// Heitz 2018 VNDF sampling (local frame, n = +z)
fn sampleVNDF(v: vec3f, a: f32, u1: f32, u2: f32) -> vec3f {
  let vh = normalize(vec3f(a * v.x, a * v.y, v.z));
  let lensq = vh.x * vh.x + vh.y * vh.y;
  let T1 = select(vec3f(1.0, 0.0, 0.0), vec3f(-vh.y, vh.x, 0.0) / sqrt(lensq), lensq > 0.0);
  let T2 = cross(vh, T1);
  let r = sqrt(u1);
  let phi = 2.0 * PI * u2;
  let t1 = r * cos(phi);
  var t2 = r * sin(phi);
  let s = 0.5 * (1.0 + vh.z);
  t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;
  let nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * vh;
  return normalize(vec3f(a * nh.x, a * nh.y, max(0.0, nh.z)));
}

struct Surf {
  p: vec3f, ng: vec3f, n: vec3f,
  base: vec3f, emissive: vec3f,
  rough: f32, metal: f32, transmission: f32, ior: f32,
  objId: u32, frontFace: bool,
  prevP: vec3f,
};

// Evaluates f * cos and the combined pdf for the diffuse + GGX lobes (local frame).
struct Eval { f: vec3f, pdf: f32 };
fn specProb(s: Surf, nv: f32) -> f32 {
  let f0 = mix(vec3f(0.04), s.base, s.metal);
  let fs = lum(fresnel(f0, nv));
  let dw = (1.0 - s.metal) * lum(s.base) * (1.0 - fs);
  return clamp(fs / max(fs + dw, 1e-4), 0.1, 0.9);
}
fn evalBSDF(s: Surf, v: vec3f, l: vec3f) -> Eval {
  var e: Eval;
  e.f = vec3f(0.0);
  e.pdf = 0.0;
  if (l.z <= 0.0 || v.z <= 0.0) { return e; }
  let a = max(s.rough * s.rough, 0.0025);
  let a2 = a * a;
  let h = normalize(v + l);
  let nh = h.z;
  let vh = dot(v, h);
  let f0 = mix(vec3f(0.04), s.base, s.metal);
  let F = fresnel(f0, vh);
  let D = D_ggx(nh, a2);
  let G = G1(v.z, a2) * G1(l.z, a2);
  let spec = F * D * G / (4.0 * v.z);  // * cos(l) folded in
  let diff = (1.0 - s.metal) * (1.0 - F) * s.base / PI * l.z;
  let ps = specProb(s, v.z);
  let pdfSpec = D * G1(v.z, a2) / (4.0 * v.z);
  let pdfDiff = l.z / PI;
  e.f = spec + diff;
  e.pdf = ps * pdfSpec + (1.0 - ps) * pdfDiff;
  return e;
}

// ---------------------------------------------------------------- surface fetch
fn fetchSurf(h: Hit, o: vec3f, d: vec3f, lodBase: f32, secondary: bool) -> Surf {
  var s: Surf;
  s.p = o + d * h.t;
  if (h.light >= 0) {
    let L = F.lights[h.light];
    s.ng = normalize(s.p - L.pos.xyz);
    s.n = s.ng;
    s.base = vec3f(0.0);
    s.emissive = L.radiance.rgb;
    s.objId = 1000u + u32(h.light);
    s.frontFace = true;
    s.prevP = L.prevPos.xyz + (s.p - L.pos.xyz);
    return s;
  }
  let ti = triIdx[h.tri];
  let m = mats[ti.w & 0xffffu];
  let w = 1.0 - h.u - h.v;
  let A = verts[ti.x];
  let B = verts[ti.y];
  let C = verts[ti.z];
  let e1 = tris[h.tri * 3u + 1u].xyz;
  let e2 = tris[h.tri * 3u + 2u].xyz;
  var ng = normalize(cross(e1, e2));
  var n = normalize(A.nrm.xyz * w + B.nrm.xyz * h.u + C.nrm.xyz * h.v);
  s.frontFace = dot(ng, d) < 0.0;
  if (!s.frontFace) { ng = -ng; }
  if (dot(n, ng) < 0.0) { n = -n; }
  let uv0 = vec2f(A.pos.w, A.nrm.w);
  let uv1 = vec2f(B.pos.w, B.nrm.w);
  let uv2 = vec2f(C.pos.w, C.nrm.w);
  let uv = (uv0 * w + uv1 * h.u + uv2 * h.v) * m.uv.xy;
  // texture LOD from per-triangle texel density and the ray footprint
  let triLod = unpack2x16float(ti.w >> 16u).x;
  let cosv = max(abs(dot(ng, d)), 0.15);
  let lod = max(0.0, triLod + log2(max(h.t * F.pixelAngle / cosv, 1e-6) * lodBase) + log2(max(m.uv.x, 1.0)));
  var base = m.base.rgb;
  if (m.tex.x >= 0) {
    let c = textureSampleLevel(texArr, samp, uv, m.tex.x, lod).rgb;
    base *= pow(c, vec3f(2.2));
  }
  var rough = m.pbr.x;
  var metal = m.pbr.y;
  if (m.tex.z >= 0) {
    let arm = textureSampleLevel(texArr, samp, uv, m.tex.z, lod);
    rough *= arm.g;
    metal *= arm.b;
  }
  if (m.tex.y >= 0) {
    // tangent frame from dP/dUV of this triangle
    let du1 = (uv1 - uv0) * m.uv.xy;
    let du2 = (uv2 - uv0) * m.uv.xy;
    let det = du1.x * du2.y - du1.y * du2.x;
    if (abs(det) > 1e-12) {
      let r = 1.0 / det;
      var t = (e1 * du2.y - e2 * du1.y) * r;
      var bt = (e2 * du1.x - e1 * du2.x) * r;
      t = normalize(t - n * dot(n, t));
      let handed = select(-1.0, 1.0, dot(cross(n, t), bt) >= 0.0);
      let b = cross(n, t) * handed;
      var tn = textureSampleLevel(texArr, samp, uv, m.tex.y, lod).xyz * 2.0 - 1.0;
      tn = vec3f(tn.xy * m.uv.z, tn.z);
      let nm = normalize(t * tn.x + b * tn.y + n * max(tn.z, 1e-3));
      if (dot(nm, ng) > 0.05) { n = nm; }
    }
  }
  s.ng = ng;
  s.n = n;
  s.base = base;
  s.emissive = m.emissive.rgb;
  s.rough = clamp(rough, 0.05, 1.0);
  s.metal = clamp(metal, 0.0, 1.0);
  s.transmission = m.pbr.z;
  s.ior = m.pbr.w;
  s.objId = ti.w & 0xffffu;
  s.prevP = A.prev.xyz * w + B.prev.xyz * h.u + C.prev.xyz * h.v;
  return s;
}

// ---------------------------------------------------------------- lights
// Cone sampling: returns direction (xyz) and pdf (w); zero pdf when not applicable.
fn coneAxis(L: Light, p: vec3f) -> vec4f {
  if (L.params.x == 0.0) { return vec4f(L.pos.xyz, L.params.y); }
  let to = L.pos.xyz - p;
  let d2 = dot(to, to);
  let r2 = L.pos.w * L.pos.w;
  if (d2 <= r2) { return vec4f(0.0, 1.0, 0.0, -2.0); }
  return vec4f(to / sqrt(d2), sqrt(1.0 - r2 / d2));
}
fn conePdf(cosMax: f32) -> f32 { return 1.0 / (2.0 * PI * max(1.0 - cosMax, 1e-7)); }

fn mis(a: f32, b: f32) -> f32 { let a2 = a * a; return a2 / max(a2 + b * b, 1e-20); }

// direct light with MIS (light-sampling half). all = true: one shadow ray per light. Otherwise (secondary bounces
// when F.shadowTau > 0) one light is picked in proportion to its unoccluded contribution and its shadow ray is
// Russian-rouletted below shadowTau; both are unbiased and cut the incoherent shadow rays that dominate the cost.
fn directLight(s: Surf, frame: mat3x3f, v: vec3f, tp: vec3f, all: bool) -> vec3f {
  var acc = vec3f(0.0);
  var wSum = 0.0;
  var pick = vec4f(0.0);   // dir, tmax
  var pickC = vec3f(0.0);
  var pickW = 0.0;
  for (var i = 0u; i < F.counts.y; i++) {
    let L = F.lights[i];
    let ax = coneAxis(L, s.p);
    if (ax.w < -1.0) { continue; }
    let cosMax = ax.w;
    let ct = 1.0 - rnd() * (1.0 - cosMax);
    let st = sqrt(max(0.0, 1.0 - ct * ct));
    let ph = 2.0 * PI * rnd();
    let lf = onb(ax.xyz);
    let dir = normalize(lf * vec3f(cos(ph) * st, sin(ph) * st, ct));
    let ll = transpose(frame) * dir;
    if (ll.z <= 0.0 || dot(dir, s.ng) <= 0.0) { continue; }
    let e = evalBSDF(s, v, ll);
    if (e.pdf <= 0.0 && lum(e.f) <= 0.0) { continue; }
    var tmax = INF;
    if (L.params.x == 1.0) { tmax = sphereT(s.p, dir, L.pos.xyz, L.pos.w); }
    let lp = conePdf(cosMax);
    let c = e.f * L.radiance.rgb / lp * mis(lp, e.pdf);
    if (all) {
      if (!visible(s.p + s.ng * 1e-3, dir, tmax)) { continue; }
      acc += c;
    } else {
      let w = lum(tp * c);
      if (w <= 0.0) { continue; }
      wSum += w;
      if (rnd() * wSum < w) { pick = vec4f(dir, tmax); pickC = c; pickW = w; }
    }
  }
  if (!all && wSum > 0.0) {
    let q = min(1.0, wSum / F.shadowTau);
    if (rnd() < q && visible(s.p + s.ng * 1e-3, pick.xyz, pick.w)) { acc = pickC * (wSum / (pickW * q)); }
  }
  // environment map: one importance-sampled shadow ray, MIS against BSDF sampling (sky hits in tracePath)
  if (F.env.w > 0.5) {
    let es = sampleEnv();
    let ll = transpose(frame) * es.xyz;
    if (es.w > 0.0 && ll.z > 0.0 && dot(es.xyz, s.ng) > 0.0) {
      let e = evalBSDF(s, v, ll);
      if (lum(e.f) > 0.0 && visible(s.p + s.ng * 1e-3, es.xyz, INF)) {
        acc += e.f * sky(es.xyz) / es.w * mis(es.w, e.pdf);
      }
    }
  }
  return acc;
}

// ---------------------------------------------------------------- path
struct PathOut { radiance: vec3f, primaryEmission: vec3f };

var<private> gNormal: vec3f;
var<private> gDepth: f32;
var<private> gMotion: vec2f;
var<private> gObj: f32;
var<private> gAlbedo: vec3f;
var<private> gRough: f32;
var<private> gPrevDepth: f32;  // linear depth of this point in the previous frame's camera (clip w)

fn projectPix(m: mat4x4f, p: vec4f) -> vec2f {
  let c = m * p;
  let ndc = c.xy / c.w;
  return vec2f((ndc.x * 0.5 + 0.5) * f32(F.size.x), (0.5 - ndc.y * 0.5) * f32(F.size.y));
}

fn tracePath(pix: vec2f, writeG: bool) -> PathOut {
  var out: PathOut;
  out.radiance = vec3f(0.0);
  out.primaryEmission = vec3f(0.0);
  let ndc = (pix / vec2f(F.size)) * 2.0 - 1.0;
  var o = F.camPos.xyz;
  var d = normalize(F.camFwd.xyz + F.camRight.xyz * ndc.x - F.camUp.xyz * ndc.y);
  var tp = vec3f(1.0);
  var lastPdf = 0.0;      // BSDF pdf of the last scattering (0 = delta / camera)
  var bounces = 0u;
  var deltaChain = 0u;
  var lodBase = 1.0;
  for (var depth = 0u; depth < 12u; depth++) {
    let h = traceScene(o, d);
    let primary = depth == 0u;
    if (h.tri == 0xffffffffu && h.light < 0) {
      // sky
      var e = sky(d);
      if (F.env.w > 0.5 && lastPdf > 0.0) { e *= mis(lastPdf, envPdf(d)); }
      // sun disk seen by BSDF sampling (MIS with light sampling)
      for (var i = 0u; i < F.counts.y; i++) {
        let L = F.lights[i];
        if (L.params.x == 0.0 && dot(d, L.pos.xyz) >= L.params.y) {
          if (lastPdf > 0.0) { e += L.radiance.rgb * mis(lastPdf, conePdf(L.params.y)); }
          else { e += L.radiance.rgb; }
        }
      }
      if (primary) {
        out.primaryEmission = e;
        if (writeG) {
          gNormal = -d; gDepth = -1.0; gObj = 65535.0; gAlbedo = vec3f(1.0); gRough = 1.0; gPrevDepth = -1.0;
          gMotion = projectPix(F.prevViewProj, vec4f(d, 0.0)) - projectPix(F.viewProj, vec4f(d, 0.0));
        }
      } else {
        var c = tp * e;
        let lc = lum(c);
        if (lc > F.clampLum) { c *= F.clampLum / lc; }
        out.radiance += c;
      }
      break;
    }
    let s = fetchSurf(h, o, d, lodBase, !primary);
    // emission (MIS against light sampling for sphere lights)
    if (lum(s.emissive) > 0.0) {
      var e = s.emissive;
      if (h.light >= 0 && lastPdf > 0.0) {
        let ax = coneAxis(F.lights[h.light], o);
        e *= mis(lastPdf, conePdf(ax.w));
      }
      if (primary) { out.primaryEmission = e; }
      else {
        var c = tp * e;
        let lc = lum(c);
        if (lc > F.clampLum) { c *= F.clampLum / lc; }
        out.radiance += c;
      }
    }
    let isGlass = s.transmission > 0.5;
    if (primary && writeG) {
      gNormal = s.n;
      gDepth = dot(s.p - F.camPos.xyz, F.camFwd.xyz);
      gObj = f32(s.objId);
      gAlbedo = select(max(s.base * (1.0 - s.metal) + s.metal * mix(vec3f(0.04), s.base, s.metal), vec3f(0.02)), vec3f(1.0), isGlass || h.light >= 0);
      gRough = select(s.rough, 0.0, isGlass);
      gMotion = projectPix(F.prevViewProj, vec4f(s.prevP, 1.0)) - projectPix(F.viewProj, vec4f(s.p, 1.0));
      gPrevDepth = (F.prevViewProj * vec4f(s.prevP, 1.0)).w;
    }
    if (h.light >= 0) { break; }

    if (isGlass) {
      deltaChain++;
      if (deltaChain > 8u) { break; }
      let eta = select(s.ior, 1.0 / s.ior, s.frontFace);
      let n = s.ng;
      let c = -dot(d, n);
      let k = 1.0 - eta * eta * (1.0 - c * c);
      var R = 1.0;
      if (k > 0.0) {
        let ct = sqrt(k);
        let rs = (eta * c - ct) / (eta * c + ct);
        let rp = (c - eta * ct) / (c + eta * ct);
        R = 0.5 * (rs * rs + rp * rp);
      }
      if (rnd() < R) {
        d = reflect(d, n);
        o = s.p + n * 1e-3;
      } else {
        d = normalize(eta * d + (eta * c - sqrt(k)) * n);
        o = s.p - n * 1e-3;
        tp *= s.base;
      }
      lastPdf = 0.0;
      lodBase *= 1.0;
      continue;
    }

    if (bounces >= F.maxBounces) { break; }
    bounces++;
    let frame = onb(s.n);
    let v = transpose(frame) * (-d);
    if (v.z <= 0.0) { break; }
    // direct light
    var dl = tp * directLight(s, frame, v, tp, primary || F.shadowTau <= 0.0);
    let cl = select(F.clampLum, select(INF, F.clampDirect, F.clampDirect > 0.0), primary);
    let ld = lum(dl);
    if (ld > cl) { dl *= cl / ld; }
    out.radiance += dl;
    // sample the BSDF
    let ps = specProb(s, v.z);
    var l: vec3f;
    if (rnd() < ps) {
      let a = max(s.rough * s.rough, 0.0025);
      let hh = sampleVNDF(v, a, rnd(), rnd());
      l = reflect(-v, hh);
    } else {
      let r = sqrt(rnd());
      let ph = 2.0 * PI * rnd();
      l = vec3f(r * cos(ph), r * sin(ph), sqrt(max(0.0, 1.0 - r * r)));
    }
    if (l.z <= 0.0) { break; }
    let e = evalBSDF(s, v, l);
    if (e.pdf <= 0.0) { break; }
    tp *= e.f / e.pdf;
    lastPdf = e.pdf;
    d = normalize(frame * l);
    if (dot(d, s.ng) <= 0.0) { break; }
    o = s.p + s.ng * 1e-3;
    lodBase = 4.0;
    if (bounces >= 2u) {
      let q = clamp(max(tp.x, max(tp.y, tp.z)), 0.05, 0.95);
      if (rnd() > q) { break; }
      tp /= q;
    }
  }
  return out;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  stackBase = li * STACK;
  if (gid.x >= F.size.x || gid.y >= F.size.y) { return; }
  let idx = gid.y * F.size.x + gid.x;
  if (F.mode == 0u) {
    rngState = pcg(idx ^ pcg(F.frame * 0x9E3779B9u + 17u));
    let pix = vec2f(gid.xy) + 0.5 + F.jitter;
    let r = tracePath(pix, true);
    textureStore(outRadiance, gid.xy, vec4f(r.radiance, 1.0));
    textureStore(outNormalDepth, gid.xy, vec4f(gNormal, gDepth));
    textureStore(outMotion, gid.xy, vec4f(gMotion, gObj, gPrevDepth));
    textureStore(outMaterial, gid.xy, vec4f(gAlbedo, gRough));
    textureStore(outEmission, gid.xy, vec4f(r.primaryEmission, 0.0));
  } else {
    var sum = vec3f(0.0);
    for (var s = 0u; s < F.spp; s++) {
      rngState = pcg(idx ^ pcg((F.counts.z + s) * 0x9E3779B9u + F.frame * 7919u + 1u));
      let pix = vec2f(gid.xy) + vec2f(rnd(), rnd());
      let r = tracePath(pix, false);
      sum += r.radiance + r.primaryEmission;
    }
    accum[idx] += vec4f(sum, f32(F.spp));
  }
}
`;
