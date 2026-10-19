// SVGF baseline (Schied et al. 2017) in WGSL compute, plus the TAA resolve the paper ends with.
//
// Inputs per frame (from the path tracer or any renderer producing the same G-buffer):
//   radiance     rgba16float  noisy radiance without primary emission
//   normalDepth  rgba16float  world normal xyz, linear view depth (w < 0 = background)
//   motion       rgba16float  xy = previous pixel - current pixel, z = object id, w = previous linear depth
//   material     rgba16float  albedo rgb, roughness
//   emission     rgba16float  primary emission rgb
// plus last frame's normalDepth / motion (the caller ping-pongs them).
// Output: `this.output` rgba16float, linear HDR, antialiased.

const COMMON = /* wgsl */ `
struct P { size: vec2u, iter: u32, step: u32, sigmaL: f32, sigmaN: f32, sigmaZ: f32, alphaColor: f32,
           alphaMoments: f32, taaAlpha: f32, flags: u32, _p: u32 };
@group(0) @binding(0) var<uniform> prm: P;
fn lum(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
fn inside(p: vec2i) -> bool { return p.x >= 0 && p.y >= 0 && p.x < i32(prm.size.x) && p.y < i32(prm.size.y); }
`;

const TEMPORAL = COMMON + /* wgsl */ `
@group(0) @binding(1) var radiance: texture_2d<f32>;
@group(0) @binding(2) var normalDepth: texture_2d<f32>;
@group(0) @binding(3) var motion: texture_2d<f32>;
@group(0) @binding(4) var material: texture_2d<f32>;
@group(0) @binding(5) var prevNormalDepth: texture_2d<f32>;
@group(0) @binding(6) var prevMotion: texture_2d<f32>;
@group(0) @binding(7) var prevColor: texture_2d<f32>;
@group(0) @binding(8) var prevMoments: texture_2d<f32>;
@group(0) @binding(9) var outColor: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var outMoments: texture_storage_2d<rgba16float, write>;

fn consistent(q: vec2i, n: vec3f, zPrevExpected: f32, obj: f32) -> bool {
  if (!inside(q)) { return false; }
  let nd = textureLoad(prevNormalDepth, q, 0);
  if (nd.w < 0.0) { return false; }
  if (textureLoad(prevMotion, q, 0).z != obj) { return false; }
  if (abs(nd.w - zPrevExpected) > 0.05 * zPrevExpected + 0.01) { return false; }
  if (dot(nd.xyz, n) < 0.9) { return false; }
  return true;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= prm.size.x || gid.y >= prm.size.y) { return; }
  let p = vec2i(gid.xy);
  let nd = textureLoad(normalDepth, p, 0);
  if (nd.w < 0.0) {
    textureStore(outColor, p, vec4f(0.0));
    textureStore(outMoments, p, vec4f(0.0));
    return;
  }
  let mv = textureLoad(motion, p, 0);
  let alb = textureLoad(material, p, 0).rgb;
  let illum = textureLoad(radiance, p, 0).rgb / max(alb, vec3f(1e-3));
  let l = lum(illum);
  // bilinear reprojection with per-tap consistency
  let pp = vec2f(p) + 0.5 + mv.xy - 0.5;
  let b = vec2i(floor(pp));
  let f = pp - floor(pp);
  var sumW = 0.0;
  var hc = vec3f(0.0);
  var hm = vec3f(0.0);
  let wts = array<f32, 4>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
  let offs = array<vec2i, 4>(vec2i(0, 0), vec2i(1, 0), vec2i(0, 1), vec2i(1, 1));
  let noHistory = (prm.flags & 2u) != 0u;
  for (var i = 0; i < 4; i++) {
    if (noHistory) { break; }
    let q = b + offs[i];
    if (consistent(q, nd.xyz, mv.w, mv.z) && wts[i] > 0.0) {
      sumW += wts[i];
      hc += wts[i] * textureLoad(prevColor, q, 0).rgb;
      hm += wts[i] * textureLoad(prevMoments, q, 0).xyz;
    }
  }
  if (sumW < 0.01 && !noHistory) {
    // wider 3x3 search around the reprojected position
    let c = vec2i(round(pp));
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let q = c + vec2i(dx, dy);
        if (consistent(q, nd.xyz, mv.w, mv.z)) {
          sumW += 1.0;
          hc += textureLoad(prevColor, q, 0).rgb;
          hm += textureLoad(prevMoments, q, 0).xyz;
        }
      }
    }
  }
  var len = 0.0;
  var color = illum;
  var mom = vec2f(l, l * l);
  if (sumW >= 0.01) {
    hc /= sumW;
    hm /= sumW;
    len = min(hm.z + 1.0, 64.0);
    let ac = max(prm.alphaColor, 1.0 / len);
    let am = max(prm.alphaMoments, 1.0 / len);
    color = mix(hc, illum, ac);
    mom = mix(hm.xy, mom, am);
  } else {
    len = 1.0;
  }
  textureStore(outColor, p, vec4f(color, 0.0));
  textureStore(outMoments, p, vec4f(mom, len, 0.0));
}
`;

const VARIANCE = COMMON + /* wgsl */ `
@group(0) @binding(1) var inColor: texture_2d<f32>;
@group(0) @binding(2) var inMoments: texture_2d<f32>;
@group(0) @binding(3) var normalDepth: texture_2d<f32>;
@group(0) @binding(4) var outColor: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= prm.size.x || gid.y >= prm.size.y) { return; }
  let p = vec2i(gid.xy);
  let nd = textureLoad(normalDepth, p, 0);
  if (nd.w < 0.0) { textureStore(outColor, p, vec4f(0.0)); return; }
  let c = textureLoad(inColor, p, 0).rgb;
  let m = textureLoad(inMoments, p, 0).xyz;
  if (m.z >= 4.0) {
    textureStore(outColor, p, vec4f(c, max(0.0, m.y - m.x * m.x)));
    return;
  }
  // short history: estimate variance spatially (7x7, depth + normal edge-stopping)
  var sw = 0.0;
  var sc = vec3f(0.0);
  var sm = vec2f(0.0);
  for (var dy = -3; dy <= 3; dy++) {
    for (var dx = -3; dx <= 3; dx++) {
      let q = p + vec2i(dx, dy);
      if (!inside(q)) { continue; }
      let ndq = textureLoad(normalDepth, q, 0);
      if (ndq.w < 0.0) { continue; }
      let wz = exp(-abs(ndq.w - nd.w) / (0.02 * nd.w * length(vec2f(f32(dx), f32(dy))) + 1e-3));
      let wn = pow(max(0.0, dot(ndq.xyz, nd.xyz)), prm.sigmaN);
      let w = wz * wn;
      sw += w;
      sc += w * textureLoad(inColor, q, 0).rgb;
      sm += w * textureLoad(inMoments, q, 0).xy;
    }
  }
  sc /= max(sw, 1e-6);
  sm /= max(sw, 1e-6);
  let v = max(0.0, sm.y - sm.x * sm.x) * 4.0 / max(m.z, 1.0);
  textureStore(outColor, p, vec4f(sc, v));
}
`;

const ATROUS = COMMON + /* wgsl */ `
@group(0) @binding(1) var inColor: texture_2d<f32>;
@group(0) @binding(2) var normalDepth: texture_2d<f32>;
@group(0) @binding(3) var outColor: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var outFeedback: texture_storage_2d<rgba16float, write>;

fn zGrad(p: vec2i, z: f32) -> f32 {
  let zl = textureLoad(normalDepth, clamp(p - vec2i(1, 0), vec2i(0), vec2i(prm.size) - 1), 0).w;
  let zr = textureLoad(normalDepth, clamp(p + vec2i(1, 0), vec2i(0), vec2i(prm.size) - 1), 0).w;
  let zu = textureLoad(normalDepth, clamp(p - vec2i(0, 1), vec2i(0), vec2i(prm.size) - 1), 0).w;
  let zd = textureLoad(normalDepth, clamp(p + vec2i(0, 1), vec2i(0), vec2i(prm.size) - 1), 0).w;
  let gx = min(select(1e9, abs(zr - z), zr > 0.0), select(1e9, abs(zl - z), zl > 0.0));
  let gy = min(select(1e9, abs(zd - z), zd > 0.0), select(1e9, abs(zu - z), zu > 0.0));
  return min(max(gx, gy), z);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= prm.size.x || gid.y >= prm.size.y) { return; }
  let p = vec2i(gid.xy);
  let nd = textureLoad(normalDepth, p, 0);
  if (nd.w < 0.0) {
    textureStore(outColor, p, vec4f(0.0));
    if (prm.iter == 0u) { textureStore(outFeedback, p, vec4f(0.0)); }
    return;
  }
  let center = textureLoad(inColor, p, 0);
  // 3x3 gaussian of the variance for the luminance edge-stopping function
  var gv = 0.0;
  let k3 = array<f32, 3>(0.25, 0.5, 0.25);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let q = clamp(p + vec2i(dx, dy), vec2i(0), vec2i(prm.size) - 1);
      gv += k3[dx + 1] * k3[dy + 1] * textureLoad(inColor, q, 0).a;
    }
  }
  let lp = lum(center.rgb);
  let denomL = prm.sigmaL * sqrt(max(gv, 0.0)) + 1e-6;
  let zg = zGrad(p, nd.w);
  let h = array<f32, 5>(1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0);
  var sw = h[2] * h[2];
  var sc = center.rgb * sw;
  var sv = center.a * sw * sw;
  let step = i32(prm.step);
  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let q = p + vec2i(dx, dy) * step;
      if (!inside(q)) { continue; }
      let ndq = textureLoad(normalDepth, q, 0);
      if (ndq.w < 0.0) { continue; }
      let cq = textureLoad(inColor, q, 0);
      let dist = length(vec2f(f32(dx), f32(dy))) * f32(step);
      let wz = abs(ndq.w - nd.w) / (prm.sigmaZ * zg * dist + 1e-3);
      let wn = pow(max(0.0, dot(ndq.xyz, nd.xyz)), prm.sigmaN);
      let wl = abs(lum(cq.rgb) - lp) / denomL;
      let w = h[dx + 2] * h[dy + 2] * wn * exp(-wz - wl);
      sw += w;
      sc += w * cq.rgb;
      sv += w * w * cq.a;
    }
  }
  let o = vec4f(sc / sw, sv / (sw * sw));
  textureStore(outColor, p, o);
  if (prm.iter == 0u) { textureStore(outFeedback, p, vec4f(o.rgb, 0.0)); }
}
`;

const RESOLVE = COMMON + /* wgsl */ `
@group(0) @binding(1) var illum: texture_2d<f32>;
@group(0) @binding(2) var material: texture_2d<f32>;
@group(0) @binding(3) var motion: texture_2d<f32>;
@group(0) @binding(4) var prevTaa: texture_2d<f32>;
@group(0) @binding(5) var outTaa: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var samp: sampler;
@group(0) @binding(7) var emission: texture_2d<f32>;

fn remod(q: vec2i) -> vec3f {
  let qq = clamp(q, vec2i(0), vec2i(prm.size) - 1);
  return textureLoad(illum, qq, 0).rgb * textureLoad(material, qq, 0).rgb + textureLoad(emission, qq, 0).rgb;
}
fn toYCoCg(c: vec3f) -> vec3f {
  return vec3f(c.r * 0.25 + c.g * 0.5 + c.b * 0.25, c.r * 0.5 - c.b * 0.5, -c.r * 0.25 + c.g * 0.5 - c.b * 0.25);
}
fn fromYCoCg(c: vec3f) -> vec3f { return vec3f(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }
// compress HDR before clipping / blending (Karis)
fn tm(c: vec3f) -> vec3f { return c / (1.0 + lum(c)); }
fn itm(c: vec3f) -> vec3f { return c / max(1.0 - lum(c), 1e-4); }

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= prm.size.x || gid.y >= prm.size.y) { return; }
  let p = vec2i(gid.xy);
  let cur = tm(remod(p));
  if ((prm.flags & 1u) == 0u) { textureStore(outTaa, p, vec4f(itm(cur), 1.0)); return; }
  var m1 = vec3f(0.0);
  var m2 = vec3f(0.0);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let c = toYCoCg(tm(remod(p + vec2i(dx, dy))));
      m1 += c; m2 += c * c;
    }
  }
  m1 /= 9.0; m2 /= 9.0;
  let sd = sqrt(max(m2 - m1 * m1, vec3f(0.0)));
  let mv = textureLoad(motion, p, 0).xy;
  let uv = (vec2f(p) + 0.5 + mv) / vec2f(prm.size);
  var a = prm.taaAlpha;
  var res = cur;
  if (all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0)) && (prm.flags & 2u) == 0u) {
    let hist = textureSampleLevel(prevTaa, samp, uv, 0.0).rgb;
    var h = toYCoCg(tm(hist));
    // variance clipping (Salvi), gamma 1.25
    let lo = m1 - 1.25 * sd;
    let hi = m1 + 1.25 * sd;
    h = clamp(h, lo, hi);
    res = mix(fromYCoCg(h), cur, a);
  }
  textureStore(outTaa, p, vec4f(itm(res), 1.0));
}
`;

export class SVGF {
  constructor(device, width, height, opts = {}) {
    this.device = device;
    this.width = width;
    this.height = height;
    this.iterations = opts.iterations ?? 5;
    this.params = {
      sigmaL: 4, sigmaN: 128, sigmaZ: 1, alphaColor: 0.2, alphaMoments: 0.2, taaAlpha: 0.1, taa: true, ...opts,
    };
    this.textures = [];
    this.width = this.height = 0;
    this.resize(width, height);
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    const pipe = (code) => device.createComputePipeline({
      layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });
    this.pTemporal = pipe(TEMPORAL);
    this.pVariance = pipe(VARIANCE);
    this.pAtrous = pipe(ATROUS);
    this.pResolve = pipe(RESOLVE);
    // one uniform buffer per dispatch (iteration index / step differ)
    this.ubos = Array.from({ length: 8 }, () => device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  }

  // reallocates the textures for a new size and resets the history
  resize(width, height) {
    if (width === this.width && height === this.height) return;
    for (const t of this.textures) t.destroy();
    this.width = width;
    this.height = height;
    this.frame = 0;
    this.resetHistory = true;
    const tex = (format) => {
      const t = this.device.createTexture({
        size: [width, height], format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
      });
      this.textures.push(t);
      return t;
    };
    this.textures = [];
    this.histColor = [tex('rgba16float'), tex('rgba16float')];
    this.histMoments = [tex('rgba16float'), tex('rgba16float')];
    this.tmpColor = tex('rgba16float');
    this.atrous = [tex('rgba16float'), tex('rgba16float')];
    this.taa = [tex('rgba16float'), tex('rgba16float')];
  }

  destroy() {
    for (const t of this.textures) t.destroy();
    for (const b of this.ubos) b.destroy();
    this.textures = [];
  }

  get output() { return this.taa[this.frame & 1]; }

  reset() { this.resetHistory = true; }

  writeParams(i, iter, step) {
    const b = new ArrayBuffer(48);
    const u = new Uint32Array(b), f = new Float32Array(b);
    const P = this.params;
    u[0] = this.width; u[1] = this.height; u[2] = iter; u[3] = step;
    f[4] = P.sigmaL; f[5] = P.sigmaN; f[6] = P.sigmaZ; f[7] = P.alphaColor; f[8] = P.alphaMoments; f[9] = P.taaAlpha;
    u[10] = (P.taa ? 1 : 0) | (this.resetHistory ? 2 : 0);
    this.device.queue.writeBuffer(this.ubos[i], 0, b);
  }

  // g = { radiance, normalDepth, motion, material, emission, prevNormalDepth, prevMotion } (GPUTextures)
  // timer: optional { pass(encoder, label) -> GPUComputePassDescriptor extras }
  encode(encoder, g, timer) {
    const d = this.device;
    const cur = this.frame & 1, prev = cur ^ 1;
    const wg = [Math.ceil(this.width / 8), Math.ceil(this.height / 8)];
    const v = (t) => t.createView();
    const run = (pipeline, entries, ubo, label) => {
      const bg = d.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: ubo } }, ...entries.map((r, i) => ({ binding: i + 1, resource: r }))],
      });
      const pass = encoder.beginComputePass(timer ? timer.desc(label) : undefined);
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wg[0], wg[1]);
      pass.end();
    };
    this.writeParams(0, 0, 0);
    run(this.pTemporal, [v(g.radiance), v(g.normalDepth), v(g.motion), v(g.material),
      v(g.prevNormalDepth), v(g.prevMotion),
      v(this.histColor[prev]), v(this.histMoments[prev]), v(this.tmpColor), v(this.histMoments[cur])], this.ubos[0], 'temporal');
    this.writeParams(1, 0, 0);
    run(this.pVariance, [v(this.tmpColor), v(this.histMoments[cur]), v(g.normalDepth), v(this.atrous[0])], this.ubos[1], 'variance');
    let src = 0;
    for (let i = 0; i < this.iterations; i++) {
      this.writeParams(2 + i, i, 1 << i);
      run(this.pAtrous, [v(this.atrous[src]), v(g.normalDepth), v(this.atrous[src ^ 1]), v(this.histColor[cur])],
        this.ubos[2 + i], 'atrous');
      src ^= 1;
    }
    this.writeParams(7, 0, 0);
    run(this.pResolve, [v(this.atrous[src]), v(g.material), v(g.motion), v(this.taa[prev]), v(this.taa[cur]), this.sampler, v(g.emission)],
      this.ubos[7], 'resolve');
    this.resetHistory = false;
    this.frame++;
    return this.taa[cur];
  }
}
