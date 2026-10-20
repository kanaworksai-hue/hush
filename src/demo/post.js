// Display post for the demo: split-screen composite of two linear HDR images, bloom, vignette, AgX tone mapping,
// sRGB. Display only: the denoisers never see it.
//
//   const post = new Post(device, format);
//   post.encode(encoder, { left, right, split, exposure, target: canvasView, width: canvas.width, height: canvas.height });
//   three panels: add mid + split2 -> left | mid | right with dividers at split and split2; same: true with
//   thirds -> each panel shows the middle third of its own image (the same region side by side)

const FULLSCREEN = /* wgsl */ `
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VO;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}
`;

// left | mid | right at split / split2, times exposure (render resolution)
const COMPOSITE = FULLSCREEN + /* wgsl */ `
struct P { split: f32, exposure: f32, split2: f32, same: f32 };
@group(0) @binding(0) var<uniform> prm: P;
@group(0) @binding(1) var left: texture_2d<f32>;
@group(0) @binding(2) var right: texture_2d<f32>;
@group(0) @binding(3) var mid: texture_2d<f32>;
@fragment fn fs(i: VO) -> @location(0) vec4f {
  var p = vec2i(i.pos.xy);
  // same = 1: every third shows the middle third of its image (side-by-side comparison of the same region)
  if (prm.same > 0.5) {
    let w = i32(textureDimensions(left).x);
    p.x = p.x - (min(p.x * 3 / w, 2) - 1) * (w / 3);
  }
  var c = textureLoad(right, p, 0).rgb;
  if (i.uv.x < prm.split2) { c = textureLoad(mid, p, 0).rgb; }
  if (i.uv.x < prm.split) { c = textureLoad(left, p, 0).rgb; }
  return vec4f(max(c, vec3f(0.0)) * prm.exposure, 1.0);
}
`;

// 13-tap downsample (Jimenez 2014); the first level uses a Karis average so single bright pixels (1 spp fireflies)
// do not bloom into blobs
const DOWN = FULLSCREEN + /* wgsl */ `
struct P { texel: vec2f, karis: f32, _a: f32 };
@group(0) @binding(0) var<uniform> prm: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
fn s(uv: vec2f, o: vec2f) -> vec3f { return textureSampleLevel(src, samp, uv + o * prm.texel, 0.0).rgb; }
fn kw(c: vec3f) -> f32 { return 1.0 / (1.0 + dot(c, vec3f(0.2126, 0.7152, 0.0722))); }
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let uv = i.uv;
  let a = s(uv, vec2f(-2.0, -2.0)); let b = s(uv, vec2f(0.0, -2.0)); let c = s(uv, vec2f(2.0, -2.0));
  let d = s(uv, vec2f(-2.0, 0.0)); let e = s(uv, vec2f(0.0, 0.0)); let f = s(uv, vec2f(2.0, 0.0));
  let g = s(uv, vec2f(-2.0, 2.0)); let h = s(uv, vec2f(0.0, 2.0)); let k = s(uv, vec2f(2.0, 2.0));
  let j = s(uv, vec2f(-1.0, -1.0)); let l = s(uv, vec2f(1.0, -1.0)); let m = s(uv, vec2f(-1.0, 1.0)); let n = s(uv, vec2f(1.0, 1.0));
  if (prm.karis > 0.5) {
    let g0 = (a + b + d + e) * 0.25; let g1 = (b + c + e + f) * 0.25; let g2 = (d + e + g + h) * 0.25;
    let g3 = (e + f + h + k) * 0.25; let g4 = (j + l + m + n) * 0.25;
    let w0 = kw(g0) * 0.125; let w1 = kw(g1) * 0.125; let w2 = kw(g2) * 0.125; let w3 = kw(g3) * 0.125; let w4 = kw(g4) * 0.5;
    return vec4f((g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4), 1.0);
  }
  let o = e * 0.125 + (a + c + g + k) * 0.03125 + (b + d + f + h) * 0.0625 + (j + l + m + n) * 0.125;
  return vec4f(o, 1.0);
}
`;

// 3x3 tent upsample, added onto the next larger level (additive blend)
const UP = FULLSCREEN + /* wgsl */ `
struct P { texel: vec2f, _a: f32, _b: f32 };
@group(0) @binding(0) var<uniform> prm: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
fn s(uv: vec2f, o: vec2f) -> vec3f { return textureSampleLevel(src, samp, uv + o * prm.texel, 0.0).rgb; }
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let uv = i.uv;
  var o = s(uv, vec2f(0.0, 0.0)) * 4.0;
  o += (s(uv, vec2f(-1.0, 0.0)) + s(uv, vec2f(1.0, 0.0)) + s(uv, vec2f(0.0, -1.0)) + s(uv, vec2f(0.0, 1.0))) * 2.0;
  o += s(uv, vec2f(-1.0, -1.0)) + s(uv, vec2f(1.0, -1.0)) + s(uv, vec2f(-1.0, 1.0)) + s(uv, vec2f(1.0, 1.0));
  return vec4f(o / 16.0, 1.0);
}
`;

// bloom mix, vignette, AgX (Blender / Filament, via three.js) with a mild "punchy" look, sRGB, divider line
const FINAL = FULLSCREEN + /* wgsl */ `
struct P { split: f32, width: f32, bloom: f32, vignette: f32, texel: vec2f, line: f32, split2: f32 };
@group(0) @binding(0) var<uniform> prm: P;
@group(0) @binding(1) var hdr: texture_2d<f32>;
@group(0) @binding(2) var bloomTex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
fn agxContrast(x: vec3f) -> vec3f {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
fn agx(c0: vec3f) -> vec3f {
  let toRec2020 = mat3x3f(vec3f(0.6274, 0.0691, 0.0164), vec3f(0.3293, 0.9195, 0.0880), vec3f(0.0433, 0.0113, 0.8956));
  let toSrgb = mat3x3f(vec3f(1.6605, -0.1246, -0.0182), vec3f(-0.5876, 1.1329, -0.1006), vec3f(-0.0728, -0.0083, 1.1187));
  let inset = mat3x3f(vec3f(0.856627153315983, 0.137318972929847, 0.11189821299995),
    vec3f(0.0951212405381588, 0.761241990602591, 0.0767994186031903), vec3f(0.0482516061458583, 0.101439036467562, 0.811302368396859));
  let outset = mat3x3f(vec3f(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
    vec3f(-0.11060664309660323, 1.157823702216272, -0.11060664309660294), vec3f(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405));
  let minEv = -12.47393;
  let maxEv = 4.026069;
  var c = inset * (toRec2020 * c0);
  c = clamp((log2(max(c, vec3f(1e-10))) - minEv) / (maxEv - minEv), vec3f(0.0), vec3f(1.0));
  c = agxContrast(c);
  // look: a little more contrast and saturation than base AgX
  let l = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  c = pow(max(c, vec3f(0.0)), vec3f(1.22));
  c = l + (c - l) * 1.15;
  c = outset * c;
  c = pow(max(vec3f(0.0), c), vec3f(2.2));
  return clamp(toSrgb * c, vec3f(0.0), vec3f(1.0));
}
fn srgb(c: vec3f) -> vec3f { return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308)); }
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let uv = i.uv;
  var c = textureSampleLevel(hdr, samp, uv, 0.0).rgb;
  let t = prm.texel;
  var b = textureSampleLevel(bloomTex, samp, uv, 0.0).rgb * 4.0;
  b += (textureSampleLevel(bloomTex, samp, uv + vec2f(-t.x, 0.0), 0.0).rgb + textureSampleLevel(bloomTex, samp, uv + vec2f(t.x, 0.0), 0.0).rgb
    + textureSampleLevel(bloomTex, samp, uv + vec2f(0.0, -t.y), 0.0).rgb + textureSampleLevel(bloomTex, samp, uv + vec2f(0.0, t.y), 0.0).rgb) * 2.0;
  b += textureSampleLevel(bloomTex, samp, uv - t, 0.0).rgb + textureSampleLevel(bloomTex, samp, uv + t, 0.0).rgb
    + textureSampleLevel(bloomTex, samp, uv + vec2f(t.x, -t.y), 0.0).rgb + textureSampleLevel(bloomTex, samp, uv + vec2f(-t.x, t.y), 0.0).rgb;
  c = mix(c, b / 16.0, prm.bloom);
  // vignette: cos^4-like falloff towards the corners
  let q = (uv - 0.5) * vec2f(1.0, 0.75);
  c *= mix(1.0, 1.0 - dot(q, q) * 1.6, prm.vignette);
  var o = srgb(agx(c));
  if (prm.line > 0.0 && min(abs(uv.x - prm.split), abs(uv.x - prm.split2)) * prm.width < prm.line) { o = mix(o, vec3f(1.0), 0.85); }
  return vec4f(o, 1.0);
}
`;

const LEVELS = 6;

export class Post {
  constructor(device, format, { bloom = 0.03, vignette = 0.35 } = {}) {
    this.device = device;
    this.bloom = bloom;
    this.vignette = vignette;
    const pipe = (code, fmt, blend) => {
      const module = device.createShaderModule({ code });
      return device.createRenderPipeline({
        layout: 'auto', vertex: { module, entryPoint: 'vs' },
        fragment: { module, entryPoint: 'fs', targets: [blend ? { format: fmt, blend } : { format: fmt }] },
      });
    };
    const add = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } };
    this.pComposite = pipe(COMPOSITE, 'rgba16float');
    this.pDown = pipe(DOWN, 'rgba16float');
    this.pUp = pipe(UP, 'rgba16float', add);
    this.pFinal = pipe(FINAL, format);
    this.samp = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.ubos = Array.from({ length: 2 + 2 * LEVELS }, () => device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  }

  resize(w, h) {
    if (w === this.w && h === this.h) return;
    this.destroyTargets();
    this.w = w; this.h = h;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT;
    this.hdr = this.device.createTexture({ size: [w, h], format: 'rgba16float', usage });
    this.levels = [];
    for (let i = 1; i <= LEVELS; i++) {
      const lw = Math.max(1, w >> i), lh = Math.max(1, h >> i);
      this.levels.push(this.device.createTexture({ size: [lw, lh], format: 'rgba16float', usage }));
    }
  }

  destroyTargets() {
    this.hdr?.destroy();
    for (const t of this.levels ?? []) t.destroy();
  }

  destroy() {
    this.destroyTargets();
    for (const b of this.ubos) b.destroy();
  }

  draw(enc, pipeline, view, entries, load = false) {
    const bg = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: entries.map((resource, binding) => ({ binding, resource })) });
    const pass = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: load ? 'load' : 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
  }

  // left / right: linear HDR textures of the same size; target: view of the output (canvas) texture
  encode(enc, { left, right, mid = right, split = 0.5, split2 = split, same = false, exposure = 1, target, width, height, line = 1.5 }) {
    this.resize(left.width, left.height);
    const q = this.device.queue;
    const ubo = (i, data) => { q.writeBuffer(this.ubos[i], 0, new Float32Array(data)); return { buffer: this.ubos[i] }; };
    this.draw(enc, this.pComposite, this.hdr.createView(), [ubo(0, [split, exposure, split2, same ? 1 : 0]), left.createView(), right.createView(), mid.createView()]);
    let src = this.hdr;
    this.levels.forEach((t, i) => {
      this.draw(enc, this.pDown, t.createView(), [ubo(2 + i, [1 / src.width, 1 / src.height, i === 0 ? 1 : 0, 0]), src.createView(), this.samp]);
      src = t;
    });
    for (let i = LEVELS - 1; i > 0; i--) {
      const s = this.levels[i];
      this.draw(enc, this.pUp, this.levels[i - 1].createView(), [ubo(2 + LEVELS + i, [1 / s.width, 1 / s.height, 0, 0]), s.createView(), this.samp], true);
    }
    const b = this.levels[0];
    this.draw(enc, this.pFinal, target, [
      ubo(1, [split, width, this.bloom, this.vignette, 1 / b.width, 1 / b.height, line, split2]), this.hdr.createView(), b.createView(), this.samp,
    ]);
    void height;
  }
}
