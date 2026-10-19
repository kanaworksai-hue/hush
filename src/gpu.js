// Small WebGPU helpers: device, timestamp timer, texture readback, compose/display passes.

export async function initDevice() {
  if (!navigator.gpu) throw new Error('WebGPU not available');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const want = ['timestamp-query', 'float32-filterable', 'shader-f16'];
  const features = want.filter((f) => adapter.features.has(f));
  const L = adapter.limits;
  const device = await adapter.requestDevice({
    requiredFeatures: features,
    requiredLimits: {
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxBufferSize: L.maxBufferSize,
      maxStorageTexturesPerShaderStage: Math.min(L.maxStorageTexturesPerShaderStage, 8),
      maxTextureArrayLayers: L.maxTextureArrayLayers,
    },
  });
  device.lost.then((i) => console.error('device lost', i.message));
  device.addEventListener?.('uncapturederror', (e) => console.error('webgpu error', e.error.message));
  const info = adapter.info ?? {};
  return { adapter, device, features, info: `${info.vendor ?? ''} ${info.architecture ?? ''} ${info.description ?? ''}`.trim() };
}

// Per-pass GPU timer. Usage: t.begin(); encoder.beginComputePass(t.desc('x')); ... t.resolve(encoder); await t.read()
export class GpuTimer {
  constructor(device, max = 64) {
    this.device = device;
    this.enabled = device.features.has('timestamp-query');
    this.max = max;
    if (!this.enabled) return;
    this.qs = device.createQuerySet({ type: 'timestamp', count: max * 2 });
    this.res = device.createBuffer({ size: max * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.read_ = device.createBuffer({ size: max * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.labels = [];
    this.busy = false;
  }
  begin() { this.labels = []; }
  desc(label) {
    if (!this.enabled || this.busy || this.labels.length >= this.max) return undefined;
    const i = this.labels.length;
    this.labels.push(label);
    return { timestampWrites: { querySet: this.qs, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 } };
  }
  // a span over several passes: spanBegin(label) on the first pass, spanEnd() on the last (one entry, no per-pass overhead)
  spanBegin(label) {
    if (!this.enabled || this.busy || this.labels.length >= this.max) return undefined;
    this.spanIndex = this.labels.length;
    this.labels.push(label);
    return { timestampWrites: { querySet: this.qs, beginningOfPassWriteIndex: this.spanIndex * 2 } };
  }
  spanEnd() {
    if (this.spanIndex === undefined) return undefined;
    const i = this.spanIndex;
    this.spanIndex = undefined;
    return { timestampWrites: { querySet: this.qs, endOfPassWriteIndex: i * 2 + 1 } };
  }
  resolve(encoder) {
    if (!this.enabled || this.busy || !this.labels.length) return false;
    const n = this.labels.length * 2;
    encoder.resolveQuerySet(this.qs, 0, n, this.res, 0);
    encoder.copyBufferToBuffer(this.res, 0, this.read_, 0, n * 8);
    return true;
  }
  // returns { label: ms } summed per label, or null if not available
  async read() {
    if (!this.enabled || this.busy || !this.labels.length) return null;
    this.busy = true;
    const labels = this.labels.slice();
    await this.read_.mapAsync(GPUMapMode.READ);
    const t = new BigInt64Array(this.read_.getMappedRange().slice(0, labels.length * 16));
    this.read_.unmap();
    this.busy = false;
    const out = {};
    labels.forEach((l, i) => { out[l] = (out[l] ?? 0) + Number(t[i * 2 + 1] - t[i * 2]) / 1e6; });
    return out;
  }
}

const BPP = { rgba16float: 8, rgba32float: 16, rgba32uint: 16, rgba8unorm: 4, bgra8unorm: 4 };

// Reads a 2D texture back as raw bytes (row padding removed).
export async function readTexture(device, tex) {
  const bpp = BPP[tex.format];
  const w = tex.width, h = tex.height;
  const row = Math.ceil((w * bpp) / 256) * 256;
  const buf = device.createBuffer({ size: row * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: row }, [w, h]);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(buf.getMappedRange());
  const out = new Uint8Array(w * h * bpp);
  for (let y = 0; y < h; y++) out.set(src.subarray(y * row, y * row + w * bpp), y * w * bpp);
  buf.unmap();
  buf.destroy();
  return out;
}

export async function readBuffer(device, src, size) {
  const buf = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, buf, 0, size);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const out = new Uint8Array(buf.getMappedRange().slice(0));
  buf.unmap();
  buf.destroy();
  return out;
}

const COMPOSE = /* wgsl */ `
struct P { size: vec2u, mode: u32, _p: u32 };
@group(0) @binding(0) var<uniform> prm: P;
@group(0) @binding(1) var radiance: texture_2d<f32>;
@group(0) @binding(2) var emission: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> accum: array<vec4f>;
@group(0) @binding(4) var outNoisy: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var outRef: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= prm.size.x || gid.y >= prm.size.y) { return; }
  let p = vec2i(gid.xy);
  let e = textureLoad(emission, p, 0).rgb;
  textureStore(outNoisy, p, vec4f(textureLoad(radiance, p, 0).rgb + e, 1.0));
  let a = accum[gid.y * prm.size.x + gid.x];
  textureStore(outRef, p, vec4f(a.rgb / max(a.w, 1.0), 1.0));
}
`;

// noisy = radiance + primary emission; ref = accum / count
export class Composer {
  constructor(device, w, h) {
    this.device = device;
    this.w = w; this.h = h;
    const tex = () => device.createTexture({
      size: [w, h], format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.noisy = tex();
    this.ref = tex();
    this.ubo = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.ubo, 0, new Uint32Array([w, h, 0, 0]));
    this.pipe = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: COMPOSE }), entryPoint: 'main' } });
  }
  encode(encoder, g, accum) {
    const bg = this.device.createBindGroup({
      layout: this.pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ubo } },
        { binding: 1, resource: g.radiance.createView() },
        { binding: 2, resource: g.emission.createView() },
        { binding: 3, resource: { buffer: accum } },
        { binding: 4, resource: this.noisy.createView() },
        { binding: 5, resource: this.ref.createView() },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(this.w / 8), Math.ceil(this.h / 8));
    pass.end();
  }
}

const DISPLAY = /* wgsl */ `
struct P { split: f32, exposure: f32, w: f32, h: f32 };
@group(0) @binding(0) var<uniform> prm: P;
@group(0) @binding(1) var left: texture_2d<f32>;
@group(0) @binding(2) var right: texture_2d<f32>;
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VO;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}
// ACES fitted (Narkowicz 2015); Hush evaluates metrics after this same curve
fn aces(x: vec3f) -> vec3f { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0)); }
fn srgb(c: vec3f) -> vec3f { return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308)); }
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let px = vec2i(i.uv * vec2f(prm.w, prm.h));
  var c: vec3f;
  if (i.uv.x < prm.split) { c = textureLoad(left, px, 0).rgb; } else { c = textureLoad(right, px, 0).rgb; }
  var o = srgb(aces(max(c, vec3f(0.0)) * prm.exposure));
  if (abs(i.uv.x - prm.split) * prm.w < 1.0) { o = vec3f(1.0); }
  return vec4f(o, 1.0);
}
`;

export class Display {
  constructor(device, context, format) {
    this.device = device;
    this.context = context;
    this.ubo = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ code: DISPLAY });
    this.pipe = device.createRenderPipeline({
      layout: 'auto', vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    });
  }
  encode(encoder, left, right, split, exposure) {
    this.device.queue.writeBuffer(this.ubo, 0, new Float32Array([split, exposure, left.width, left.height]));
    const bg = this.device.createBindGroup({
      layout: this.pipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.ubo } }, { binding: 1, resource: left.createView() }, { binding: 2, resource: right.createView() }],
    });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
  }
}

// POST to a dev-server save endpoint (when the server has one); retried, a long capture should not die on one dropped request
export async function save(path, bytes, tries = 5) {
  for (let i = 1; ; i++) {
    try {
      const r = await fetch(`/api/save?path=${encodeURIComponent(path)}`, { method: 'POST', body: bytes });
      if (r.ok) return;
      if (i >= tries) throw new Error(`save failed ${path}: ${r.status}`);
    } catch (e) {
      if (i >= tries) throw e;
    }
    await new Promise((res) => setTimeout(res, 500 * i));
  }
}
