// Small WebGPU helpers for the demo: device, timestamp timer, compose pass.

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
