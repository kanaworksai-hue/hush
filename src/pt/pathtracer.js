import ptCode from './pathtracer.wgsl.js';

const MAX_LIGHTS = 8;
const UBO_SIZE = 960;

// Halton(2,3) subpixel jitter, 16 phases, in pixels [-0.5, 0.5]
function halton(i, b) {
  let f = 1, r = 0;
  while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); }
  return r;
}
export const JITTER = Array.from({ length: 16 }, (_, i) => [halton(i + 1, 2) - 0.5, halton(i + 1, 3) - 0.5]);

export class PathTracer {
  constructor(device, sceneGPU, width, height) {
    this.device = device;
    this.scene = sceneGPU;
    this.ubo = device.createBuffer({ size: UBO_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.pipeline = device.createComputePipeline({
      layout: 'auto', compute: { module: device.createShaderModule({ code: ptCode }), entryPoint: 'main' },
    });
    this.envSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'clamp-to-edge' });
    this.setEnvMap(null);
    this.resize(width, height);
  }

  // env.map (envmap.js prepareEnvMap) -> GPU; null binds a 1x1 placeholder (analytic sky)
  setEnvMap(map) {
    if (map === this.envMap && this.envTex) return;
    this.envTex?.destroy();
    this.envBuf?.destroy();
    const d = this.device;
    const w = map?.width ?? 1, h = map?.height ?? 1;
    this.envTex = d.createTexture({ size: [w, h], format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    if (map) d.queue.writeTexture({ texture: this.envTex }, map.half, { bytesPerRow: w * 8 }, [w, h]);
    const table = map?.table ?? new Float32Array(4);
    this.envBuf = d.createBuffer({ size: table.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.envBuf, 0, table);
    this.envMap = map;
    this.bind = null;
  }

  resize(width, height) {
    if (width === this.width && height === this.height) return;
    this.destroyTargets();
    this.width = width;
    this.height = height;
    const d = this.device;
    const tex = (format) => d.createTexture({
      size: [width, height], format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    // ping-pong G-buffers so the denoiser can read last frame's
    this.gbuf = [0, 1].map(() => ({
      radiance: tex('rgba16float'), normalDepth: tex('rgba16float'), motion: tex('rgba16float'), material: tex('rgba16float'),
      emission: tex('rgba16float'),
    }));
    this.accum = d.createBuffer({ size: width * height * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.bind = null;
  }

  destroyTargets() {
    for (const g of this.gbuf ?? []) for (const t of Object.values(g)) t.destroy();
    this.accum?.destroy();
  }

  destroy() {
    this.destroyTargets();
    this.ubo.destroy();
    this.envTex.destroy();
    this.envBuf.destroy();
  }

  bindGroup(i) {
    const s = this.scene, g = this.gbuf[i];
    return this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ubo } },
        { binding: 1, resource: { buffer: s.bufNodes } },
        { binding: 2, resource: { buffer: s.bufTris } },
        { binding: 3, resource: { buffer: s.bufTriIdx } },
        { binding: 4, resource: { buffer: s.bufVerts } },
        { binding: 5, resource: { buffer: s.bufMats } },
        { binding: 6, resource: { buffer: this.accum } },
        { binding: 7, resource: s.texArray.createView({ dimension: '2d-array' }) },
        { binding: 8, resource: s.sampler },
        { binding: 9, resource: g.radiance.createView() },
        { binding: 10, resource: g.normalDepth.createView() },
        { binding: 11, resource: g.motion.createView() },
        { binding: 12, resource: g.material.createView() },
        { binding: 13, resource: g.emission.createView() },
        { binding: 14, resource: this.envTex.createView() },
        { binding: 15, resource: { buffer: this.envBuf } },
        { binding: 16, resource: this.envSampler },
      ],
    });
  }

  // cam: { camera: THREE.PerspectiveCamera (matrices updated), prevViewProj: Matrix4 }
  // env: { skyZenith, skyHorizon, skyIntensity, ground, lights: [{kind, pos|dir, prevPos, radius, radiance, cosMax}],
  //        map?, mapIntensity = 1, mapRotation = 0 (radians about +y), clampLum? }   map (envmap.js) replaces the analytic sky;
  //        env.clampLum overrides the clampLum argument (per-scene firefly clamp on indirect light), env.clampDirect
  //        clamps direct light at the primary hit (0 / unset = off)
  writeUniforms({ camera, prevViewProj, frame, mode, jitter, spp = 1, sampleOffset = 0, maxBounces = 3, clampLum = 16, shadowTau = 0, env }) {
    const b = new ArrayBuffer(UBO_SIZE);
    const f = new Float32Array(b), u = new Uint32Array(b);
    const e = camera.matrixWorld.elements;
    const th = Math.tan((camera.fov * Math.PI) / 360);
    const right = [e[0], e[1], e[2]], up = [e[4], e[5], e[6]], fwd = [-e[8], -e[9], -e[10]];
    f.set([e[12], e[13], e[14], 0], 0);
    f.set([...right.map((x) => x * th * camera.aspect), 0], 4);
    f.set([...up.map((x) => x * th), 0], 8);
    f.set([...fwd, 0], 12);
    const vp = camera.projectionMatrix.clone().multiply(camera.matrixWorldInverse);
    f.set(vp.elements, 16);
    f.set((prevViewProj ?? vp).elements, 32);
    u[48] = this.width; u[49] = this.height; u[50] = frame; u[51] = mode;
    f[52] = jitter[0]; f[53] = jitter[1]; u[54] = spp; u[55] = maxBounces;
    f.set([...env.skyZenith, 0], 56);
    f.set([...env.skyHorizon, env.skyIntensity ?? 1], 60);
    f.set([...env.ground, 0], 64);
    const blas = this.scene.blasTable();
    const lights = env.lights.slice(0, MAX_LIGHTS);
    u[68] = blas.length; u[69] = lights.length; u[70] = sampleOffset;
    f[72] = env.clampLum ?? clampLum;
    f[73] = (2 * th) / this.height;
    f[74] = shadowTau;
    f[75] = env.clampDirect ?? 0;
    this.setEnvMap(env.map ?? null);
    const rot = env.mapRotation ?? 0;
    f.set([env.mapIntensity ?? 1, Math.cos(rot), Math.sin(rot), env.map ? 1 : 0], 76);
    blas.forEach((bl, i) => { u.set([bl.nodeOffset, bl.triOffset, bl.nodeCount, bl.dynamic], 80 + i * 4); });
    lights.forEach((L, i) => {
      const o = 112 + i * 16;
      if (L.kind === 0) {
        f.set([...L.dir, 0], o);
        f.set([...L.dir, 0], o + 4);
        f.set([...L.radiance, 0], o + 8);
        f.set([0, L.cosMax, 0, 0], o + 12);
      } else {
        f.set([...L.pos, L.radius], o);
        f.set([...(L.prevPos ?? L.pos), L.radius], o + 4);
        f.set([...L.radiance, 0], o + 8);
        f.set([1, 0, 0, 0], o + 12);
      }
    });
    this.device.queue.writeBuffer(this.ubo, 0, b);
    return vp;
  }

  encode(encoder, slot, timer, label = 'pathtrace') {
    this.bind ??= [this.bindGroup(0), this.bindGroup(1)];
    const pass = encoder.beginComputePass(timer ? timer.desc(label) : undefined);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bind[slot]);
    pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
    pass.end();
  }

  clearAccum(encoder) { encoder.clearBuffer(this.accum); }
}
