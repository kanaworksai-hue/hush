// Turns a three.js scene graph into the GPU buffers the path tracer reads.
//
// Every mesh is baked into world space. Meshes under an object with userData.dynamic = true
// (rigid movers, skinned characters) go into their own BLAS that is re-skinned / re-transformed
// on the CPU and refit every frame; their per-vertex previous positions give exact motion vectors.
// Meshes under an object with userData.noShadow = true (light fixtures around a sphere light) get their own BLAS that
// shadow rays skip: the light shines through its fixture instead of every shadow ray crossing the fixture's mesh.
import * as THREE from 'three';
import { buildBVH, refitBVH, packBVH, packedCount } from './bvh.js';

const MAX_BLAS = 8;
export const TEX_SIZE = 1024;
const TEX_MIPS = 11;

function geomTris(mesh) {
  const g = mesh.geometry;
  const n = g.attributes.position.count;
  const index = g.index ? g.index.array : Array.from({ length: n }, (_, i) => i);
  return { index, vertexCount: n };
}

export class SceneGPU {
  constructor(device) {
    this.device = device;
    this.materials = [];
    this.matIndex = new Map();
    this.textures = [];       // images in layer order
    this.texIndex = new Map();
    this.blas = [];
  }

  // ------------------------------------------------------------ materials
  layerFor(tex) {
    if (!tex || !tex.image) return -1;
    const key = tex.source?.uuid ?? tex.uuid;
    if (!this.texIndex.has(key)) {
      this.texIndex.set(key, this.textures.length);
      this.textures.push(tex.image);
    }
    return this.texIndex.get(key);
  }

  materialFor(m) {
    if (this.matIndex.has(m.uuid)) return this.matIndex.get(m.uuid);
    const id = this.materials.length;
    const ud = m.userData ?? {};
    const em = m.emissive ? m.emissive.clone().multiplyScalar(m.emissiveIntensity ?? 1) : new THREE.Color(0, 0, 0);
    const transmission = m.transmission ?? ud.transmission ?? 0;
    const alphaCut = m.alphaTest > 0 ? m.alphaTest : 0;
    this.materials.push({
      base: [m.color?.r ?? 1, m.color?.g ?? 1, m.color?.b ?? 1, alphaCut],
      emissive: [em.r, em.g, em.b, 0],
      pbr: [m.roughness ?? 1, m.metalness ?? 0, transmission, m.ior ?? 1.5],
      tex: [this.layerFor(m.map), this.layerFor(m.normalMap), this.layerFor(m.roughnessMap ?? m.metalnessMap), 0],
      uv: [ud.uvScale?.[0] ?? 1, ud.uvScale?.[1] ?? 1, m.normalScale ? m.normalScale.x : 1, 0],
      texSize: m.map?.image?.width ?? TEX_SIZE,
    });
    this.matIndex.set(m.uuid, id);
    return id;
  }

  // ------------------------------------------------------------ build
  // groups: [{ root: Object3D, dynamic: bool }]
  async build(scene) {
    scene.updateMatrixWorld(true);
    const groups = [{ dynamic: false, meshes: [] }];
    const dynRoots = [];
    scene.traverse((o) => {
      if (o.userData.dynamic || o.userData.noShadow) dynRoots.push(o);
    });
    for (const r of dynRoots) groups.push({ dynamic: !!r.userData.dynamic, noShadow: !!r.userData.noShadow, root: r, meshes: [] });
    scene.traverse((o) => {
      if (!o.isMesh || !o.visible || o.userData.skipPT) return;
      let p = o, g = groups[0];
      while (p) {
        const i = dynRoots.indexOf(p);
        if (i >= 0) { g = groups[i + 1]; break; }
        p = p.parent;
      }
      g.meshes.push(o);
    });
    if (groups.length > MAX_BLAS) throw new Error('too many dynamic groups');

    // count
    let triTotal = 0, vertTotal = 0;
    for (const g of groups) {
      g.triCount = 0; g.vertCount = 0;
      for (const m of g.meshes) {
        const { index, vertexCount } = geomTris(m);
        g.triCount += index.length / 3;
        g.vertCount += vertexCount;
      }
      triTotal += g.triCount; vertTotal += g.vertCount;
    }
    this.triTotal = triTotal;
    this.vertTotal = vertTotal;
    this.verts = new Float32Array(vertTotal * 12);
    this.triIdx = new Uint32Array(triTotal * 4);
    this.triPos = new Float32Array(triTotal * 9);
    let triOff = 0, vertOff = 0, nodeOff = 0;
    const tmp = new THREE.Vector3(), nrm = new THREE.Matrix3();
    for (const g of groups) {
      g.triOffset = triOff; g.vertOffset = vertOff;
      // local (unordered) triangle list for this group
      const pos = new Float32Array(g.triCount * 9);
      const idx = new Uint32Array(g.triCount * 4);
      let t = 0, v = vertOff;
      g.meshParts = [];
      for (const m of g.meshes) {
        const geo = m.geometry;
        const { index, vertexCount } = geomTris(m);
        const P = geo.attributes.position, N = geo.attributes.normal, UV = geo.attributes.uv;
        const mats = Array.isArray(m.material) ? m.material : [m.material];
        const groupsOf = geo.groups.length ? geo.groups : [{ start: 0, count: index.length, materialIndex: 0 }];
        nrm.getNormalMatrix(m.matrixWorld);
        g.meshParts.push({ mesh: m, vertOffset: v, vertexCount, index });
        for (let i = 0; i < vertexCount; i++) {
          const o = (v + i) * 12;
          if (m.isSkinnedMesh) m.getVertexPosition(i, tmp); else tmp.fromBufferAttribute(P, i);
          tmp.applyMatrix4(m.matrixWorld);
          this.verts[o] = tmp.x; this.verts[o + 1] = tmp.y; this.verts[o + 2] = tmp.z;
          this.verts[o + 3] = UV ? UV.getX(i) : 0;
          this.verts[o + 4] = tmp.x; this.verts[o + 5] = tmp.y; this.verts[o + 6] = tmp.z;
          if (N) tmp.fromBufferAttribute(N, i).applyMatrix3(nrm).normalize(); else tmp.set(0, 1, 0);
          this.verts[o + 8] = tmp.x; this.verts[o + 9] = tmp.y; this.verts[o + 10] = tmp.z;
          this.verts[o + 11] = UV ? UV.getY(i) : 0;
        }
        for (const gr of groupsOf) {
          const mat = mats[gr.materialIndex ?? 0] ?? mats[0];
          const mid = this.materialFor(mat);
          const end = Math.min(index.length, gr.start + gr.count);
          for (let k = gr.start; k < end; k += 3) {
            const a = v + index[k], b = v + index[k + 1], c = v + index[k + 2];
            idx[t * 4] = a; idx[t * 4 + 1] = b; idx[t * 4 + 2] = c; idx[t * 4 + 3] = mid;
            for (let j = 0; j < 3; j++) {
              const vo = [a, b, c][j] * 12;
              pos[t * 9 + j * 3] = this.verts[vo];
              pos[t * 9 + j * 3 + 1] = this.verts[vo + 1];
              pos[t * 9 + j * 3 + 2] = this.verts[vo + 2];
            }
            t++;
          }
        }
        v += vertexCount;
      }
      g.triCount = t;
      if (g.dynamic && m0Skinned(g)) g.recomputeNormals = true;
      const bvh = buildBVH(pos, t);
      g.order = bvh.order;
      g.nodes = bvh.nodes;
      g.nodeCount = bvh.nodeCount;
      this.maxDepth = Math.max(this.maxDepth ?? 0, bvh.maxDepth);
      g.packCount = packedCount(bvh.nodes, bvh.nodeCount);
      g.packOffset = nodeOff;
      g.localIdx = idx;
      // reorder triangles into BVH order
      for (let i = 0; i < t; i++) {
        const s = bvh.order[i];
        this.triPos.set(pos.subarray(s * 9, s * 9 + 9), (triOff + i) * 9);
        this.triIdx.set(idx.subarray(s * 4, s * 4 + 4), (triOff + i) * 4);
      }
      nodeOff += g.packCount;
      triOff += t;
      vertOff += g.vertCount;
    }
    this.groups = groups;
    this.nodes = new Float32Array(nodeOff * 16);
    this.tris = new Float32Array(triTotal * 12);
    for (const g of groups) {
      packBVH(g.nodes, g.nodeCount, this.nodes.subarray(g.packOffset * 16, (g.packOffset + g.packCount) * 16));
    }
    if (this.maxDepth > 32) console.warn(`BVH depth ${this.maxDepth} > traversal stack (32, pathtracer.wgsl.js)`);
    this.computeTriLod();
    this.packTris(0, triTotal);
    await this.upload();
  }

  computeTriLod() {
    // f16 log2(texels per world unit) in the high half of triIdx.w
    const f16 = new Uint16Array(1);
    const f32 = new Float32Array(1);
    const toHalf = (val) => {
      f32[0] = val;
      const x = new Uint32Array(f32.buffer)[0];
      const sign = (x >> 16) & 0x8000;
      let e = ((x >> 23) & 0xff) - 127 + 15;
      const m = (x >> 13) & 0x3ff;
      if (e <= 0) return sign;
      if (e >= 31) return sign | 0x7c00;
      return sign | (e << 10) | m;
    };
    for (let t = 0; t < this.triTotal; t++) {
      const i = this.triIdx.subarray(t * 4, t * 4 + 4);
      const mat = this.materials[i[3] & 0xffff];
      const V = this.verts;
      const a = i[0] * 12, b = i[1] * 12, c = i[2] * 12;
      const e1 = [V[b] - V[a], V[b + 1] - V[a + 1], V[b + 2] - V[a + 2]];
      const e2 = [V[c] - V[a], V[c + 1] - V[a + 1], V[c + 2] - V[a + 2]];
      const cr = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      const wa = Math.hypot(...cr) * 0.5;
      const u1 = [V[b + 3] - V[a + 3], V[b + 11] - V[a + 11]], u2 = [V[c + 3] - V[a + 3], V[c + 11] - V[a + 11]];
      const ua = Math.abs(u1[0] * u2[1] - u1[1] * u2[0]) * 0.5;
      const lod = wa > 1e-12 && ua > 1e-14 ? Math.log2(Math.sqrt(ua / wa) * TEX_SIZE) : 0;
      f16[0] = toHalf(lod);
      i[3] = (i[3] & 0xffff) | (f16[0] << 16);
    }
    void f16;
  }

  // GPU triangle records (3 x vec4f): [v0, alpha-tested ? 1 : 0], [v1 - v0, 0], [v2 - v0, 0]
  packTris(first, count) {
    const P = this.triPos, T = this.tris;
    for (let t = first; t < first + count; t++) {
      const p = t * 9, o = t * 12;
      const m = this.materials[this.triIdx[t * 4 + 3] & 0xffff];
      T[o] = P[p]; T[o + 1] = P[p + 1]; T[o + 2] = P[p + 2];
      T[o + 3] = m.base[3] > 0 && m.tex[0] >= 0 ? 1 : 0;
      for (let k = 1; k < 3; k++) {
        T[o + k * 4] = P[p + k * 3] - P[p]; T[o + k * 4 + 1] = P[p + k * 3 + 1] - P[p + 1]; T[o + k * 4 + 2] = P[p + k * 3 + 2] - P[p + 2];
      }
    }
  }

  // ------------------------------------------------------------ per-frame update of dynamic groups
  update() {
    const tmp = new THREE.Vector3(), nrm = new THREE.Matrix3();
    let changed = false;
    for (const g of this.groups) {
      if (!g.dynamic) continue;
      changed = true;
      g.root.updateMatrixWorld(true);
      const V = this.verts;
      for (const part of g.meshParts) {
        const m = part.mesh;
        const P = m.geometry.attributes.position, N = m.geometry.attributes.normal;
        nrm.getNormalMatrix(m.matrixWorld);
        for (let i = 0; i < part.vertexCount; i++) {
          const o = (part.vertOffset + i) * 12;
          V[o + 4] = V[o]; V[o + 5] = V[o + 1]; V[o + 6] = V[o + 2];
          if (m.isSkinnedMesh) m.getVertexPosition(i, tmp); else tmp.fromBufferAttribute(P, i);
          tmp.applyMatrix4(m.matrixWorld);
          V[o] = tmp.x; V[o + 1] = tmp.y; V[o + 2] = tmp.z;
          if (!m.isSkinnedMesh) {
            if (N) tmp.fromBufferAttribute(N, i).applyMatrix3(nrm).normalize(); else tmp.set(0, 1, 0);
            V[o + 8] = tmp.x; V[o + 9] = tmp.y; V[o + 10] = tmp.z;
          }
        }
        if (m.isSkinnedMesh) this.skinnedNormals(part);
      }
      // triangles in BVH order
      for (let i = 0; i < g.triCount; i++) {
        const s = g.order[i];
        const dst = (g.triOffset + i) * 9;
        for (let j = 0; j < 3; j++) {
          const vo = g.localIdx[s * 4 + j] * 12;
          this.triPos[dst + j * 3] = V[vo];
          this.triPos[dst + j * 3 + 1] = V[vo + 1];
          this.triPos[dst + j * 3 + 2] = V[vo + 2];
        }
      }
      refitBVH(g.nodes, g.nodeCount, this.triPos.subarray(g.triOffset * 9, (g.triOffset + g.triCount) * 9));
      packBVH(g.nodes, g.nodeCount, this.nodes.subarray(g.packOffset * 16, (g.packOffset + g.packCount) * 16));
      this.packTris(g.triOffset, g.triCount);
      const q = this.device.queue;
      q.writeBuffer(this.bufNodes, g.packOffset * 64, this.nodes, g.packOffset * 16, g.packCount * 16);
      q.writeBuffer(this.bufTris, g.triOffset * 48, this.tris, g.triOffset * 12, g.triCount * 12);
      q.writeBuffer(this.bufVerts, g.vertOffset * 48, this.verts, g.vertOffset * 12, g.vertCount * 12);
    }
    return changed;
  }

  skinnedNormals(part) {
    // area-weighted smooth normals from the skinned positions
    const V = this.verts, idx = part.index, base = part.vertOffset;
    for (let i = 0; i < part.vertexCount; i++) {
      const o = (base + i) * 12;
      V[o + 8] = 0; V[o + 9] = 0; V[o + 10] = 0;
    }
    for (let k = 0; k < idx.length; k += 3) {
      const a = (base + idx[k]) * 12, b = (base + idx[k + 1]) * 12, c = (base + idx[k + 2]) * 12;
      const e1x = V[b] - V[a], e1y = V[b + 1] - V[a + 1], e1z = V[b + 2] - V[a + 2];
      const e2x = V[c] - V[a], e2y = V[c + 1] - V[a + 1], e2z = V[c + 2] - V[a + 2];
      const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
      for (const o of [a, b, c]) { V[o + 8] += nx; V[o + 9] += ny; V[o + 10] += nz; }
    }
    for (let i = 0; i < part.vertexCount; i++) {
      const o = (base + i) * 12;
      const l = Math.hypot(V[o + 8], V[o + 9], V[o + 10]) || 1;
      V[o + 8] /= l; V[o + 9] /= l; V[o + 10] /= l;
    }
  }

  // ------------------------------------------------------------ GPU
  async upload() {
    const d = this.device;
    const mk = (data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) => {
      const b = d.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 16) * 16), usage });
      d.queue.writeBuffer(b, 0, data);
      return b;
    };
    this.bufNodes = mk(this.nodes);
    this.bufTris = mk(this.tris);
    this.bufTriIdx = mk(this.triIdx);
    this.bufVerts = mk(this.verts);
    const md = new Float32Array(Math.max(1, this.materials.length) * 20);
    const mi = new Int32Array(md.buffer);
    this.materials.forEach((m, i) => {
      md.set(m.base, i * 20); md.set(m.emissive, i * 20 + 4); md.set(m.pbr, i * 20 + 8);
      mi.set(m.tex, i * 20 + 12); md.set(m.uv, i * 20 + 16);
    });
    this.bufMats = mk(md);

    const layers = Math.max(1, this.textures.length);
    this.texArray = d.createTexture({
      size: [TEX_SIZE, TEX_SIZE, layers], mipLevelCount: TEX_MIPS, format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    for (let l = 0; l < this.textures.length; l++) {
      const img = this.textures[l];
      for (let mip = 0; mip < TEX_MIPS; mip++) {
        const s = Math.max(1, TEX_SIZE >> mip);
        const bmp = await createImageBitmap(img, {
          resizeWidth: s, resizeHeight: s, resizeQuality: 'high', premultiplyAlpha: 'none', colorSpaceConversion: 'none',
        });
        d.queue.copyExternalImageToTexture({ source: bmp }, { texture: this.texArray, mipLevel: mip, origin: [0, 0, l] }, [s, s]);
        bmp.close();
      }
    }
    this.sampler = d.createSampler({
      magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', maxAnisotropy: 1,
    });
  }

  destroy() {
    for (const b of [this.bufNodes, this.bufTris, this.bufTriIdx, this.bufVerts, this.bufMats, this.texArray]) b?.destroy();
  }

  blasTable() {
    // flags: 1 = dynamic, 2 = skipped by shadow rays
    return this.groups.map((g) => ({
      nodeOffset: g.packOffset, triOffset: g.triOffset, nodeCount: g.packCount, dynamic: (g.dynamic ? 1 : 0) | (g.noShadow ? 2 : 0),
    }));
  }
}

function m0Skinned(g) {
  return g.meshes.some((m) => m.isSkinnedMesh);
}
