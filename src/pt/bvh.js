// Binned-SAH BVH over triangles, flattened for the WGSL traversal.
//
// Node layout (2 x vec4f = 32 bytes):
//   [minX, minY, minZ, bitcast<u32>(leftOrFirst)], [maxX, maxY, maxZ, bitcast<u32>(count)]
// Interior: count == 0, children at leftOrFirst and leftOrFirst + 1 (always allocated after the parent,
// so a reverse walk over the node array is a valid bottom-up refit order).
// Leaf: triangles [leftOrFirst, leftOrFirst + count) in BVH order.

const BINS = 12;
const MAX_LEAF = 4;

export function buildBVH(pos, triCount) {
  // pos: Float32Array of 9 floats per triangle (v0, v1, v2).
  const cx = new Float32Array(triCount * 3);
  const bmin = new Float32Array(triCount * 3);
  const bmax = new Float32Array(triCount * 3);
  for (let t = 0; t < triCount; t++) {
    for (let a = 0; a < 3; a++) {
      const p0 = pos[t * 9 + a], p1 = pos[t * 9 + 3 + a], p2 = pos[t * 9 + 6 + a];
      const lo = Math.min(p0, p1, p2), hi = Math.max(p0, p1, p2);
      bmin[t * 3 + a] = lo;
      bmax[t * 3 + a] = hi;
      cx[t * 3 + a] = (lo + hi) * 0.5;
    }
  }
  const order = new Uint32Array(triCount);
  for (let i = 0; i < triCount; i++) order[i] = i;

  const maxNodes = Math.max(1, 2 * triCount);
  const f = new Float32Array(maxNodes * 8);
  const u = new Uint32Array(f.buffer);
  let used = 1;

  const binMin = new Float32Array(BINS * 3), binMax = new Float32Array(BINS * 3), binCnt = new Uint32Array(BINS);
  const rMin = new Float32Array(3), rMax = new Float32Array(3);

  function bounds(node, first, count) {
    let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
    for (let i = first; i < first + count; i++) {
      const t = order[i] * 3;
      if (bmin[t] < x0) x0 = bmin[t];
      if (bmin[t + 1] < y0) y0 = bmin[t + 1];
      if (bmin[t + 2] < z0) z0 = bmin[t + 2];
      if (bmax[t] > x1) x1 = bmax[t];
      if (bmax[t + 1] > y1) y1 = bmax[t + 1];
      if (bmax[t + 2] > z1) z1 = bmax[t + 2];
    }
    const o = node * 8;
    f[o] = x0; f[o + 1] = y0; f[o + 2] = z0; f[o + 4] = x1; f[o + 5] = y1; f[o + 6] = z1;
  }
  const area = (dx, dy, dz) => (dx < 0 ? 0 : dx * dy + dy * dz + dz * dx);

  const stack = [[0, 0, triCount, 0]];
  let maxDepth = 0;
  while (stack.length) {
    const [node, first, count, depth] = stack.pop();
    if (depth > maxDepth) maxDepth = depth;
    bounds(node, first, count);
    const o = node * 8;
    if (count <= MAX_LEAF) {
      u[o + 3] = first;
      u[o + 7] = count;
      continue;
    }
    // centroid bounds
    let c0 = [Infinity, Infinity, Infinity], c1 = [-Infinity, -Infinity, -Infinity];
    for (let i = first; i < first + count; i++) {
      const t = order[i] * 3;
      for (let a = 0; a < 3; a++) {
        const c = cx[t + a];
        if (c < c0[a]) c0[a] = c;
        if (c > c1[a]) c1[a] = c;
      }
    }
    let bestCost = Infinity, bestAxis = -1, bestSplit = 0;
    for (let a = 0; a < 3; a++) {
      const ext = c1[a] - c0[a];
      if (ext <= 1e-12) continue;
      binMin.fill(Infinity); binMax.fill(-Infinity); binCnt.fill(0);
      const scale = BINS / ext;
      for (let i = first; i < first + count; i++) {
        const t = order[i] * 3;
        const b = Math.min(BINS - 1, ((cx[t + a] - c0[a]) * scale) | 0);
        binCnt[b]++;
        for (let k = 0; k < 3; k++) {
          if (bmin[t + k] < binMin[b * 3 + k]) binMin[b * 3 + k] = bmin[t + k];
          if (bmax[t + k] > binMax[b * 3 + k]) binMax[b * 3 + k] = bmax[t + k];
        }
      }
      // sweep from the right
      const rightArea = new Float64Array(BINS), rightCnt = new Uint32Array(BINS);
      rMin.fill(Infinity); rMax.fill(-Infinity);
      let rc = 0;
      for (let b = BINS - 1; b > 0; b--) {
        rc += binCnt[b];
        for (let k = 0; k < 3; k++) {
          rMin[k] = Math.min(rMin[k], binMin[b * 3 + k]);
          rMax[k] = Math.max(rMax[k], binMax[b * 3 + k]);
        }
        rightArea[b] = rc ? area(rMax[0] - rMin[0], rMax[1] - rMin[1], rMax[2] - rMin[2]) : 0;
        rightCnt[b] = rc;
      }
      let lx0 = Infinity, ly0 = Infinity, lz0 = Infinity, lx1 = -Infinity, ly1 = -Infinity, lz1 = -Infinity, lc = 0;
      for (let b = 0; b < BINS - 1; b++) {
        lc += binCnt[b];
        lx0 = Math.min(lx0, binMin[b * 3]); ly0 = Math.min(ly0, binMin[b * 3 + 1]); lz0 = Math.min(lz0, binMin[b * 3 + 2]);
        lx1 = Math.max(lx1, binMax[b * 3]); ly1 = Math.max(ly1, binMax[b * 3 + 1]); lz1 = Math.max(lz1, binMax[b * 3 + 2]);
        if (!lc || !rightCnt[b + 1]) continue;
        const cost = lc * area(lx1 - lx0, ly1 - ly0, lz1 - lz0) + rightCnt[b + 1] * rightArea[b + 1];
        if (cost < bestCost) { bestCost = cost; bestAxis = a; bestSplit = b + 1; }
      }
    }
    const leafCost = count * area(f[o + 4] - f[o], f[o + 5] - f[o + 1], f[o + 6] - f[o + 2]);
    let mid;
    if (bestAxis < 0 || (bestCost >= leafCost && count <= 16)) {
      if (bestAxis < 0) {
        // all centroids coincide: median split by index
        mid = first + (count >> 1);
      } else {
        u[o + 3] = first;
        u[o + 7] = count;
        continue;
      }
    } else {
      const a = bestAxis, scale = BINS / (c1[a] - c0[a]);
      let i = first, j = first + count - 1;
      while (i <= j) {
        const b = Math.min(BINS - 1, ((cx[order[i] * 3 + a] - c0[a]) * scale) | 0);
        if (b < bestSplit) i++;
        else { const tmp = order[i]; order[i] = order[j]; order[j] = tmp; j--; }
      }
      mid = i;
      if (mid === first || mid === first + count) mid = first + (count >> 1);
    }
    const left = used;
    used += 2;
    u[o + 3] = left;
    u[o + 7] = 0;
    stack.push([left + 1, mid, first + count - mid, depth + 1]);
    stack.push([left, first, mid - first, depth + 1]);
  }
  return { nodes: f.slice(0, used * 8), nodeCount: used, order, maxDepth };
}

// Recompute node bounds after triangle positions moved (topology unchanged).
// triPos: Float32Array, 9 floats per triangle in BVH order.
export function refitBVH(nodes, nodeCount, triPos) {
  const u = new Uint32Array(nodes.buffer, nodes.byteOffset, nodes.length);
  for (let n = nodeCount - 1; n >= 0; n--) {
    const o = n * 8;
    const count = u[o + 7];
    let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
    if (count > 0) {
      const first = u[o + 3];
      for (let t = first; t < first + count; t++) {
        for (let v = 0; v < 3; v++) {
          const p = t * 9 + v * 3;
          const x = triPos[p], y = triPos[p + 1], z = triPos[p + 2];
          if (x < x0) x0 = x; if (y < y0) y0 = y; if (z < z0) z0 = z;
          if (x > x1) x1 = x; if (y > y1) y1 = y; if (z > z1) z1 = z;
        }
      }
    } else {
      const l = u[o + 3] * 8, r = l + 8;
      x0 = Math.min(nodes[l], nodes[r]); y0 = Math.min(nodes[l + 1], nodes[r + 1]); z0 = Math.min(nodes[l + 2], nodes[r + 2]);
      x1 = Math.max(nodes[l + 4], nodes[r + 4]); y1 = Math.max(nodes[l + 5], nodes[r + 5]); z1 = Math.max(nodes[l + 6], nodes[r + 6]);
    }
    nodes[o] = x0; nodes[o + 1] = y0; nodes[o + 2] = z0;
    nodes[o + 4] = x1; nodes[o + 5] = y1; nodes[o + 6] = z1;
  }
}

// GPU layout: only interior nodes, each holding both children's bounds (4 x vec4f = 64 bytes):
//   [lmin, lData], [lmax, lCount], [rmin, rData], [rmax, rCount]
// count > 0: leaf, data = first triangle (BVH order); count == 0: interior, data = packed node index.
// One 64-byte fetch per visited node gives both child boxes (the build layout needs three node fetches).
// A BVH that is a single leaf becomes one packed node with that leaf on both sides.
export function packedCount(nodes, nodeCount) {
  const u = new Uint32Array(nodes.buffer, nodes.byteOffset, nodeCount * 8);
  let k = 0;
  for (let n = 0; n < nodeCount; n++) if (u[n * 8 + 7] === 0) k++;
  return Math.max(k, 1);
}

// nodes: build layout (after buildBVH / refitBVH); out: Float32Array with packedCount(...) * 16 floats
export function packBVH(nodes, nodeCount, out) {
  const u = new Uint32Array(nodes.buffer, nodes.byteOffset, nodeCount * 8);
  const ou = new Uint32Array(out.buffer, out.byteOffset, out.length);
  const id = new Int32Array(nodeCount);
  let k = 0;
  for (let n = 0; n < nodeCount; n++) id[n] = u[n * 8 + 7] === 0 ? k++ : -1;
  const child = (o, c) => {
    const s = c * 8;
    for (let a = 0; a < 3; a++) { out[o + a] = nodes[s + a]; out[o + 4 + a] = nodes[s + 4 + a]; }
    const count = u[s + 7];
    ou[o + 3] = count > 0 ? u[s + 3] : id[c];
    ou[o + 7] = count;
  };
  if (k === 0) {
    child(0, 0);
    child(8, 0);
    return;
  }
  for (let n = 0; n < nodeCount; n++) {
    if (id[n] < 0) continue;
    const o = id[n] * 16, l = u[n * 8 + 3];
    child(o, l);
    child(o + 8, l + 1);
  }
}
