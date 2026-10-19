// Demo / training scenes. Everything is CC0 (Poly Haven, RobotExpressive) or procedural.
// Each scene: build() -> { scene, env(t), camera(t) -> {pos, target}, animate(t) }.
// Lighting: courtyard = golden-hour HDRI + its low sun, living room = afternoon sun through the window
// + warm chandelier, box = a swinging shaded pendant lamp.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { loadEnvMap, envSunLight } from './pt/envmap.js';

const A = '/assets/';
const loader = new GLTFLoader();
const texLoader = new THREE.ImageBitmapLoader();
texLoader.setOptions({ imageOrientation: 'none', premultiplyAlpha: 'none' });

async function tex(path) {
  const bmp = await texLoader.loadAsync(A + path);
  const t = new THREE.Texture(bmp);
  t.needsUpdate = true;
  return t;
}

// rough scales the roughness map, normal scales the normal map
async function pbr(name, { color = 0xffffff, uv = [1, 1], rough = 1, metal = 0, normal = 1 } = {}) {
  const [map, normalMap, arm] = await Promise.all([
    tex(`textures/${name}/${name}_diff.jpg`), tex(`textures/${name}/${name}_nor.jpg`), tex(`textures/${name}/${name}_arm.jpg`),
  ]);
  const m = new THREE.MeshStandardMaterial({
    color, map, normalMap, roughnessMap: arm, metalnessMap: arm, roughness: rough, metalness: metal,
    normalScale: new THREE.Vector2(normal, normal),
  });
  m.userData.uvScale = uv;
  return m;
}

// Loads a Poly Haven model, scales it, puts its bbox bottom on y = pos[1] and centers it at pos.xz.
async function model(id, { pos = [0, 0, 0], rotY = 0, scale = 1, height } = {}) {
  const g = await loader.loadAsync(`${A}polyhaven/${id}/${id}.gltf`);
  const root = new THREE.Group();
  const s = g.scene;
  root.add(s);
  s.updateMatrixWorld(true);
  let box = new THREE.Box3().setFromObject(s);
  let k = scale;
  if (height) k = height / (box.max.y - box.min.y);
  s.scale.multiplyScalar(k);
  s.updateMatrixWorld(true);
  box = new THREE.Box3().setFromObject(s);
  const c = box.getCenter(new THREE.Vector3());
  s.position.sub(new THREE.Vector3(c.x, box.min.y, c.z));
  root.position.set(...pos);
  root.rotation.y = rotY;
  root.name = id;
  return root;
}

// calls f(mesh) for the meshes of a loaded model whose material name contains `part`
function eachPart(root, part, f) {
  root.traverse((o) => { if (o.isMesh && o.material?.name?.includes(part)) f(o); });
}

function box(w, h, d, mat, pos, uvTiles) {
  const geo = new THREE.BoxGeometry(w, h, d);
  if (uvTiles) {
    // world-scale UVs so textures keep their size on every face
    const p = geo.attributes.position, n = geo.attributes.normal, uv = geo.attributes.uv;
    for (let i = 0; i < p.count; i++) {
      const ax = Math.abs(n.getX(i)), ay = Math.abs(n.getY(i));
      const x = p.getX(i) + pos[0], y = p.getY(i) + pos[1], z = p.getZ(i) + pos[2];
      if (ax > 0.5) uv.setXY(i, z / uvTiles, y / uvTiles);
      else if (ay > 0.5) uv.setXY(i, x / uvTiles, z / uvTiles);
      else uv.setXY(i, x / uvTiles, y / uvTiles);
    }
  }
  const m = new THREE.Mesh(geo, mat);
  m.position.set(...pos);
  return m;
}

async function robot({ clip = 'Dance', height = 1.3 } = {}) {
  const g = await loader.loadAsync(`${A}RobotExpressive/RobotExpressive.glb`);
  const s = g.scene;
  s.updateMatrixWorld(true);
  const bb = new THREE.Box3().setFromObject(s);
  s.scale.multiplyScalar(height / (bb.max.y - bb.min.y));
  const root = new THREE.Group();
  root.add(s);
  root.userData.dynamic = true;
  const mixer = new THREE.AnimationMixer(s);
  const a = THREE.AnimationClip.findByName(g.animations, clip) ?? g.animations[0];
  mixer.clipAction(a).play();
  return { root, mixer };
}

function sphereLight(pos, radius, power, color = [1, 0.85, 0.65], prevPos) {
  // power ~ radiant intensity scale: radiance chosen so that the sphere's intensity ~ power
  const L = power / (Math.PI * radius * radius);
  return { kind: 1, pos, prevPos, radius, radiance: color.map((c) => c * L) };
}

// ------------------------------------------------------------------ scenes
async function boxScene() {
  const scene = new THREE.Scene();
  const [wall, floor] = await Promise.all([
    pbr('plastered_wall_02', { uv: [1, 1], rough: 0.8, normal: 1.6 }), pbr('wood_floor_worn', { uv: [1, 1], rough: 0.5, normal: 1.3 }),
  ]);
  const red = wall.clone(); red.color = new THREE.Color(0.75, 0.12, 0.1);
  const green = wall.clone(); green.color = new THREE.Color(0.15, 0.55, 0.15);
  const white = wall.clone(); white.color = new THREE.Color(0.8, 0.8, 0.78);
  scene.add(box(4.2, 0.1, 4.2, floor, [0, -0.05, 0], 2));
  scene.add(box(4.2, 0.1, 4.2, white, [0, 3.05, 0], 2));
  scene.add(box(0.1, 3, 4.2, red, [-2.05, 1.5, 0], 2));
  scene.add(box(0.1, 3, 4.2, green, [2.05, 1.5, 0], 2));
  scene.add(box(4.2, 3, 0.1, white, [0, 1.5, -2.05], 2));
  const pedestal = new THREE.MeshStandardMaterial({ color: 0x8a8a90, roughness: 0.6 });
  scene.add(box(0.6, 0.8, 0.6, pedestal, [-1.2, 0.4, -1.1]));
  scene.add(await model('marble_bust_01', { pos: [-1.2, 0.8, -1.1], height: 0.65, rotY: 0.5 }));
  const glass = new THREE.MeshPhysicalMaterial({ color: 0xffffff, transmission: 1, ior: 1.5, roughness: 0 });
  const g = new THREE.Mesh(new THREE.SphereGeometry(0.42, 64, 32), glass);
  g.position.set(1.0, 0.42, 0.6);
  scene.add(g);
  const gold = new THREE.MeshStandardMaterial({ color: new THREE.Color(1.0, 0.76, 0.33), metalness: 1, roughness: 0.25 });
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.3, 64, 32), gold);
  m.position.set(1.3, 0.3, -1.2);
  scene.add(m);
  scene.add(await model('brass_vase_01', { pos: [-1.5, 0, 0.9], height: 0.5 }));
  scene.add(await model('rubber_duck_toy', { pos: [0.3, 0, 1.3], height: 0.18, rotY: -0.6 }));
  const bot = await robot({ clip: 'Dance', height: 1.25 });
  bot.root.position.set(-0.1, 0, -0.3);
  scene.add(bot.root);
  // a rigid mover: a cube that slides and spins
  const spin = new THREE.Group();
  spin.userData.dynamic = true;
  const cube = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.35, 0.35), new THREE.MeshStandardMaterial({ color: 0x3060c0, roughness: 0.35 }));
  spin.add(cube);
  scene.add(spin);
  // pendant lamp swinging slowly from the ceiling: cord + brass cone shade capped on top; the bulb is a sphere light
  // inside the shade, so the room is lit from below the shade and the ceiling only indirectly
  const PIVOT = [0.15, 3.0, 0.2], CORD = 0.62, SHADE_H = 0.22, BULB = CORD + 0.13;
  const lamp = new THREE.Group();
  lamp.userData.dynamic = true;
  lamp.position.set(...PIVOT);
  const brass = new THREE.MeshStandardMaterial({ color: new THREE.Color(0.95, 0.72, 0.42), metalness: 1, roughness: 0.32 });
  const cord = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, CORD, 8), new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.6 }));
  cord.position.y = -CORD / 2;
  const shade = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.21, SHADE_H, 48, 1, true), brass);
  shade.position.y = -CORD - SHADE_H / 2;
  // dark matte inside (a bright or metal inside right next to the bulb is a second light that only BSDF sampling
  // finds: fireflies on the floor)
  const enamel = new THREE.MeshStandardMaterial({ color: new THREE.Color(0.1, 0.09, 0.08), roughness: 0.7 });
  const inner = new THREE.Mesh(new THREE.CylinderGeometry(0.054, 0.203, SHADE_H - 0.004, 48, 1, true), enamel);
  inner.position.y = shade.position.y - 0.002;
  const cap = new THREE.Mesh(new THREE.CircleGeometry(0.06, 24).rotateX(-Math.PI / 2), brass);
  cap.position.y = -CORD;
  lamp.add(cord, shade, inner, cap);
  scene.add(lamp);
  const swing = (t) => [0.2 * Math.sin(t * 1.3), 0.07 * Math.sin(t * 0.9 + 1)];  // rotation about z, x
  const bulbPos = (t) => {
    const [az, ax] = swing(t);
    const v = new THREE.Vector3(0, -BULB, 0).applyEuler(new THREE.Euler(ax, 0, az));
    return [PIVOT[0] + v.x, PIVOT[1] + v.y, PIVOT[2] + v.z];
  };
  return {
    scene,
    exposure: 1.6,
    display: 1.4,  // extra display gain (demo / video post only)
    animate(t) {
      bot.mixer.setTime(t);
      spin.position.set(0.9 + 0.5 * Math.sin(t * 0.9), 0.175 + 0.25 * Math.abs(Math.sin(t * 1.8)), 1.3);
      spin.rotation.set(t * 0.8, t * 1.1, 0);
      const [az, ax] = swing(t);
      lamp.rotation.set(ax, 0, az);
    },
    env(t, dt) {
      return {
        skyZenith: [0.25, 0.35, 0.6], skyHorizon: [0.6, 0.65, 0.75], skyIntensity: 0.03, ground: [0.1, 0.1, 0.1],
        clampLum: 1.5, clampDirect: 3,  // firefly clamps as in the living room
        lights: [sphereLight(bulbPos(t), 0.05, 16, [1, 0.8, 0.56], bulbPos(t - dt))],
      };
    },
    camera(t) {
      const a = 0.35 * Math.sin(t * 0.25);
      return { pos: [Math.sin(a) * 5.2, 1.5 + 0.2 * Math.sin(t * 0.4), 2.2 + Math.cos(a) * 3.0], target: [0, 1.0, 0], fov: 50 };
    },
  };
}

// Living-room window in the left wall (opening x in [-3.6, -3.5], z in [-1.6, 1.6], y in [0.7, 2.5]): lined frame,
// four casement sashes with muntins under a transom, interior sill and casing, curtains on a rod. No glass: the path
// tracer's glass is a solid dielectric that blocks shadow rays (the sun would only come in as caustic noise) and the
// denoiser would filter the view outside as if it were on the glass surface.
function addWindow(scene) {
  const paint = new THREE.MeshStandardMaterial({ color: new THREE.Color(0.86, 0.85, 0.82), roughness: 0.45 });
  const Z0 = -1.6, Z1 = 1.6, Y0 = 0.7, Y1 = 2.5, F = 0.07;  // opening, frame board width
  const zc = (Z0 + Z1) / 2, yc = (Y0 + Y1) / 2, zw = Z1 - Z0, yh = Y1 - Y0;
  // frame lining the opening (deeper than the wall: a reveal)
  scene.add(box(0.16, F, zw, paint, [-3.53, Y0 + F / 2, zc]), box(0.16, F, zw, paint, [-3.53, Y1 - F / 2, zc]));
  scene.add(box(0.16, yh, F, paint, [-3.53, yc, Z0 + F / 2]), box(0.16, yh, F, paint, [-3.53, yc, Z1 - F / 2]));
  // transom bar, mullions between the four sashes
  const TY = 2.02;
  scene.add(box(0.08, 0.07, zw, paint, [-3.55, TY, zc]));
  for (const z of [-0.8, 0, 0.8]) scene.add(box(0.08, yh, z === 0 ? 0.08 : 0.06, paint, [-3.55, yc, z]));
  // sashes: a border, one vertical and one horizontal muntin each (4 panes); transom lights get one vertical muntin
  const S = 0.045, M = 0.022;
  for (const [a, b] of [[Z0 + F, -0.83], [-0.77, -0.04], [0.04, 0.77], [0.83, Z1 - F]]) {
    const z = (a + b) / 2, w = b - a;
    const y0 = Y0 + F, y1 = TY - 0.035, h = y1 - y0, ym = (y0 + y1) / 2;
    scene.add(box(0.05, S, w, paint, [-3.55, y0 + S / 2, z]), box(0.05, S, w, paint, [-3.55, y1 - S / 2, z]));
    scene.add(box(0.05, h, S, paint, [-3.55, ym, a + S / 2]), box(0.05, h, S, paint, [-3.55, ym, b - S / 2]));
    scene.add(box(0.04, h, M, paint, [-3.55, ym, z]), box(0.04, M, w, paint, [-3.55, ym, z]));
    scene.add(box(0.04, Y1 - F - TY - 0.035, M, paint, [-3.55, (TY + 0.035 + Y1 - F) / 2, z]));
  }
  // interior sill, apron and casing on the wall face
  scene.add(box(0.24, 0.035, zw + 0.3, paint, [-3.4, Y0 + 0.0175, zc]));
  scene.add(box(0.025, 0.09, zw + 0.2, paint, [-3.49, Y0 - 0.045, zc]));
  scene.add(box(0.025, yh + 0.09, 0.09, paint, [-3.49, yc + 0.045, Z0 - 0.045]), box(0.025, yh + 0.09, 0.09, paint, [-3.49, yc + 0.045, Z1 + 0.045]));
  scene.add(box(0.025, 0.09, zw + 0.18, paint, [-3.49, Y1 + 0.045, zc]));
  // curtains (folded cloth) on a rod
  const cloth = new THREE.MeshStandardMaterial({ color: new THREE.Color(0.8, 0.76, 0.68), roughness: 0.95 });
  for (const zMid of [Z0 - 0.38, Z1 + 0.38]) {
    const geo = new THREE.PlaneGeometry(0.72, 2.62, 48, 1);
    const p = geo.attributes.position;
    for (let i = 0; i < p.count; i++) p.setZ(i, 0.035 * Math.sin(p.getX(i) * 26) + 0.012 * Math.sin(p.getX(i) * 61));
    geo.computeVertexNormals();
    const c = new THREE.Mesh(geo, cloth);
    c.rotation.y = Math.PI / 2;
    c.position.set(-3.36, 1.39, zMid);
    scene.add(c);
  }
  const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, zw + 1.7, 16).rotateX(Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: 0x2a2724, metalness: 1, roughness: 0.4 }));
  rod.position.set(-3.33, 2.72, zc);
  scene.add(rod);
}

async function livingScene() {
  const scene = new THREE.Scene();
  const [wall, floor, map] = await Promise.all([
    pbr('beige_wall_001', { uv: [1, 1], rough: 0.85, normal: 2 }), pbr('herringbone_parquet', { uv: [1, 1], rough: 0.6, normal: 1.2 }),
    loadEnvMap(`${A}hdri/spaichingen_hill_2k.hdr`),
  ]);
  wall.color = new THREE.Color(0.94, 0.97, 1.0);
  const W = 7, D = 6, H = 3;
  scene.add(box(W + 0.4, 0.1, D + 0.4, floor, [0, -0.05, 0], 2.5));
  scene.add(box(W + 0.4, 0.1, D + 0.4, wall, [0, H + 0.05, 0], 2));
  scene.add(box(W + 0.4, H, 0.1, wall, [0, H / 2, -D / 2 - 0.05], 2));
  scene.add(box(W + 0.4, H, 0.1, wall, [0, H / 2, D / 2 + 0.05], 2));
  scene.add(box(0.1, H, D + 0.2, wall, [W / 2 + 0.05, H / 2, 0], 2));
  // left wall with a large window (z in [-1.6, 1.6], y in [0.7, 2.5])
  const x = -W / 2 - 0.05;
  scene.add(box(0.1, 0.7, D + 0.2, wall, [x, 0.35, 0], 2));
  scene.add(box(0.1, H - 2.5, D + 0.2, wall, [x, (2.5 + H) / 2, 0], 2));
  scene.add(box(0.1, 1.8, D / 2 - 1.6 + 0.1, wall, [x, 1.6, -(1.6 + D / 2 + 0.1) / 2], 2));
  scene.add(box(0.1, 1.8, D / 2 - 1.6 + 0.1, wall, [x, 1.6, (1.6 + D / 2 + 0.1) / 2], 2));
  addWindow(scene);
  scene.add(await model('Sofa_01', { pos: [0, 0, -2.45], height: 0.85 }));
  scene.add(await model('throw_pillows_01', { pos: [0.6, 0.45, -2.5], height: 0.4 }));
  scene.add(await model('CoffeeTable_01', { pos: [0, 0, -1.1], height: 0.42 }));
  scene.add(await model('tea_set_01', { pos: [0.1, 0.42, -1.1], height: 0.18 }));
  scene.add(await model('ArmChair_01', { pos: [2.2, 0, -0.6], height: 0.95, rotY: -Math.PI / 2 - 0.4 }));
  scene.add(await model('wooden_bookshelf_worn', { pos: [3.2, 0, 1.4], height: 2.0, rotY: -Math.PI / 2 }));
  scene.add(await model('potted_plant_04', { pos: [-2.9, 0, -2.5], height: 1.3 }));
  scene.add(await model('side_table_01', { pos: [-1.5, 0, -2.5], height: 0.55 }));
  scene.add(await model('wine_bottles_01', { pos: [-1.5, 0.55, -2.5], height: 0.33 }));
  scene.add(await model('ceramic_vase_01', { pos: [1.3, 0, -2.6], height: 0.5 }));
  const chandelier = await model('Chandelier_01', { pos: [0, H - 0.9, -0.8], height: 0.9 });
  // the lamp shades glow (meshes get no light sampling, so they stay dim); the light is a sphere light in the middle
  eachPart(chandelier, 'lamps', (o) => {
    o.material = o.material.clone();
    o.material.emissive = new THREE.Color(1, 0.62, 0.32);
    o.material.emissiveIntensity = 2.5;
  });
  chandelier.userData.noShadow = true;  // its own light shines through it (shadow rays skip it, see pt/scene.js)
  scene.add(chandelier);
  const bot = await robot({ clip: 'Walking', height: 1.1 });
  scene.add(bot.root);
  // late-afternoon sun through the window: the HDRI's own sun, cut out of the map and used as the sun light
  const ROT = -2.15;
  const sun = envSunLight(map, ROT);
  const lamp = sphereLight([0, H - 0.55, -0.8], 0.07, 2.5, [1, 0.7, 0.42]);
  return {
    scene,
    exposure: 1.6,
    camBox: [[-3.0, 0.4, -2.6], [3.0, 2.6, 2.7]],
    animate(t) {
      bot.mixer.setTime(t);
      const a = t * 0.35;
      bot.root.position.set(Math.cos(a) * 1.3 - 0.4, 0, Math.sin(a) * 0.9 + 0.6);
      bot.root.rotation.y = -a;
    },
    env() {
      return {
        skyZenith: [0.35, 0.55, 0.95], skyHorizon: [0.75, 0.82, 0.92], skyIntensity: 1.2, ground: [0.3, 0.28, 0.25],
        map, mapRotation: ROT, mapIntensity: 1, lights: [sun, lamp],
        // indirect firefly clamp (default 16): rare bright indirect paths showed up as flashes the denoiser spreads
        // into glowing blobs; 1.5 removes them, mean brightness -0.3%. clampDirect: glints on the teacups
        clampLum: 1.5, clampDirect: 3,
      };
    },
    camera(t) {
      const a = 0.4 * Math.sin(t * 0.2);
      return { pos: [1.2 + 1.2 * Math.sin(a), 1.45, 2.6 + 0.3 * Math.cos(t * 0.3)], target: [-0.4, 0.8, -1.4], fov: 60 };
    },
  };
}

async function courtyardScene() {
  const scene = new THREE.Scene();
  const [ground, brick, map] = await Promise.all([
    pbr('cobblestone_floor_01', { uv: [1, 1], rough: 0.8, normal: 1.6 }), pbr('brick_wall_001', { uv: [1, 1], normal: 1.5 }),
    loadEnvMap(`${A}hdri/hilly_terrain_01_2k.hdr`),
  ]);
  scene.add(box(26, 0.1, 26, ground, [0, -0.05, 0], 2));
  scene.add(box(14, 3.2, 0.4, brick, [0, 1.6, -6.2], 2.5));
  scene.add(box(0.4, 3.2, 12, brick, [-7, 1.6, -0.4], 2.5));
  scene.add(await model('wooden_picnic_table', { pos: [0, 0, -2], height: 0.78 }));
  scene.add(await model('concrete_cat_statue', { pos: [0.3, 0.78, -2.1], height: 0.35, rotY: 0.8 }));
  scene.add(await model('Barrel_01', { pos: [-4.5, 0, -4.8], height: 0.9 }));
  scene.add(await model('Barrel_01', { pos: [-3.7, 0, -5.2], height: 0.9, rotY: 1.2 }));
  scene.add(await model('wooden_crate_01', { pos: [-5.6, 0, -3.2], height: 0.6 }));
  scene.add(await model('wooden_crate_01', { pos: [-5.6, 0.6, -3.2], height: 0.6, rotY: 0.3 }));
  scene.add(await model('garden_gnome', { pos: [1.8, 0, -0.2], height: 0.45, rotY: -0.5 }));
  scene.add(await model('boulder_01', { pos: [4.5, 0, -4.2], height: 1.2 }));
  scene.add(await model('tree_stump_01', { pos: [3.2, 0, 1.2], height: 0.5 }));
  scene.add(await model('potted_plant_02', { pos: [-2.2, 0, -5.4], height: 1.1 }));
  const streetLamp = await model('street_lamp_01', { pos: [5.5, 0, -5.5], height: 3.5 });
  // the lamp is on at dusk: a glowing bulb behind clear glass (emission only: a light there would cost ~10 ms at 1080p
  // for almost no light next to the sun)
  eachPart(streetLamp, 'glass', (o) => { o.material = new THREE.MeshPhysicalMaterial({ color: 0xffffff, transmission: 1, ior: 1.5, roughness: 0 }); });
  eachPart(streetLamp, 'bulb', (o) => {
    o.material = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: new THREE.Color(1, 0.7, 0.4), emissiveIntensity: 30 });
  });
  scene.add(streetLamp);
  scene.add(await model('fir_sapling', { pos: [-5.2, 0, 1.5], height: 1.8 }));
  scene.add(await model('shrub_01', { pos: [2.5, 0, -5.4], height: 1.0 }));
  scene.add(await model('planter_box_01', { pos: [-0.8, 0, -5.6], height: 0.5 }));
  const bot = await robot({ clip: 'Running', height: 1.2 });
  scene.add(bot.root);
  // golden hour: the HDRI's low sun (cut out of the map) from the open right-back side, across the camera's view, so
  // shadows run long towards the front left
  const ROT = 1.2;
  const sun = envSunLight(map, ROT, { scale: 3 });
  return {
    scene,
    exposure: 1.3,
    camBox: [[-12, 0.4, -12], [12, 4, 12]],
    animate(t) {
      bot.mixer.setTime(t);
      const a = t * 0.6;
      bot.root.position.set(Math.cos(a) * 2.8, 0, Math.sin(a) * 2.0 - 1.5);
      bot.root.rotation.y = -a;
    },
    env() {
      return {
        skyZenith: [0.3, 0.5, 0.95], skyHorizon: [0.8, 0.8, 0.85], skyIntensity: 1.0, ground: [0.3, 0.27, 0.22],
        map, mapRotation: ROT, mapIntensity: 0.75, lights: [sun],
      };
    },
    camera(t) {
      const a = 0.6 + 0.5 * Math.sin(t * 0.15);
      return { pos: [Math.sin(a) * 7.5, 1.7 + 0.3 * Math.sin(t * 0.25), Math.cos(a) * 7.5 - 1], target: [-0.5, 0.7, -2.2], fov: 55 };
    },
  };
}

export const SCENES = { box: boxScene, living: livingScene, courtyard: courtyardScene };
