// Hush + three.js WebGPURenderer.
//
// three.js owns the device and the canvas. The scene graph is path traced at 1 spp by the Hush path tracer
// (G-buffer + exact motion vectors), denoised by Hush on the same device, and the result is drawn by three.js as an
// ExternalTexture on a fullscreen quad (three.js does the tone mapping). Drag to orbit.
//
//   examples/three/?scale=0.75   render scale (default: about 1 MPixel)
import * as THREE from 'three/webgpu';
import { texture } from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';
import { Hush } from 'hush';
import { SceneGPU, PathTracer, JITTER, ThreeEnvironment } from 'hush/pathtracer';

const q = new URLSearchParams(location.search);
const hud = document.getElementById('hud');

// 1. renderer + device (Hush needs a few storage textures more than the WebGPU defaults)
const renderer = new THREE.WebGPURenderer({ antialias: false, requiredLimits: { ...Hush.requiredLimits } });
await renderer.init();
if (!renderer.backend.isWebGPUBackend) throw new Error('needs WebGPU');
const device = renderer.backend.device;
document.body.appendChild(renderer.domElement);
renderer.toneMapping = THREE.AgXToneMapping;
renderer.toneMappingExposure = 1.0;

// 2. an ordinary three.js scene (CC0 Poly Haven models). userData.dynamic marks subtrees that move.
const scene = new THREE.Scene();
const loader = new GLTFLoader();
async function model(id, { pos, height, rotY = 0 }) {
  const g = (await loader.loadAsync(`/assets/polyhaven/${id}/${id}.gltf`)).scene;
  const box = new THREE.Box3().setFromObject(g);
  g.scale.setScalar(height / (box.max.y - box.min.y));
  box.setFromObject(g);
  const c = box.getCenter(new THREE.Vector3());
  g.position.set(-c.x, -box.min.y, -c.z);
  const root = new THREE.Group();
  root.add(g);
  root.position.set(...pos);
  root.rotation.y = rotY;
  return root;
}
const plaster = new THREE.MeshStandardMaterial({ color: 0xd8d2c8, roughness: 0.9 });
const floor = new THREE.Mesh(new THREE.BoxGeometry(12, 0.1, 12), new THREE.MeshStandardMaterial({ color: 0x8a8378, roughness: 0.7 }));
floor.position.y = -0.05;
const wall = new THREE.Mesh(new THREE.BoxGeometry(12, 4, 0.2), plaster);
wall.position.set(0, 2, -2.5);
const sideWall = new THREE.Mesh(new THREE.BoxGeometry(0.2, 4, 6), plaster);
sideWall.position.set(-3.2, 2, 0.5);
scene.add(floor, wall, sideWall);
const pedestals = [[-1.6, 0.9], [0, 1.1], [1.6, 0.8]];
for (const [x, h] of pedestals) {
  const p = new THREE.Mesh(new THREE.BoxGeometry(0.6, h, 0.6), plaster);
  p.position.set(x, h / 2, -1.2);
  scene.add(p);
}
scene.add(await model('marble_bust_01', { pos: [0, 1.1, -1.2], height: 0.6 }));
scene.add(await model('brass_vase_01', { pos: [-1.6, 0.9, -1.2], height: 0.45 }));
scene.add(await model('ceramic_vase_01', { pos: [1.6, 0.8, -1.2], height: 0.5 }));
scene.add(await model('potted_plant_02', { pos: [2.6, 0, -1.8], height: 1.2 }));
const duck = await model('rubber_duck_toy', { pos: [0, 0, 0.6], height: 0.3 });
duck.userData.dynamic = true;
scene.add(duck);

// lights: an HDRI as scene.environment (sky + its sun, importance sampled; Poly Haven CC0) and a small moving warm lamp.
// DirectionalLight and HemisphereLight work too (see ThreeEnvironment).
scene.environment = await new HDRLoader().loadAsync('/assets/hdri/hilly_terrain_01_2k.hdr');
scene.environment.mapping = THREE.EquirectangularReflectionMapping;
scene.environmentRotation.y = -0.5;  // low sun from the open front right
const lamp = new THREE.PointLight(0xffb070, 1.2);
lamp.userData.radius = 0.06;
scene.add(lamp);

const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
camera.position.set(1.4, 1.5, 3.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.9, -0.8);
controls.enableDamping = true;
controls.update();

// 3. path tracer + Hush on three's device
const sg = new SceneGPU(device);
await sg.build(scene);
const envOf = new ThreeEnvironment();
let pt = null, hush = null;

// 4. show the denoised texture through three.js: two ExternalTextures (Hush alternates two outputs)
const ext = new Map();
const tex = texture(new THREE.ExternalTexture(null));
const quad = new THREE.QuadMesh(new THREE.MeshBasicNodeMaterial({ colorNode: tex }));

let W = 0, H = 0;
function resize() {
  const scale = +(q.get('scale') ?? Math.min(1, Math.sqrt(1.0e6 / (innerWidth * innerHeight * devicePixelRatio ** 2))));
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  const w = Math.max(16, Math.round(innerWidth * devicePixelRatio * scale)), h = Math.max(16, Math.round(innerHeight * devicePixelRatio * scale));
  if (w === W && h === H) return;
  W = w; H = h;
  if (!pt) pt = new PathTracer(device, sg, W, H); else pt.resize(W, H);
  if (!hush) hush = new Hush(device, { width: W, height: H }); else hush.resize(W, H);
  for (const e of ext.values()) e.dispose();
  ext.clear();
}
addEventListener('resize', resize);
resize();

let frame = 0, prevVP = null, last = performance.now(), fps = 0;
const clock = new THREE.Timer();
renderer.setAnimationLoop(() => {
  clock.update();
  const t = clock.getElapsed();
  controls.update();
  camera.updateMatrixWorld();
  duck.rotation.y = t * 0.8;
  duck.position.x = Math.sin(t * 0.6) * 0.8;
  lamp.position.set(Math.cos(t * 0.7) * 1.2, 1.7 + 0.2 * Math.sin(t * 1.1), -0.2 + Math.sin(t * 0.7) * 0.6);
  sg.update();

  const enc = device.createCommandEncoder();
  const vp = pt.writeUniforms({ camera, prevViewProj: prevVP, frame, mode: 0, jitter: JITTER[frame % 16], env: envOf.update(scene) });
  pt.encode(enc, frame & 1);
  const out = hush.denoise(enc, pt.gbuf[frame & 1]);
  device.queue.submit([enc.finish()]);
  prevVP = vp.clone();
  frame++;

  if (!ext.has(out)) ext.set(out, new THREE.ExternalTexture(out));
  tex.value = ext.get(out);
  quad.render(renderer);

  const now = performance.now();
  fps = fps * 0.95 + 0.05 * (1000 / (now - last));
  last = now;
  if (frame % 30 === 0) hud.textContent = `Hush + three.js WebGPURenderer  ${W}x${H}  ${fps.toFixed(0)} fps  (drag to orbit)`;
});
window.__three = { ready: true };
