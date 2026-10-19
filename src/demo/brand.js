// KanaWorks_AI artwork placed in the demo scenes (demo only: the training / evaluation scenes in scenes.js stay
// unchanged). Images in public/brand/ (© KanaWorks_AI, see public/brand/README.md).
import * as THREE from 'three';

const loader = new THREE.ImageBitmapLoader();
loader.setOptions({ imageOrientation: 'none', premultiplyAlpha: 'none' });

async function art(file, { width, pos, rotY = 0, alpha = false, rough = 0.6 }) {
  const bmp = await loader.loadAsync(`/brand/${file}`);
  const map = new THREE.Texture(bmp);
  map.colorSpace = THREE.SRGBColorSpace;
  map.needsUpdate = true;
  const height = (width * bmp.height) / bmp.width;
  // alpha: cut-out decal (the wall shows through the transparent parts)
  const mat = new THREE.MeshStandardMaterial({ map, roughness: rough, alphaTest: alpha ? 0.5 : 0 });
  const geo = new THREE.PlaneGeometry(width, height);
  // the path tracer uploads images without flipY (glTF convention), so flip v
  const uv = geo.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setY(i, 1 - uv.getY(i));
  const m = new THREE.Mesh(geo, mat);
  m.position.set(...pos);
  m.rotation.y = rotY;
  return { mesh: m, height };
}

export async function addBrand(name, scene) {
  if (name === 'box') {
    // logo decal on the white back wall (front face z = -2.0), right of the dancing robot
    const { mesh } = await art('kana_logo.png', { width: 1.1, pos: [0.95, 1.6, -1.99], alpha: true, rough: 0.5 });
    scene.add(mesh);
  } else if (name === 'living') {
    // framed portrait above the sofa on the back wall (front face z = -3.0)
    const { mesh, height } = await art('kana_portrait.jpg', { width: 0.72, pos: [0, 1.78, -2.955], rough: 0.75 });
    const frame = new THREE.Mesh(new THREE.BoxGeometry(0.72 + 0.1, height + 0.1, 0.04),
      new THREE.MeshStandardMaterial({ color: 0x2a1c12, roughness: 0.45 }));
    frame.position.set(0, 1.78, -2.98);
    scene.add(frame, mesh);
  } else if (name === 'courtyard') {
    // character sheet as a poster on the back brick wall (front face z = -6.0)
    const { mesh } = await art('kana_turnaround.jpg', { width: 2.4, pos: [1.6, 1.6, -5.99], rough: 0.55 });
    scene.add(mesh);
  }
}
