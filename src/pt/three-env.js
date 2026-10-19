// Path tracer environment (sky + lights, see pathtracer.js writeUniforms) from the lights of a three.js scene:
//   scene.environment -> environment map (equirect DataTexture, e.g. from HDRLoader; scene.environmentRotation.y and
//                        scene.environmentIntensity apply). Its sun is cut out into a sun light (envmap.js) unless
//                        environment.userData.sun === false.
//   DirectionalLight -> sun (intensity = irradiance, userData.angle = angular diameter in degrees, default 0.53)
//   PointLight       -> sphere light (userData.radius, default 0.05; radiance = intensity / (pi r^2))
//   HemisphereLight  -> analytic sky (color = zenith, userData.horizon = horizon colour, groundColor, intensity);
//                       ignored when there is an environment map
// Other lights are ignored. Keep one instance per scene: it remembers last frame's light positions (motion vectors
// of moving sphere lights) and the prepared environment map.
import { DataUtils } from 'three';
import { prepareEnvMap, envSunLight } from './envmap.js';

export class ThreeEnvironment {
  constructor() {
    this.prev = new Map();
    this.mapKey = null;
    this.map = null;
  }

  envMap(tex) {
    const key = tex ? `${tex.uuid}:${tex.version}` : null;
    if (key === this.mapKey) return this.map;
    this.mapKey = key;
    this.map = null;
    const img = tex?.image;
    if (img?.data && img.width && img.height) {
      let data = img.data;
      if (data instanceof Uint16Array) data = Float32Array.from(data, (h) => DataUtils.fromHalfFloat(h));  // HalfFloatType
      const ch = data.length / (img.width * img.height);
      if (ch === 3) data = Float32Array.from({ length: img.width * img.height * 4 }, (_, i) => ((i & 3) === 3 ? 1 : data[(i >> 2) * 3 + (i & 3)]));
      this.map = prepareEnvMap({ data, width: img.width, height: img.height }, { sun: tex.userData.sun !== false });
    } else if (tex) {
      console.warn('ThreeEnvironment: scene.environment must be an equirect DataTexture (e.g. HDRLoader); ignored');
    }
    return this.map;
  }

  update(scene) {
    const env = { skyZenith: [0, 0, 0], skyHorizon: [0, 0, 0], skyIntensity: 0, ground: [0, 0, 0], lights: [] };
    const map = this.envMap(scene.environment);
    if (map) {
      env.map = map;
      env.mapRotation = scene.environmentRotation?.y ?? 0;
      env.mapIntensity = scene.environmentIntensity ?? 1;
      const sun = envSunLight(map, env.mapRotation, { scale: env.mapIntensity });
      if (sun) env.lights.push(sun);
    }
    const seen = new Set();
    scene.traverseVisible((o) => {
      if (!o.isLight) return;
      const c = [o.color.r, o.color.g, o.color.b];
      if (o.isHemisphereLight) {
        if (map) return;
        const h = o.userData.horizon ?? o.color;
        env.skyZenith = c;
        env.skyHorizon = [h.r, h.g, h.b];
        env.ground = [o.groundColor.r, o.groundColor.g, o.groundColor.b];
        env.skyIntensity = o.intensity;
      } else if (o.isDirectionalLight) {
        const p = o.getWorldPosition(o.position.clone());
        const t = o.target.getWorldPosition(o.position.clone());
        const d = p.sub(t).normalize();
        const cosMax = Math.cos(((o.userData.angle ?? 0.53) / 2) * (Math.PI / 180));
        const omega = 2 * Math.PI * (1 - cosMax);
        env.lights.push({ kind: 0, dir: [d.x, d.y, d.z], cosMax, radiance: c.map((x) => (x * o.intensity) / omega) });
      } else if (o.isPointLight) {
        const p = o.getWorldPosition(o.position.clone()).toArray();
        const r = o.userData.radius ?? 0.05;
        env.lights.push({ kind: 1, pos: p, prevPos: this.prev.get(o.uuid) ?? p, radius: r, radiance: c.map((x) => (x * o.intensity) / (Math.PI * r * r)) });
        this.prev.set(o.uuid, p);
        seen.add(o.uuid);
      }
    });
    for (const k of this.prev.keys()) if (!seen.has(k)) this.prev.delete(k);
    return env;
  }
}
