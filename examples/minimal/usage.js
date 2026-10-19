// @ts-check
// Minimal use of the Hush API with any renderer that produces the G-buffer described in src/index.d.ts
// (type-checked against src/index.d.ts by `npm run typecheck`).
import { Hush } from 'hush';

/**
 * @param {GPUDevice} device created with `requiredLimits: Hush.requiredLimits`
 * @param {import('hush').HushInputs} gbuffer this frame's G-buffer (rgba16float textures, see HushInputs)
 * @param {(encoder: GPUCommandEncoder) => void} renderNoisy records the 1 spp path trace that fills `gbuffer`
 * @param {(encoder: GPUCommandEncoder, image: GPUTexture) => void} present tone maps / displays the result
 */
export function setup(device, gbuffer, renderNoisy, present) {
  const hush = new Hush(device, { width: gbuffer.radiance.width, height: gbuffer.radiance.height, exposure: 1 });
  return {
    frame() {
      const encoder = device.createCommandEncoder();
      renderNoisy(encoder);
      const image = hush.denoise(encoder, gbuffer); // rgba16float, linear HDR
      present(encoder, image);
      device.queue.submit([encoder.finish()]);
    },
    resize(/** @type {number} */ w, /** @type {number} */ h) { hush.resize(w, h); },
    cameraCut() { hush.reset(); },
    setExposure(/** @type {number} */ e) { hush.exposure = e; },
    dispose() { hush.destroy(); },
  };
}
