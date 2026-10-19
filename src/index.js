// Hush: real-time denoiser for 1 spp path tracing on WebGPU. Inputs and output are GPUTextures on your GPUDevice;
// no dependency on three.js (get the device from WebGPURenderer: renderer.backend.device).
export { Hush, hiddenOf } from './denoise/hush.js';
export { SVGF } from './denoise/svgf.js';
