
/**
 * G-buffer for one frame. All textures are `rgba16float`, the denoiser's size, with `TEXTURE_BINDING` usage.
 * Layout:
 * - `radiance`: noisy 1 spp radiance (rgb) without the emission of the primary hit
 * - `normalDepth`: world-space shading normal (xyz), linear view depth (w; < 0 = background / sky)
 * - `motion`: xy = previous pixel position − current pixel position (pixels), z = object id (integer < 2048),
 *   w = linear depth of this point in the previous frame's camera
 * - `material`: albedo (rgb; 1 for glass / lights / sky), roughness (a)
 * - `emission`: emission of the primary hit (rgb)
 */
export interface HushInputs {
  radiance: GPUTexture;
  normalDepth: GPUTexture;
  motion: GPUTexture;
  material: GPUTexture;
  emission: GPUTexture;
}

export interface HushOptions {
  width: number;
  height: number;
  /** Display exposure the scene is viewed with (the network sees exposed luminance). Default 1. */
  exposure?: number;
  /** f16 MLP weights in the format of the built-in ones (ArrayBuffer, typed array or base64). Default: built-in weights. */
  weights?: ArrayBuffer | ArrayBufferView | string;
  /** Run the MLP in f16 when the device has 'shader-f16'. Default false. */
  f16?: boolean;
}

/** Optional per-pass GPU timer (see src/gpu.js GpuTimer). */
export type HushTimer =
  | { desc(label: string): Partial<GPUComputePassDescriptor> | undefined }
  | { span: true; spanBegin(label: string): Partial<GPUComputePassDescriptor> | undefined; spanEnd(): Partial<GPUComputePassDescriptor> | undefined };

export declare class Hush {
  /** Limits to request when creating the device, e.g. `new WebGPURenderer({ requiredLimits: Hush.requiredLimits })`. */
  static readonly requiredLimits: Readonly<Record<string, number>>;
  constructor(device: GPUDevice, options: HushOptions);
  readonly device: GPUDevice;
  readonly width: number;
  readonly height: number;
  /** Display exposure; can change at any time. */
  exposure: number;
  /**
   * Records one frame of denoising into `encoder` and returns the output texture (rgba16float, linear HDR,
   * antialiased). Submit the encoder before the next call.
   */
  denoise(encoder: GPUCommandEncoder, inputs: HushInputs, timer?: HushTimer): GPUTexture;
  /** Output of the last `denoise()` (alternates between two textures from frame to frame). */
  readonly output: GPUTexture;
  /** Reallocates for a new size and resets the history. */
  resize(width: number, height: number): void;
  /** Drops the temporal history (camera cut, scene change). */
  reset(): void;
  destroy(): void;
}

/** Hidden width of the MLP for a given number of f16 weights (0 if none fits). */
export declare function hiddenOf(count: number): number;

export interface SVGFOptions {
  iterations?: number;
  sigmaL?: number;
  sigmaN?: number;
  sigmaZ?: number;
  alphaColor?: number;
  alphaMoments?: number;
  taaAlpha?: number;
  taa?: boolean;
}

/** SVGF baseline (Schied et al. 2017) + TAA, same inputs as Hush plus last frame's normalDepth / motion. */
export declare class SVGF {
  constructor(device: GPUDevice, width: number, height: number, options?: SVGFOptions);
  readonly width: number;
  readonly height: number;
  encode(encoder: GPUCommandEncoder, inputs: HushInputs & { prevNormalDepth: GPUTexture; prevMotion: GPUTexture }, timer?: HushTimer): GPUTexture;
  readonly output: GPUTexture;
  resize(width: number, height: number): void;
  reset(): void;
  destroy(): void;
}
