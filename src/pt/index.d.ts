import type { Object3D, PerspectiveCamera, Matrix4 } from 'three';

/** Sun (kind 0: direction towards the sun, cone half-angle cosine) or sphere light (kind 1). Radiance in linear rgb. */
export type Light =
  | { kind: 0; dir: [number, number, number]; cosMax: number; radiance: [number, number, number] }
  | { kind: 1; pos: [number, number, number]; prevPos?: [number, number, number]; radius: number; radiance: [number, number, number] };

export interface Environment {
  skyZenith: [number, number, number];
  skyHorizon: [number, number, number];
  skyIntensity?: number;
  ground: [number, number, number];
  lights: Light[];
  /** Equirect environment map (replaces the analytic sky; importance sampled). */
  map?: EnvMap | null;
  mapIntensity?: number;
  /** Rotation of the map about +y, radians. */
  mapRotation?: number;
  /** Firefly clamp on indirect light (luminance per path contribution); overrides writeUniforms' clampLum (16). */
  clampLum?: number;
  /** Firefly clamp on direct light at the primary hit (glints on small glossy parts); unset = off. */
  clampDirect?: number;
}

/** Environment map prepared for the path tracer (the sun is cut out into `sun`, see envSunLight). */
export interface EnvMap {
  readonly width: number;
  readonly height: number;
  readonly half: Uint16Array;
  readonly table: Float32Array;
  readonly sun: { dir: [number, number, number]; irradiance: [number, number, number] } | null;
}

/** Equirect RGBA float data, row 0 = up (three.js equirect convention). */
export declare function prepareEnvMap(
  image: { data: Float32Array; width: number; height: number }, options?: { sun?: boolean; sunConeDeg?: number },
): EnvMap;
/** Loads a Radiance .hdr and prepares it. */
export declare function loadEnvMap(url: string, options?: { sun?: boolean; sunConeDeg?: number }): Promise<EnvMap>;
/** The sun cut out of the map as a light, rotated like the map; null if the map has no distinct sun. */
export declare function envSunLight(
  map: EnvMap, rotation?: number, options?: { angularDeg?: number; scale?: number; tint?: [number, number, number] },
): Light | null;
export declare const ENV_GRID: [number, number];

/**
 * A three.js scene baked into path tracer buffers. Meshes under an object with `userData.dynamic = true` get their own
 * BVH that is re-skinned / re-transformed and refit by `update()` every frame (exact motion vectors).
 * Meshes under an object with `userData.noShadow = true` (e.g. a lamp fixture around a sphere light) get their own BVH
 * that shadow rays skip. `userData.skipPT = true` leaves a mesh out entirely.
 */
export declare class SceneGPU {
  constructor(device: GPUDevice);
  build(scene: Object3D): Promise<void>;
  destroy(): void;
  /** Re-transforms and refits the dynamic groups (call once per frame after animating). */
  update(): boolean;
  readonly triTotal: number;
}

export interface GBuffer {
  radiance: GPUTexture;
  normalDepth: GPUTexture;
  motion: GPUTexture;
  material: GPUTexture;
  emission: GPUTexture;
}

export declare class PathTracer {
  constructor(device: GPUDevice, scene: SceneGPU, width: number, height: number);
  readonly width: number;
  readonly height: number;
  /** Two G-buffers; frame N writes gbuf[N & 1]. */
  readonly gbuf: [GBuffer, GBuffer];
  resize(width: number, height: number): void;
  /** Returns this frame's view-projection matrix (pass it as prevViewProj next frame). mode 0 = 1 spp + G-buffer. */
  writeUniforms(o: {
    camera: PerspectiveCamera; prevViewProj: Matrix4 | null; frame: number; mode: 0 | 1; jitter: [number, number];
    spp?: number; sampleOffset?: number; maxBounces?: number; clampLum?: number; shadowTau?: number; env: Environment;
  }): Matrix4;
  encode(encoder: GPUCommandEncoder, slot: 0 | 1, timer?: { desc(label: string): object | undefined } | null, label?: string): void;
  destroy(): void;
}

/** 16-phase Halton(2,3) subpixel jitter in pixels, [-0.5, 0.5]. */
export declare const JITTER: [number, number][];

/**
 * Environment from a three.js scene's lights: DirectionalLight -> sun, PointLight (userData.radius) -> sphere light,
 * HemisphereLight -> sky. Call update() once per frame (it tracks light motion).
 */
export declare class ThreeEnvironment {
  update(scene: Object3D): Environment;
}
