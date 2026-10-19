# Hush

Real-time denoising for 1 sample-per-pixel path tracing in the browser (WebGPU / WGSL compute).

**Live demo: [hush.tokyoaivisuals.club](https://hush.tokyoaivisuals.club)** (drag the divider to compare 1 spp, SVGF
and Hush; needs a browser with WebGPU) · [three.js example](https://hush.tokyoaivisuals.club/examples/three/)

![Living room at 1 spp (left) and denoised by Hush (right)](docs/media/hero.jpg)

![The divider sweeps across: 1 spp | Hush](docs/media/compare.webp)

Hush combines SVGF-style temporal accumulation with a tiny per-pixel network that predicts the filter parameters
(temporal blend, edge-stopping strength, detail and coarse-level weights) at half resolution, followed by edge-aware
à-trous filtering and TAA. The network never outputs pixels, so unfamiliar content degrades to a classic
edge-aware filter instead of hallucinating. Inputs and output are `GPUTexture`s on your `GPUDevice`; it does not
depend on any renderer.

Status: research prototype. The built-in weights were trained on the three example scenes below.

| | | |
|---|---|---|
| ![Box](docs/media/box.jpg) | ![Living room](docs/media/living.jpg) | ![Courtyard](docs/media/courtyard.jpg) |

## What's in this repository

- `src/index.js` — the denoiser (`Hush`) with built-in weights, and an `SVGF` baseline with the same inputs
- `src/pt/` (`hush/pathtracer`) — a small WebGPU path tracer for three.js scene graphs that writes the G-buffer Hush
  needs (1 spp radiance, normals, depth, albedo, exact motion vectors)
- `index.html` + `src/demo/` — the comparison demo
- `examples/three/` — Hush in a three.js `WebGPURenderer` app; `examples/minimal/` — the API without a renderer

The training code is not included in this first release.

## Use

```js
import { Hush } from 'hush';

// the device needs a few more storage textures than the WebGPU defaults
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: Hush.requiredLimits });

const hush = new Hush(device, { width, height, exposure: 1 });   // weights are built in

// every frame, after your path tracer wrote this frame's G-buffer:
const encoder = device.createCommandEncoder();
const image = hush.denoise(encoder, { radiance, normalDepth, motion, material, emission });
// ... tone map / display `image` (rgba16float, linear HDR) ...
device.queue.submit([encoder.finish()]);

hush.resize(w, h);     // new size
hush.reset();          // camera cut
hush.exposure = 2;     // display exposure
```

Inputs are `rgba16float` textures of the denoiser's size with `TEXTURE_BINDING` usage:

| texture | contents |
|---|---|
| `radiance` | noisy 1 spp radiance, without the emission of the primary hit |
| `normalDepth` | world-space normal (xyz), linear view depth (w, negative = sky) |
| `motion` | previous − current pixel position (xy, pixels), object id (z, integer < 2048), previous linear depth (w) |
| `material` | albedo (rgb), roughness (a) |
| `emission` | emission of the primary hit |

The output alternates between two textures from frame to frame (`hush.output` is the latest). Types:
[`src/index.d.ts`](src/index.d.ts), [`src/pt/index.d.ts`](src/pt/index.d.ts).

### With three.js

`WebGPURenderer` gives you the device; pass the limits when creating it:

```js
import * as THREE from 'three/webgpu';
const renderer = new THREE.WebGPURenderer({ requiredLimits: { ...Hush.requiredLimits } });
await renderer.init();
const device = renderer.backend.device;
```

[`examples/three/`](examples/three/main.js) path traces an ordinary three.js scene graph (glTF models, an HDRI as
`scene.environment`, a `PointLight`) with the included path tracer (`hush/pathtracer`), denoises it with Hush and draws
the result through three.js (an `ExternalTexture` per output texture on a fullscreen quad, three.js tone mapping).
Drag to orbit.

The path tracer handles meshes with `MeshStandardMaterial` / `MeshPhysicalMaterial` (base colour, roughness / metalness,
normal maps, emissive, smooth glass); moving or skinned objects go under a group with `userData.dynamic = true`.
Lights: `scene.environment` (an equirectangular HDR texture, e.g. from `HDRLoader`; importance sampled, its sun is
extracted into a sharp sun light), `DirectionalLight`, `PointLight` (`userData.radius`) and `HemisphereLight`.

## Run the demos

Requires Node.js 20+ and Python 3 (only for the asset download script).

```
npm install
python tools/fetch_assets.py  # CC0 demo assets (Poly Haven, RobotExpressive) -> public/assets/
npm run dev                   # http://localhost:5180/                 comparison demo
                              # http://localhost:5180/examples/three/  three.js example
npm run build                 # static site in dist/
```

The comparison page shows any two of 1 spp / SVGF / Hush side by side (drag the divider), per-pass GPU times when
the browser exposes timestamp queries, and a render-resolution setting. Requires a browser with WebGPU
(Chrome / Edge 113+, Safari 26+). WebGPU only works in a secure context, so to try it on a phone use the live demo or
serve over https.

## Limitations

- The weights were trained on the three example scenes. On other content Hush still works, but it is not always
  better than SVGF; large areas of low-frequency noise (flat walls, floors lit only by the sky) are the weakest case.
- Like SVGF, it filters what the G-buffer sees: content behind glass or mirrors is blurred.
- The path tracer is a demo renderer: no next-event estimation for emissive meshes (only for sphere lights, the sun
  and the environment map), smooth solid glass only, and shadow rays do not pass through glass.
- Object ids are stored in f16 and must stay below 2048.

## Author

Made by KanaWorks_AI — [GitHub](https://github.com/kanaworksai-hue/) · [X](https://x.com/KanaWorks_AI)

## License

[MIT](LICENSE) © 東京AI映像合同会社. Third-party assets and their licenses: [`THIRD_PARTY.md`](THIRD_PARTY.md).
The KanaWorks_AI artwork in `public/brand/` (shown in the demo scenes) is not covered by the MIT license; see
[`public/brand/README.md`](public/brand/README.md).
