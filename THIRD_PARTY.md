# Third-party content

Hush's own code and weights are under the [MIT license](LICENSE). This repository also uses the following.

## Downloaded by `tools/fetch_assets.py` (not included in the repository)

The demo scenes and the three.js example load models, textures and HDRIs that `tools/fetch_assets.py` downloads into
`public/assets/`:

| content | author / source | license |
|---|---|---|
| Models, textures and HDRIs (sofa, chairs, plants, lamps, bust, crates, floors, walls, `hilly_terrain_01`, `spaichingen_hill`, …; the full list is in `tools/fetch_assets.py`) | [Poly Haven](https://polyhaven.com) | [CC0 1.0](https://polyhaven.com/license) |
| `RobotExpressive.glb` | Tomás Laulhé, from the [three.js examples](https://github.com/mrdoob/three.js/tree/dev/examples/models/gltf/RobotExpressive) | CC0 1.0 |

## npm dependencies

| package | license |
|---|---|
| [three.js](https://github.com/mrdoob/three.js) (`three`, used by the path tracer's scene loading, the demo and the three.js example) | MIT |
| [Vite](https://github.com/vitejs/vite), [TypeScript](https://github.com/microsoft/TypeScript), `@types/three` (development only) | MIT / Apache-2.0 / MIT |

## KanaWorks_AI artwork

`public/brand/`, `docs/brand/` and the banners in `docs/assets/` (the KanaWorks_AI logo and the character Kana) are © 東京AI映像合同会社 and **not** covered by
the MIT license; see [`docs/brand/README.md`](docs/brand/README.md). The demo runs without them with `?brand=0`.
