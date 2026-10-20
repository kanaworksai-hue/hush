<div align="center">

<img src="docs/assets/banner-zh.jpg" alt="Hush：在浏览器里实时降噪 1 spp 路径追踪" width="820">

<p>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-e0a33a?style=flat-square"></a>
  <img alt="WebGPU / WGSL" src="https://img.shields.io/badge/WebGPU-WGSL_compute-005a9c?style=flat-square">
  <a href="https://threejs.org"><img alt="three.js WebGPURenderer" src="https://img.shields.io/badge/three.js-WebGPURenderer-111?style=flat-square&logo=threedotjs&logoColor=white"></a>
  <img alt="Denoiser has no dependencies" src="https://img.shields.io/badge/denoiser_deps-none-2f9e6e?style=flat-square">
  <a href="https://hush.tokyoaivisuals.club"><img alt="Live demo" src="https://img.shields.io/badge/live_demo-online-f2c879?style=flat-square&logo=googlechrome&logoColor=white"></a>
  <a href="https://github.com/kanaworksai-hue/hush/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/kanaworksai-hue/hush?style=flat-square&logo=github&label=stars&color=f5c518"></a>
</p>

<p><a href="https://github.com/kanaworksai-hue"><img src="docs/brand/kanaworks-mark.png" height="22" alt="" align="absmiddle"></a>&nbsp;<a href="https://github.com/kanaworksai-hue"><b>KanaWorks_AI 出品</b></a></p>

[English](README.md) | **简体中文** | [日本語](README.ja.md)

[🖥️ 在线 demo](https://hush.tokyoaivisuals.club) · [three.js 示例](https://hush.tokyoaivisuals.club/examples/three/) · [快速开始](#-快速开始) · [API](#-api) · [限制](#%EF%B8%8F-限制) · [⭐ Star](#-喜欢的话请点-star)

</div>

**Hush** 在浏览器里实时把每像素 1 个采样（1 spp）的路径追踪噪声图变成干净、帧间稳定的画面。它是几个跑在你自己 `GPUDevice` 上的 WebGPU compute pass：把噪声图和 G-buffer（法线、深度、albedo、运动向量）以 `GPUTexture` 传进去，拿回降噪后的纹理。three.js `WebGPURenderer` 和其他 WebGPU 渲染器都能用。

<p align="center">
  <img src="docs/media/hero.jpg" alt="客厅：左边 1 spp，右边 Hush 降噪后" width="820">
  <br><sub>同一帧：左边是每像素 1 个采样，右边是 Hush 处理后。</sub>
</p>

## ✨ 特点

<table>
<tr>
<td align="center" width="25%"><b>⚡ 实时</b><br><sub>每帧几个 compute pass；网络在半分辨率运行，只负责调节滤波器</sub></td>
<td align="center" width="25%"><b>🎞️ 稳定</b><br><sub>用精确运动向量做时域累积，再加 TAA，画面不会蠕动或闪烁</sub></td>
<td align="center" width="25%"><b>🧩 不绑定渲染器</b><br><sub>输入输出都是普通的 <code>GPUTexture</code>；附带 three.js 示例</sub></td>
<td align="center" width="25%"><b>📦 开箱即用</b><br><sub>权重已内置，另附一个给 three.js 场景用的小型路径追踪器和 SVGF 基线</sub></td>
</tr>
</table>

<p align="center">
  <img src="docs/media/compare.webp" alt="分割线扫过画面：1 spp | Hush" width="820">
</p>

## 🧭 原理

Hush 把 SVGF 式的时域累积和一个很小的逐像素网络结合起来：网络在半分辨率预测**滤波参数**（时域混合、边缘保持强度、细节层和粗层的权重），之后做边缘感知的 à-trous 滤波和 TAA。网络从不直接输出像素，所以遇到没见过的内容时，它会退回到像传统边缘感知滤波器那样工作，而不是凭空编造细节。

| | | |
|---|---|---|
| ![Box](docs/media/box.jpg) | ![客厅](docs/media/living.jpg) | ![庭院](docs/media/courtyard.jpg) |

状态：研究原型。内置权重是在这三个示例场景上训练的。

## 🚀 快速开始

最快的方式是打开 **[在线 demo](https://hush.tokyoaivisuals.club)**：拖动分割线比较 1 spp、SVGF 和 Hush。需要支持 WebGPU 的浏览器（Chrome / Edge 113+、Safari 26+）。

在本地运行 demo（需要 Node.js 20.19+，下载素材需要 Python 3）：

```bash
git clone https://github.com/kanaworksai-hue/hush.git
cd hush
npm install
python tools/fetch_assets.py   # CC0 demo 素材（Poly Haven、RobotExpressive）-> public/assets/
npm run dev                    # http://localhost:5180/                 对比 demo
                               # http://localhost:5180/examples/three/  three.js 示例
```

`npm run build` 会在 `dist/` 生成静态网站。WebGPU 只能在安全上下文（https 或 localhost）里使用，想在手机上试本地构建的话要用 https 提供（或者直接用在线 demo）。

## 🧰 API

Hush 还没有发布到 npm（npm 上的 `hush` 是另一个无关的包），请从 GitHub 安装，导入名仍然是 `hush`：

```bash
npm install github:kanaworksai-hue/hush
```

```js
import { Hush } from 'hush';

// 设备需要比 WebGPU 默认值多几个 storage texture
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: Hush.requiredLimits });

const hush = new Hush(device, { width, height, exposure: 1 });   // 权重已内置

// 每帧，在路径追踪器写好这一帧的 G-buffer 之后：
const encoder = device.createCommandEncoder();
const image = hush.denoise(encoder, { radiance, normalDepth, motion, material, emission });
// ... 对 `image`（rgba16float，线性 HDR）做色调映射 / 显示 ...
device.queue.submit([encoder.finish()]);

hush.resize(w, h);     // 改尺寸
hush.reset();          // 镜头切换
hush.exposure = 2;     // 显示曝光
```

输入是和降噪器同尺寸、带 `TEXTURE_BINDING` 用途的 `rgba16float` 纹理：

| 纹理 | 内容 |
|---|---|
| `radiance` | 1 spp 的含噪辐射亮度，不含首次命中点的自发光 |
| `normalDepth` | 世界空间法线（xyz），线性视深（w，负数 = 天空） |
| `motion` | 上一帧 − 当前像素位置（xy，像素），物体 id（z，整数 < 2048），上一帧线性深度（w） |
| `material` | albedo（rgb），粗糙度（a） |
| `emission` | 首次命中点的自发光 |

输出每帧在两张纹理之间交替（`hush.output` 是最新的）。同样输入的 SVGF 基线导出为 `SVGF`。类型定义：[`src/index.d.ts`](src/index.d.ts)、[`src/pt/index.d.ts`](src/pt/index.d.ts)。

### 配合 three.js

`WebGPURenderer` 提供设备；创建时传入 limits：

```js
import * as THREE from 'three/webgpu';
const renderer = new THREE.WebGPURenderer({ requiredLimits: { ...Hush.requiredLimits } });
await renderer.init();
const device = renderer.backend.device;
```

[`examples/three/`](examples/three/main.js) 用附带的路径追踪器（`hush/pathtracer`）追踪一个普通的 three.js 场景图（glTF 模型、作为 `scene.environment` 的 HDRI、一个 `PointLight`），用 Hush 降噪，再通过 three.js 画出来（每张输出纹理一个 `ExternalTexture`，贴在全屏四边形上，由 three.js 做色调映射）。拖动可以环绕视角。

路径追踪器支持 `MeshStandardMaterial` / `MeshPhysicalMaterial` 的网格（基础色、粗糙度 / 金属度、法线贴图、自发光、光滑玻璃）；会动的物体或蒙皮物体放在 `userData.dynamic = true` 的组下面。光源：`scene.environment`（等距柱状投影 HDR 纹理，例如 `HDRLoader` 加载的；做重要性采样，其中的太阳会被提取成锐利的太阳光）、`DirectionalLight`、`PointLight`（`userData.radius`）和 `HemisphereLight`。

## 📁 仓库内容

```
src/index.js        Hush（内置权重）和 SVGF 基线
src/denoise/        WGSL compute pass 和权重
src/pt/             hush/pathtracer：给 three.js 场景图用的小型 WebGPU 路径追踪器
index.html          对比 demo（src/demo/）
examples/three/     在 three.js WebGPURenderer 应用里使用 Hush
examples/minimal/   不依赖渲染器的 API 用法
```

第一版不包含训练代码。

## ⚠️ 限制

- 权重是在三个示例场景上训练的。用在其他内容上也能工作，但不一定总比 SVGF 好；大面积的低频噪声（平坦墙面、只被天光照亮的地面）是最弱的情况。
- 和 SVGF 一样，它按 G-buffer 看到的表面滤波：玻璃或镜子后面的内容会被糊掉。
- 附带的路径追踪器是 demo 用的：自发光网格没有直接光采样（只有球光、太阳和环境贴图有），只有光滑实心玻璃，阴影光线不穿过玻璃。
- 物体 id 存在 f16 里，必须小于 2048。

## ⭐ 喜欢的话请点 Star

如果 Hush 对你有用，请在[仓库页面](https://github.com/kanaworksai-hue/hush)顶部点 **⭐ Star**。这能让更多人发现这个项目，也让我们知道值得继续做下去。

欢迎在 [issues](https://github.com/kanaworksai-hue/hush/issues) 里报告 bug 或提想法。用它做了东西？发到 X 上并标记 [@KanaWorks_AI](https://x.com/KanaWorks_AI)。

## 📄 许可与致谢

- **本项目：** [MIT](LICENSE) © 東京AI映像合同会社。
- **Demo 素材：** 模型、贴图和 HDRI 来自 [Poly Haven](https://polyhaven.com)（CC0），RobotExpressive 作者 Tomás Laulhé（CC0），由 `tools/fetch_assets.py` 下载，不包含在仓库里。详见 [`THIRD_PARTY.md`](THIRD_PARTY.md)。
- **品牌：** KanaWorks_AI 名称、logo 和 Kana 形象（`docs/brand/`、`public/brand/`、横幅和 demo 场景里）版权归東京AI映像合同会社所有，**不在** MIT 授权范围内，见 [docs/brand](docs/brand/README.md)。demo 加 `?brand=0` 可以不显示。
- **基于：** [WebGPU](https://www.w3.org/TR/webgpu/) 和 [three.js](https://threejs.org)。降噪器参照了 SVGF（Schied et al. 2017）和学习预测滤波参数方面的公开研究。

<br>

<div align="center">

<a href="https://x.com/KanaWorks_AI"><img src="docs/brand/kana-wave.png" width="150" alt="KanaWorks_AI 吉祥物 Kana 在挥手"></a>

<p><b>由 <a href="https://x.com/KanaWorks_AI">KanaWorks_AI</a> 用 ♥ 制作</b></p>

<p><sub>如果 Hush 帮你省了时间，点个 ⭐ 能让更多人发现它。更多项目：<a href="https://github.com/kanaworksai-hue">github.com/kanaworksai-hue</a></sub></p>

</div>
