import { defineConfig } from 'vite';
import path from 'node:path';

export default defineConfig({
  // the examples import the library by its package name; one three.js instance (three/webgpu re-exports the core)
  resolve: {
    alias: [
      { find: /^three$/, replacement: 'three/webgpu' },
      { find: /^hush\/pathtracer$/, replacement: path.resolve('src/pt/index.js') },
      { find: /^hush$/, replacement: path.resolve('src/index.js') },
    ],
  },
  server: { port: 5180 },
  build: {
    target: 'es2022',
    rollupOptions: { input: { demo: 'index.html', three: 'examples/three/index.html' } },
  },
});
