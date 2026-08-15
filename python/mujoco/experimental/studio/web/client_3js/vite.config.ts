import { defineConfig } from 'vite';

// Two dev modes:
//  * `npm run dev`: vite serves the app; /model is served statically from
//    public/model (generate it with a small mujoco Python snippet). No live
//    state; the client falls back to a static pose.
//  * `npm run watch` + MUJOCO_WEB_VIEWER_DIST_3JS=<outDir> on the Python side:
//    the Studio web server serves the built bundle plus the real /model and
//    /state endpoints.
export default defineConfig({
  base: './',
  build: {
    outDir: '../dist_3js',
    emptyOutDir: true,
    target: 'es2022',
    chunkSizeWarningLimit: 6000
  },
  optimizeDeps: {
    // @mujoco/mujoco must stay unbundled so its .wasm asset resolves; Spark
    // must be prebundled or its worker-based splat decode hangs silently in
    // dev (the production build bundles it and works either way).
    exclude: ['@mujoco/mujoco']
  }
});
