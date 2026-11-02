import {defineConfig} from 'vite';
import {svelte} from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
    plugins: [svelte()],
    // Keep whitespace string literals escaped in generated assets.
    esbuild: {supported: {'template-literal': false}},
    build: {
        outDir: '../src/slipagent/web',
        emptyOutDir: true,
        license: {fileName: 'third-party-notices.md'},
        rollupOptions: {output: {entryFileNames: 'app.js', assetFileNames: 'app[extname]'}},
    },
});
