import { defineConfig } from 'vite';

export default defineConfig({
  root: 'templates',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: 'templates/index.html'
      }
    }
  },
  server: {
    port: 3000
  }
});