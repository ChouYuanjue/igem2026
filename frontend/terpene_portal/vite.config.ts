import { defineConfig } from 'vite'

export default defineConfig({
  base: '/portal/',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
  },
})
