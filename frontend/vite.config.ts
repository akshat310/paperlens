import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // Proxy /api to the FastAPI backend during development.
    //
    // Why a proxy instead of calling http://localhost:8000 directly: the browser
    // then sees every request as same-origin, so CORS never enters the picture
    // in dev, and no API base URL has to be configured. In production both are
    // served from one origin anyway -- FastAPI serves the built bundle -- so
    // this mirrors the real setup rather than papering over a difference.
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
