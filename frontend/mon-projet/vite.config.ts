import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  server: {
    port: 5173,
    host: true, // Ceci est l'équivalent du --host
  },
  plugins: [react()],
})
