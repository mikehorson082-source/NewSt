/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Разработка: Vite (порт 5173) отдаёт фронтенд с горячей перезагрузкой,
// а запросы /api пересылает в FastAPI (порт 8000).
// Сборка: npm run build → dist/, его отдаёт сам FastAPI на порту 8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
  test: {
    include: ['src/**/*.test.ts'],
  },
})
