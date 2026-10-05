/// <reference types="vitest/config" />
import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Разработка: Vite (порт 5173) отдаёт фронтенд с горячей перезагрузкой,
// а запросы /api пересылает в FastAPI (порт 8000).
// Сборка: npm run build → dist/, его отдаёт сам FastAPI на порту 8000.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    // «@/components/ui/button» → src/components/ui/button (так импортирует shadcn).
    alias: { '@': path.resolve(import.meta.dirname, './src') },
  },
  build: {
    // React + Base UI (shadcn) + TanStack Table + dnd-kit ≈ 650 КБ, в gzip ≈ 210 КБ.
    // Для пульта в домашней сети это нормально; предупреждение — если вырастет заметно.
    chunkSizeWarningLimit: 800,
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
  test: {
    include: ['src/**/*.test.ts'],
  },
})
