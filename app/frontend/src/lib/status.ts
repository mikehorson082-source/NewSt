import type { DatasetStatus } from '../api/schema'

export type Status = DatasetStatus['status']

/** Тон статуса → цвет значка (палитра статусов в styles.css). */
export type Tone = 'good' | 'warning' | 'serious' | 'critical' | 'neutral'

interface StatusMeta {
  label: string // у строки: «Устарел»
  many: string // у плитки-фильтра: «Устарели»
  tone: Tone
}

// Цвет никогда не несёт смысл один: у каждого статуса свой значок и подпись.
export const STATUS: Record<Status, StatusMeta> = {
  ok: { label: 'Актуален', many: 'Актуальны', tone: 'good' },
  lagging: { label: 'Отстаёт', many: 'Отстают', tone: 'warning' },
  stale: { label: 'Устарел', many: 'Устарели', tone: 'serious' },
  missing: { label: 'Нет данных', many: 'Нет данных', tone: 'critical' },
  error: { label: 'Ошибка', many: 'Ошибки', tone: 'critical' },
  unchecked: { label: 'Без проверки', many: 'Без проверки', tone: 'neutral' },
}

/** Порядок плиток и сводок: сначала то, что требует внимания. */
export const STATUS_ORDER: Status[] = ['error', 'missing', 'stale', 'lagging', 'ok', 'unchecked']

/** Статусы, которые требуют внимания. */
export function isProblem(status: Status): boolean {
  return status === 'error' || status === 'missing' || status === 'stale' || status === 'lagging'
}
