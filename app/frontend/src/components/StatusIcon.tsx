import { cn } from 'cn'
import { STATUS, type Status, type Tone } from '../lib/status'

// Свой знак у каждого статуса: смысл читается и без цвета (дальтонизм, печать).
const GLYPHS: Record<Status, string> = {
  ok: 'M4.6 8.3l2.2 2.2 4.6-4.8',
  lagging: 'M8 4.6V8l2.2 1.5',
  stale: 'M8 4.4v4.4M8 11.2v.4',
  missing: 'M5.6 5.6l4.8 4.8M10.4 5.6l-4.8 4.8',
  error: 'M5.6 5.6l4.8 4.8M10.4 5.6l-4.8 4.8',
  unchecked: 'M5 8h6',
}

const TONES: Record<Tone, string> = {
  good: 'fill-status-good stroke-white',
  warning: 'fill-status-warning stroke-neutral-950', // на жёлтом белый знак не читается
  serious: 'fill-status-serious stroke-white',
  critical: 'fill-status-critical stroke-white',
  neutral: 'fill-status-neutral stroke-white',
}

export default function StatusIcon({ status, className }: { status: Status; className?: string }) {
  return (
    <svg className={cn('size-4 shrink-0', TONES[STATUS[status].tone], className)} viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="8" stroke="none" />
      <path d={GLYPHS[status]} fill="none" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}
