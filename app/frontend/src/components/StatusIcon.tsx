import { STATUS, type Status } from '../lib/status'

// Свой знак у каждого статуса: смысл читается и без цвета (дальтонизм, печать).
const GLYPHS: Record<Status, string> = {
  ok: 'M4.6 8.3l2.2 2.2 4.6-4.8',
  lagging: 'M8 4.6V8l2.2 1.5',
  stale: 'M8 4.4v4.4M8 11.2v.4',
  missing: 'M5.6 5.6l4.8 4.8M10.4 5.6l-4.8 4.8',
  error: 'M5.6 5.6l4.8 4.8M10.4 5.6l-4.8 4.8',
  unchecked: 'M5 8h6',
}

export default function StatusIcon({ status }: { status: Status }) {
  return (
    <svg className={`status-icon status-icon--${STATUS[status].tone}`} viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="8" />
      <path d={GLYPHS[status]} />
    </svg>
  )
}
