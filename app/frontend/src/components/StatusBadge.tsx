import { tradingDays } from '../lib/format'
import { STATUS, type Status } from '../lib/status'
import StatusIcon from './StatusIcon'

/** Значок + подпись статуса; у отстающих — на сколько торговых дней. */
export default function StatusBadge({ status, lagDays }: { status: Status; lagDays: number | null }) {
  const behind = (status === 'lagging' || status === 'stale') && lagDays ? tradingDays(lagDays) : null
  return (
    <span className="inline-flex items-start gap-2">
      <StatusIcon status={status} className="mt-0.5" />
      <span className="flex flex-col leading-snug">
        <span className="font-medium">{STATUS[status].label}</span>
        {behind && <span className="text-xs text-muted-foreground">{behind} позади</span>}
      </span>
    </span>
  )
}
