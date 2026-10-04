import { tradingDays } from '../lib/format'
import { STATUS, type Status } from '../lib/status'
import StatusIcon from './StatusIcon'

/** Значок + подпись статуса; у отстающих — на сколько торговых дней. */
export default function StatusBadge({ status, lagDays }: { status: Status; lagDays: number | null }) {
  const behind = (status === 'lagging' || status === 'stale') && lagDays ? tradingDays(lagDays) : null
  return (
    <span className="status">
      <StatusIcon status={status} />
      <span className="status__text">
        <span className="status__label">{STATUS[status].label}</span>
        {behind && <span className="status__detail">{behind} позади</span>}
      </span>
    </span>
  )
}
