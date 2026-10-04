import { formatNumber } from '../lib/format'
import { STATUS, STATUS_ORDER, type Status } from '../lib/status'
import StatusIcon from './StatusIcon'

interface Props {
  counts: Record<Status, number>
  total: number
  active: Status | null
  onChange: (status: Status | null) => void
}

/** Плитки со счётчиками по статусам; плитка — фильтр, повторное нажатие его снимает. */
export default function StatusFilter({ counts, total, active, onChange }: Props) {
  const statuses = STATUS_ORDER.filter((status) => counts[status] > 0 || status === active)
  return (
    <div className="tiles" role="group" aria-label="Фильтр по статусу">
      <button type="button" className="tile" aria-pressed={active === null} onClick={() => onChange(null)}>
        <span className="tile__value">{formatNumber(total)}</span>
        <span className="tile__label">Все</span>
      </button>
      {statuses.map((status) => (
        <button
          key={status}
          type="button"
          className="tile"
          aria-pressed={active === status}
          onClick={() => onChange(active === status ? null : status)}
        >
          <span className="tile__value">{formatNumber(counts[status])}</span>
          <span className="tile__label">
            <StatusIcon status={status} />
            {STATUS[status].many}
          </span>
        </button>
      ))}
    </div>
  )
}
