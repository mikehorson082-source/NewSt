import type { DatasetStatus } from '../api/schema'
import { formatBytes, formatCountNoun, formatDate, formatDateTime, formatNumber } from '../lib/format'
import type { Axis } from '../lib/inventory'
import { CoverageBar } from './Coverage'
import StatusBadge from './StatusBadge'

interface Props {
  row: DatasetStatus
  axis: Axis | null
  expectedDate: string
}

/** Набор данных инструмента: на телефоне — карточка, на широком экране — строка таблицы. */
export default function DatasetRow({ row, axis, expectedDate }: Props) {
  return (
    <li className={row.note ? 'row row--note' : 'row'}>
      <div className="row__title">
        <span className="row__name">{row.title}</span>
        <span className="row__desc">{row.description}</span>
      </div>
      <div className="row__status">
        <StatusBadge status={row.status} lagDays={row.lag_days} />
      </div>
      <div className="row__bar">
        {axis && row.first_date && row.last_date ? (
          <CoverageBar first={row.first_date} last={row.last_date} axis={axis} expectedDate={expectedDate} />
        ) : (
          <span className="row__nodates">без дат</span>
        )}
      </div>
      <div className="row__meta">
        {row.first_date && row.last_date && (
          <span className="row__dates">
            {formatDate(row.first_date)} → {formatDate(row.last_date)}
          </span>
        )}
        {row.rows !== null && <span>{formatCountNoun(row.rows, ['строка', 'строки', 'строк'])}</span>}
        {row.size_bytes > 0 && <span>{formatBytes(row.size_bytes)}</span>}
        {row.facts.map((fact) => (
          <span key={fact.label}>
            {fact.label}: {typeof fact.value === 'number' ? formatNumber(fact.value) : fact.value}
          </span>
        ))}
        {row.modified_at && <span className="row__muted">изменён {formatDateTime(row.modified_at)}</span>}
        <span className="row__path">{row.path}</span>
      </div>
      {row.note && <p className="row__note">{row.note}</p>}
    </li>
  )
}
