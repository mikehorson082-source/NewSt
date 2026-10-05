import { useMemo } from 'react'
import type { DatasetStatus } from '@/api/schema'
import { formatBytes, formatCountNoun, formatDate, formatDateTime, formatNumber } from '@/lib/format'
import { timeAxis, type Axis } from '@/lib/inventory'
import { CoverageAxis, CoverageBar } from '../Coverage'
import StatusBadge from '../StatusBadge'

// Телефон: имя и статус, под ними полоса, ниже подробности.
// Широкий экран: одна строка, полосы всех наборов на одной оси лет.
const GRID =
  'grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-2 px-4 lg:grid-cols-[minmax(10rem,1.1fr)_10rem_minmax(12rem,2fr)_minmax(14rem,1.5fr)] lg:items-start'

interface Props {
  datasets: DatasetStatus[]
  expectedDate: string
}

/** Наборы данных инструмента: раскрываются под его строкой в таблице. */
export default function DatasetList({ datasets, expectedDate }: Props) {
  const axis = useMemo(() => timeAxis(datasets, expectedDate), [datasets, expectedDate])
  return (
    <ul className="divide-y bg-muted/40 text-sm">
      {axis && (
        <li className={`${GRID} pt-2 pb-0.5`} aria-hidden="true">
          <div className="col-span-2 lg:col-span-1 lg:col-start-3">
            <CoverageAxis axis={axis} expectedDate={expectedDate} />
          </div>
        </li>
      )}
      {datasets.map((row) => (
        <DatasetItem key={row.id} row={row} axis={axis} expectedDate={expectedDate} />
      ))}
    </ul>
  )
}

function DatasetItem({ row, axis, expectedDate }: { row: DatasetStatus; axis: Axis | null; expectedDate: string }) {
  return (
    <li className={`${GRID} py-3`}>
      <div className="flex min-w-0 flex-col">
        <span className="font-medium [overflow-wrap:anywhere]">{row.title}</span>
        <span className="text-muted-foreground">{row.description}</span>
      </div>
      <div>
        <StatusBadge status={row.status} lagDays={row.lag_days} />
      </div>
      <div className="col-span-2 min-w-0 lg:col-span-1 lg:pt-0.5">
        {axis && row.first_date && row.last_date ? (
          <CoverageBar first={row.first_date} last={row.last_date} axis={axis} expectedDate={expectedDate} />
        ) : (
          <span className="text-muted-foreground">без дат</span>
        )}
      </div>
      <div className="col-span-2 flex flex-wrap gap-x-3.5 gap-y-0.5 text-muted-foreground tabular-nums lg:col-span-1">
        {row.first_date && row.last_date && (
          <span className="text-foreground">
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
        {row.modified_at && <span>изменён {formatDateTime(row.modified_at)}</span>}
        <span className="font-mono text-xs [overflow-wrap:anywhere]">{row.path}</span>
      </div>
      {row.note && (
        <p className="col-span-2 rounded-md bg-muted px-3 py-2 [overflow-wrap:anywhere] lg:col-span-2 lg:col-start-3">
          {row.note}
        </p>
      )}
    </li>
  )
}
