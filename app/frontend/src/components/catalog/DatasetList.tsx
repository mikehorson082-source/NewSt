import { useMemo, type ReactNode } from 'react'
import type { DatasetStatus } from '@/api/schema'
import { TableCell, TableRow } from '@/components/ui/table'
import { formatBytes, formatCount, formatDate, formatDateTime, formatNumber } from '@/lib/format'
import { timeAxis, type Axis } from '@/lib/inventory'
import { CoverageAxis, CoverageBar, CoverageChart } from '../Coverage'
import StatusBadge from '../StatusBadge'
import { DateCell } from './cells'

/** Видимый столбец таблицы инструментов: id и классы его ячеек (ширина, выравнивание, скрытие на узком экране). */
export interface DatasetColumn {
  id: string
  className?: string
}

// Под строкой набора — полоса и подробности на всю ширину таблицы.
// Отступ слева ставит их под названием набора, а не под ручкой ⋮⋮.
const DETAILS =
  'grid gap-x-6 gap-y-2 pr-4 pl-[4.25rem] lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)] lg:items-start'

interface Props {
  datasets: DatasetStatus[]
  expectedDate: string
  columns: DatasetColumn[]
}

/**
 * Наборы данных инструмента: строки той же таблицы, что и инструменты. Даты, строки
 * и размер набора стоят в столбцах шапки; график покрытия — строкой ниже.
 */
export default function DatasetList({ datasets, expectedDate, columns }: Props) {
  const axis = useMemo(() => timeAxis(datasets, expectedDate), [datasets, expectedDate])
  const span = columns.length
  return (
    <>
      {axis && (
        <TableRow className="border-b-0 bg-muted/40 hover:bg-muted/40" aria-hidden="true">
          <TableCell colSpan={span} className="px-0 pt-2 pb-0.5 whitespace-normal">
            <div className={DETAILS}>
              <div className="lg:col-start-2">
                <CoverageAxis axis={axis} expectedDate={expectedDate} />
              </div>
            </div>
          </TableCell>
        </TableRow>
      )}
      {datasets.map((row) => (
        <DatasetRows key={row.id} row={row} axis={axis} expectedDate={expectedDate} columns={columns} />
      ))}
    </>
  )
}

interface RowsProps {
  row: DatasetStatus
  axis: Axis | null
  expectedDate: string
  columns: DatasetColumn[]
}

function DatasetRows({ row, axis, expectedDate, columns }: RowsProps) {
  return (
    <>
      <TableRow className="border-b-0 bg-muted/40 hover:bg-muted/40">
        {columns.map((column) => (
          // Подпись статуса («11 торговых дней позади») переносится, а не раздвигает таблицу.
          <TableCell key={column.id} className={`pt-3 align-top ${column.className ?? ''} ${column.id === 'status' ? 'whitespace-normal' : ''}`}>
            {datasetCell(column.id, row)}
          </TableCell>
        ))}
      </TableRow>
      <TableRow className="bg-muted/40 hover:bg-muted/40">
        <TableCell colSpan={columns.length} className="px-0 pt-0 pb-3 text-sm whitespace-normal">
          <div className={DETAILS}>
            <div className="flex flex-wrap gap-x-3.5 gap-y-0.5 text-muted-foreground tabular-nums lg:row-start-1">
              {/* На узком экране столбцов дат, строк и размера нет — тогда эти числа здесь. */}
              {row.first_date && row.last_date && (
                <span className="text-foreground md:hidden">
                  {formatDate(row.first_date)} → {formatDate(row.last_date)}
                </span>
              )}
              {row.rows !== null && <span className="lg:hidden">{formatCount(row.rows)} строк</span>}
              {row.size_bytes > 0 && <span className="lg:hidden">{formatBytes(row.size_bytes)}</span>}
              {row.facts.map((fact) => (
                <span key={fact.label}>
                  {fact.label}: {typeof fact.value === 'number' ? formatNumber(fact.value) : fact.value}
                </span>
              ))}
              {row.modified_at && <span>изменён {formatDateTime(row.modified_at)}</span>}
              <span className="font-mono text-xs [overflow-wrap:anywhere]">{row.path}</span>
            </div>
            <div className="min-w-0 lg:col-start-2 lg:row-start-1">
              {axis && row.first_date && row.last_date ? (
                row.monthly.length > 0 ? (
                  <CoverageChart
                    first={row.first_date}
                    last={row.last_date}
                    monthly={row.monthly}
                    axis={axis}
                    expectedDate={expectedDate}
                  />
                ) : (
                  // По месяцам не считается (нет колонки даты, сбой чтения) — просто отрезок.
                  <CoverageBar first={row.first_date} last={row.last_date} axis={axis} expectedDate={expectedDate} />
                )
              ) : (
                <span className="text-muted-foreground">без дат</span>
              )}
            </div>
            {/* Правило из src/data/inventory/rules.json: какая норма и почему. */}
            {row.rule && (
              <p className="text-muted-foreground [overflow-wrap:anywhere] lg:col-span-2">{row.rule}</p>
            )}
            {row.note && (
              <p className="rounded-md bg-muted px-3 py-2 [overflow-wrap:anywhere] lg:col-span-2">{row.note}</p>
            )}
          </div>
        </TableCell>
      </TableRow>
    </>
  )
}

/** Ячейка набора под столбцом шапки; у столбцов без смысла для набора — пусто. */
function datasetCell(columnId: string, row: DatasetStatus): ReactNode {
  const dash = <span className="text-muted-foreground">—</span>
  switch (columnId) {
    case 'instrument':
      return (
        <div className="flex min-w-0 flex-col pl-6">
          <span className="font-medium [overflow-wrap:anywhere]">{row.title}</span>
          <span className="text-muted-foreground">{row.description}</span>
        </div>
      )
    case 'status':
      return <StatusBadge status={row.status} lagDays={row.lag_days} />
    case 'firstDate':
      return <DateCell iso={row.first_date ?? undefined} />
    case 'lastDate':
      return <DateCell iso={row.last_date ?? undefined} />
    case 'totalRows':
      return row.rows === null ? dash : formatCount(row.rows)
    case 'sizeBytes':
      return row.size_bytes > 0 ? formatBytes(row.size_bytes) : dash
    default:
      return null
  }
}
