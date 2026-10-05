import type { Row } from '@tanstack/react-table'
import { cn } from 'cn'
import { ArrowDownToLine, ArrowUpToLine, ChevronRight, ChevronsDownUp, ChevronsUpDown, MoreHorizontal } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { formatDate } from '@/lib/format'
import { countByStatus, type InstrumentRow } from '@/lib/inventory'
import { STATUS, STATUS_ORDER, isProblem } from '@/lib/status'
import StatusIcon from '../StatusIcon'
import type { CatalogFeatures } from './features'

// Ячейки таблицы инструментов; какие где стоят — в columns.tsx.

/** Название инструмента с шевроном: нажатие раскрывает его наборы. На телефоне — крупная цель для пальца. */
export function InstrumentToggle({ row }: { row: Row<CatalogFeatures, InstrumentRow> }) {
  const open = row.getIsExpanded()
  const { title, subtitle } = row.original.instrument
  return (
    <button
      type="button"
      onClick={row.getToggleExpandedHandler()}
      aria-expanded={open}
      title={open ? 'Свернуть наборы' : 'Показать наборы'}
      className="group/toggle -my-1 flex w-full items-start gap-2 rounded-md py-1 text-left outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
    >
      <ChevronRight
        className={cn('mt-0.5 size-4 shrink-0 text-muted-foreground transition-transform', open && 'rotate-90')}
        aria-hidden="true"
      />
      <span className="flex flex-col">
        <span className="font-medium underline-offset-4 group-hover/toggle:underline">{title}</span>
        <span className="text-xs text-muted-foreground">{subtitle}</span>
      </span>
    </button>
  )
}

/** Сводка статусов наборов: значок и число у каждого встретившегося статуса. */
export function Health({ row }: { row: InstrumentRow }) {
  const counts = countByStatus(row.datasets)
  return (
    <div className="flex gap-2.5 tabular-nums">
      {STATUS_ORDER.filter((status) => counts[status] > 0).map((status) => (
        <span
          key={status}
          className={cn('inline-flex items-center gap-1', isProblem(status) ? 'font-semibold' : 'text-muted-foreground')}
          title={STATUS[status].many}
        >
          <StatusIcon status={status} />
          {counts[status]}
          <span className="sr-only"> {STATUS[status].many.toLowerCase()}</span>
        </span>
      ))}
    </div>
  )
}

export function DateCell({ iso }: { iso: string | undefined }) {
  return iso ? formatDate(iso) : <span className="text-muted-foreground">—</span>
}

interface ActionsProps {
  row: Row<CatalogFeatures, InstrumentRow>
  canReorder: boolean
  onMove: (edge: 'top' | 'bottom') => void
}

/** Меню строки: в начало или конец раздела — перетаскиванием между страницами так не сделать. */
export function RowActions({ row, canReorder, onMove }: ActionsProps) {
  const open = row.getIsExpanded()
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={<Button variant="ghost" size="icon-sm" aria-label={`Действия: ${row.original.instrument.title}`} />}
      >
        <MoreHorizontal />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-auto">
        <DropdownMenuItem disabled={!canReorder} onClick={() => onMove('top')}>
          <ArrowUpToLine />
          Переместить в начало
        </DropdownMenuItem>
        <DropdownMenuItem disabled={!canReorder} onClick={() => onMove('bottom')}>
          <ArrowDownToLine />
          Переместить в конец
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => row.toggleExpanded()}>
          {open ? <ChevronsDownUp /> : <ChevronsUpDown />}
          {open ? 'Свернуть наборы' : 'Показать наборы'}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
