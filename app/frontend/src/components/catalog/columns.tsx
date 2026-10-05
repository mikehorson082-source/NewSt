import { createColumnHelper, type SortFn } from '@tanstack/react-table'
import { formatBytes, formatCount, formatNumber } from '@/lib/format'
import type { InstrumentRow } from '@/lib/inventory'
import { STATUS_ORDER } from '@/lib/status'
import { DateCell, Health, InstrumentToggle, RowActions } from './cells'
import ColumnHeader from './ColumnHeader'
import type { CatalogFeatures } from './features'
import { DragHandle } from './sortable'

const columnHelper = createColumnHelper<CatalogFeatures, InstrumentRow>()
const collator = new Intl.Collator('ru', { numeric: true })

/** Сравнение значений столбца; пустые (undefined) TanStack сам ставит в конец — см. sortUndefined. */
function compareBy(compare: (a: never, b: never) => number): SortFn<CatalogFeatures, InstrumentRow> {
  return (rowA, rowB, columnId) => compare(rowA.getValue(columnId) as never, rowB.getValue(columnId) as never)
}
const byText = compareBy((a: string, b: string) => collator.compare(a, b))
const byValue = compareBy((a: number | string, b: number | string) => (a < b ? -1 : a > b ? 1 : 0))

export const columns = columnHelper.columns([
  columnHelper.display({
    id: 'drag',
    header: () => <span className="sr-only">Порядок</span>,
    cell: () => <DragHandle />,
    enableHiding: false,
    meta: { label: 'Порядок', className: 'w-9 pr-0 pl-2' },
  }),
  columnHelper.accessor((row) => row.instrument.title, {
    id: 'instrument',
    // Отступ — под шеврон: подпись столбца стоит над названиями, а не над шевронами.
    header: ({ column }) => <ColumnHeader column={column} title="Инструмент" className="pl-6" />,
    cell: ({ row }) => <InstrumentToggle row={row} />,
    sortFn: byText,
    sortDescFirst: false,
    enableHiding: false,
    meta: { label: 'Инструмент', className: 'min-w-36 whitespace-normal' },
  }),
  columnHelper.accessor((row) => STATUS_ORDER.indexOf(row.worst), {
    id: 'status',
    // По возрастанию — сначала худшие: ошибки, нет данных, устарели…
    header: ({ column }) => <ColumnHeader column={column} title="Статус" />,
    cell: ({ row }) => <Health row={row.original} />,
    sortFn: byValue,
    sortDescFirst: false,
    meta: { label: 'Статус' },
  }),
  columnHelper.accessor((row) => row.datasets.length, {
    id: 'datasets',
    header: ({ column }) => <ColumnHeader column={column} title="Наборов" className="justify-end" />,
    cell: ({ getValue }) => formatNumber(getValue()),
    sortFn: byValue,
    sortDescFirst: true,
    meta: { label: 'Наборов', className: 'hidden text-right tabular-nums sm:table-cell' },
  }),
  columnHelper.accessor((row) => row.firstDate ?? undefined, {
    id: 'firstDate',
    header: ({ column }) => <ColumnHeader column={column} title="Данные с" />,
    cell: ({ getValue }) => <DateCell iso={getValue()} />,
    sortFn: byValue,
    sortDescFirst: false,
    sortUndefined: 'last',
    meta: { label: 'Данные с', className: 'hidden tabular-nums md:table-cell' },
  }),
  columnHelper.accessor((row) => row.lastDate ?? undefined, {
    id: 'lastDate',
    header: ({ column }) => <ColumnHeader column={column} title="по" />,
    cell: ({ getValue }) => <DateCell iso={getValue()} />,
    sortFn: byValue,
    sortDescFirst: true,
    sortUndefined: 'last',
    meta: { label: 'Данные по', className: 'hidden tabular-nums md:table-cell' },
  }),
  columnHelper.accessor((row) => row.totalRows ?? undefined, {
    id: 'totalRows',
    header: ({ column }) => <ColumnHeader column={column} title="Строк" className="justify-end" />,
    cell: ({ getValue }) => {
      const value = getValue()
      return value === undefined ? <span className="text-muted-foreground">—</span> : formatCount(value)
    },
    sortFn: byValue,
    sortDescFirst: true,
    sortUndefined: 'last',
    meta: { label: 'Строк', className: 'hidden text-right tabular-nums lg:table-cell' },
  }),
  columnHelper.accessor((row) => row.sizeBytes, {
    id: 'sizeBytes',
    header: ({ column }) => <ColumnHeader column={column} title="Размер" className="justify-end" />,
    cell: ({ getValue }) => formatBytes(getValue()),
    sortFn: byValue,
    sortDescFirst: true,
    meta: { label: 'Размер', className: 'hidden text-right tabular-nums lg:table-cell' },
  }),
  columnHelper.display({
    id: 'actions',
    header: () => <span className="sr-only">Действия</span>,
    cell: ({ row, table }) => (
      <RowActions
        row={row}
        canReorder={table.options.meta?.canReorder ?? false}
        onMove={(edge) => table.options.meta?.moveToEdge(row.id, edge)}
      />
    ),
    enableHiding: false,
    meta: { label: 'Действия', className: 'w-10 pr-2' },
  }),
])
