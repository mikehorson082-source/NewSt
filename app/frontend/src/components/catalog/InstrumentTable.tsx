import {
  closestCenter,
  DndContext,
  KeyboardSensor,
  MouseSensor,
  pointerWithin,
  TouchSensor,
  useSensor,
  useSensors,
  type CollisionDetection,
  type DragEndEvent,
} from '@dnd-kit/core'
import { restrictToVerticalAxis } from '@dnd-kit/modifiers'
import { SortableContext, sortableKeyboardCoordinates, verticalListSortingStrategy } from '@dnd-kit/sortable'
import {
  useTable,
  type ColumnVisibilityState,
  type ExpandedState,
  type OnChangeFn,
  type PaginationState,
  type SortingState,
} from '@tanstack/react-table'
import { GripVertical, Undo2 } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Table, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import type { InstrumentRow } from '@/lib/inventory'
import { moveToEdge, moveVisible } from '@/lib/order'
import { columns } from './columns'
import DatasetList from './DatasetList'
import { features } from './features'
import Pagination from './Pagination'
import { SortableRow } from './sortable'
import ViewOptions from './ViewOptions'

const DEFAULT_PAGE_SIZE = 50

// Куда бросили — строка под указателем. «Ближайший центр» ошибается с раскрытой строкой:
// она вместе с наборами выше экрана, и её центр остаётся ближе всего к ней самой.
// С клавиатуры указателя нет — тогда ближайший центр.
const collisionDetection: CollisionDetection = (args) => {
  const hits = pointerWithin(args)
  return hits.length > 0 ? hits : closestCenter(args)
}

// Подсказки экранному диктору при перетаскивании — по-русски.
const accessibility = {
  screenReaderInstructions: {
    draggable: 'Чтобы взять строку, нажмите пробел. Стрелки вверх и вниз двигают её, пробел — отпустить, Escape — отменить.',
  },
  announcements: {
    onDragStart: () => 'Строка взята.',
    onDragOver: () => undefined,
    onDragEnd: ({ over }: { over: unknown }) => (over ? 'Строка перемещена.' : 'Строка возвращена на место.'),
    onDragCancel: () => 'Перемещение отменено.',
  },
}

interface Props {
  rows: InstrumentRow[] // инструменты раздела под фильтром, в своём порядке
  order: string[] | null // свой порядок всего раздела; null — не загружен, переставлять нельзя
  onReorder: (ids: string[]) => void
  expectedDate: string
  filterKey: string // поменялся фильтр — страница на первую, раскрытие заново
  filtering: boolean
  columnVisibility: ColumnVisibilityState
  onColumnVisibilityChange: OnChangeFn<ColumnVisibilityState>
}

/** Таблица инструментов раздела: сортировка по столбцам, свой порядок перетаскиванием, наборы под строкой. */
export default function InstrumentTable(props: Props) {
  const { rows, order, onReorder, expectedDate, filterKey, filtering } = props
  const [sorting, setSorting] = useState<SortingState>([])
  // При фильтре раскрыты все: видно, какие именно наборы нашлись.
  const [expanded, setExpanded] = useState<ExpandedState>(filtering ? true : {})
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: DEFAULT_PAGE_SIZE })
  const [seenFilter, setSeenFilter] = useState(filterKey)
  if (seenFilter !== filterKey) {
    setSeenFilter(filterKey)
    setExpanded(filtering ? true : {})
    setPagination((current) => ({ ...current, pageIndex: 0 }))
  }

  // Свой порядок виден, только пока строки не отсортированы по столбцу.
  const canReorder = order !== null && sorting.length === 0
  const table = useTable({
    features,
    data: rows,
    columns,
    getRowId: (row) => row.id,
    getRowCanExpand: () => true,
    state: { sorting, expanded, pagination, columnVisibility: props.columnVisibility },
    onSortingChange: setSorting,
    onExpandedChange: setExpanded,
    onPaginationChange: setPagination,
    onColumnVisibilityChange: props.onColumnVisibilityChange,
    // Перестановка меняет данные; без этого таблица прыгала бы на первую страницу и сворачивала строки.
    autoResetPageIndex: false,
    autoResetExpanded: false,
    meta: {
      canReorder,
      moveToEdge: (id, edge) => order && onReorder(moveToEdge(order, id, edge)),
    },
  })

  const pageRows = table.getRowModel().rows
  const visibleIds = pageRows.map((row) => row.id)
  const sensors = useSensors(
    useSensor(MouseSensor),
    useSensor(TouchSensor),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  )

  function handleDragEnd({ active, over }: DragEndEvent) {
    if (order && over && active.id !== over.id) {
      onReorder(moveVisible(order, visibleIds, String(active.id), String(over.id)))
    }
  }

  const sortedBy = sorting
    .map(({ id }) => table.getColumn(id)?.columnDef.meta?.label)
    .filter(Boolean)
    .join(', ')

  return (
    <div className="flex flex-col gap-3">
      <div className="flex min-h-7 flex-wrap items-center justify-between gap-2 text-sm text-muted-foreground">
        {sorting.length > 0 ? (
          <div className="flex flex-wrap items-center gap-2">
            <span>Сортировка: {sortedBy}</span>
            <Button variant="ghost" size="sm" onClick={() => setSorting([])}>
              <Undo2 />
              Свой порядок
            </Button>
          </div>
        ) : order === null ? (
          <span>Свой порядок недоступен — показан порядок каталога.</span>
        ) : (
          <span className="inline-flex items-center gap-1">
            Тяните за <GripVertical className="size-4" aria-hidden="true" />
            <span className="sr-only">ручку</span>, чтобы задать свой порядок.
          </span>
        )}
        <ViewOptions table={table} />
      </div>

      <div className="overflow-hidden rounded-lg border">
        <DndContext
          sensors={sensors}
          collisionDetection={collisionDetection}
          modifiers={[restrictToVerticalAxis]}
          onDragEnd={handleDragEnd}
          accessibility={accessibility}
        >
          <Table>
            <TableHeader className="bg-muted/50">
              {table.getHeaderGroups().map((group) => (
                <TableRow key={group.id} className="hover:bg-transparent">
                  {group.headers.map((header) => (
                    <TableHead key={header.id} className={header.column.columnDef.meta?.className}>
                      {header.isPlaceholder ? null : <table.FlexRender header={header} />}
                    </TableHead>
                  ))}
                </TableRow>
              ))}
            </TableHeader>
            {pageRows.length === 0 ? (
              <tbody>
                <TableRow className="hover:bg-transparent">
                  <TableCell colSpan={table.getVisibleLeafColumns().length} className="h-24 text-center text-muted-foreground">
                    Ничего не найдено.
                  </TableCell>
                </TableRow>
              </tbody>
            ) : (
              <SortableContext items={visibleIds} strategy={verticalListSortingStrategy} disabled={!canReorder}>
                {pageRows.map((row) => (
                  <SortableRow key={row.id} id={row.id} label={row.original.instrument.title} disabled={!canReorder}>
                    <TableRow data-state={row.getIsExpanded() ? 'selected' : undefined}>
                      {row.getVisibleCells().map((cell) => (
                        <TableCell key={cell.id} className={cell.column.columnDef.meta?.className}>
                          <table.FlexRender cell={cell} />
                        </TableCell>
                      ))}
                    </TableRow>
                    {row.getIsExpanded() && (
                      <TableRow className="hover:bg-transparent">
                        <TableCell colSpan={row.getVisibleCells().length} className="p-0 whitespace-normal">
                          <DatasetList datasets={row.original.datasets} expectedDate={expectedDate} />
                        </TableCell>
                      </TableRow>
                    )}
                  </SortableRow>
                ))}
              </SortableContext>
            )}
          </Table>
        </DndContext>
      </div>

      <Pagination table={table} />
    </div>
  )
}
