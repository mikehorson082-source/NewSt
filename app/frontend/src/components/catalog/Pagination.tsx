import type { ReactTable } from '@tanstack/react-table'
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { formatNumber } from '@/lib/format'
import type { InstrumentRow } from '@/lib/inventory'
import type { CatalogFeatures } from './features'

const PAGE_SIZES = [25, 50, 100, 200]

/** Низ таблицы, как в примере shadcn: сколько показано, строк на странице, листание. */
export default function Pagination({ table }: { table: ReactTable<CatalogFeatures, InstrumentRow> }) {
  const { pageIndex, pageSize } = table.state.pagination
  const total = table.getRowCount()
  const pages = table.getPageCount()
  const from = total === 0 ? 0 : pageIndex * pageSize + 1
  const to = Math.min(total, (pageIndex + 1) * pageSize)

  return (
    <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2 px-1 text-sm">
      <div className="text-muted-foreground tabular-nums">
        {formatNumber(from)}–{formatNumber(to)} из {formatNumber(total)}
      </div>
      <div className="flex items-center gap-4 lg:gap-6">
        <div className="hidden items-center gap-2 sm:flex">
          <span className="font-medium">Строк на странице</span>
          <Select value={String(pageSize)} onValueChange={(value) => table.setPageSize(Number(value))}>
            <SelectTrigger size="sm" className="w-18">
              <SelectValue />
            </SelectTrigger>
            <SelectContent side="top">
              {PAGE_SIZES.map((size) => (
                <SelectItem key={size} value={String(size)}>
                  {size}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        {pages > 1 && (
          <>
            <div className="font-medium tabular-nums">
              Стр. {pageIndex + 1} из {pages}
            </div>
            <div className="flex items-center gap-1.5">
              <Button
                variant="outline"
                size="icon-sm"
                className="hidden lg:flex"
                onClick={() => table.setPageIndex(0)}
                disabled={!table.getCanPreviousPage()}
                aria-label="Первая страница"
              >
                <ChevronsLeft />
              </Button>
              <Button
                variant="outline"
                size="icon-sm"
                onClick={() => table.previousPage()}
                disabled={!table.getCanPreviousPage()}
                aria-label="Предыдущая страница"
              >
                <ChevronLeft />
              </Button>
              <Button
                variant="outline"
                size="icon-sm"
                onClick={() => table.nextPage()}
                disabled={!table.getCanNextPage()}
                aria-label="Следующая страница"
              >
                <ChevronRight />
              </Button>
              <Button
                variant="outline"
                size="icon-sm"
                className="hidden lg:flex"
                onClick={() => table.setPageIndex(pages - 1)}
                disabled={!table.getCanNextPage()}
                aria-label="Последняя страница"
              >
                <ChevronsRight />
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
