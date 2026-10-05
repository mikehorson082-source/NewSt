import type { Column } from '@tanstack/react-table'
import { cn } from 'cn'
import { ArrowDown, ArrowUp, ChevronsUpDown, EyeOff, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import type { InstrumentRow } from '@/lib/inventory'
import type { CatalogFeatures } from './features'

interface Props<TValue> {
  column: Column<CatalogFeatures, InstrumentRow, TValue>
  title: string
  className?: string
}

/** Заголовок столбца, как в примере shadcn: меню «по возрастанию / по убыванию / скрыть». */
export default function ColumnHeader<TValue>({ column, title, className }: Props<TValue>) {
  if (!column.getCanSort()) return <div className={className}>{title}</div>
  const sorted = column.getIsSorted()
  return (
    <div className={cn('flex items-center', className)}>
      <DropdownMenu>
        <DropdownMenuTrigger render={<Button variant="ghost" size="sm" className="-mx-2.5" />}>
          <span>{title}</span>
          {sorted === 'desc' ? <ArrowDown /> : sorted === 'asc' ? <ArrowUp /> : <ChevronsUpDown />}
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-auto">
          <DropdownMenuItem onClick={() => column.toggleSorting(false)}>
            <ArrowUp />
            По возрастанию
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => column.toggleSorting(true)}>
            <ArrowDown />
            По убыванию
          </DropdownMenuItem>
          {sorted && (
            <DropdownMenuItem onClick={() => column.clearSorting()}>
              <X />
              Свой порядок
            </DropdownMenuItem>
          )}
          {column.getCanHide() && (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => column.toggleVisibility(false)}>
                <EyeOff />
                Скрыть столбец
              </DropdownMenuItem>
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}
