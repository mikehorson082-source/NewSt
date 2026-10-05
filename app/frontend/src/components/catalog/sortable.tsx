import { useSortable } from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { cn } from 'cn'
import { GripVertical } from 'lucide-react'
import { createContext, use, type ReactNode } from 'react'
import { Button } from '@/components/ui/button'

// Ручка рисуется внутри ячейки таблицы, а тащится вся строка: строка создаёт ручку
// и передаёт её в ячейку через контекст.
const HandleContext = createContext<ReactNode>(null)

interface RowProps {
  id: string
  label: string // для диктора: «Переместить: Нефть Brent»
  disabled: boolean
  children: ReactNode
}

/**
 * Инструмент вместе с раскрытыми наборами — свой tbody: при перетаскивании
 * едут обе строки. В одной таблице может быть сколько угодно tbody.
 */
export function SortableRow({ id, label, disabled, children }: RowProps) {
  const { attributes, listeners, setNodeRef, setActivatorNodeRef, transform, transition, isDragging } = useSortable({
    id,
    disabled,
  })
  // Тянуть мышью или пальцем; с клавиатуры — пробел, стрелки, пробел.
  const handle = (
    <Button
      ref={setActivatorNodeRef}
      variant="ghost"
      size="icon-sm"
      className="cursor-grab touch-none text-muted-foreground hover:bg-transparent active:cursor-grabbing disabled:opacity-30"
      disabled={disabled}
      title={disabled ? 'Перетаскивание выключено: сбросьте сортировку по столбцу' : 'Перетащите, чтобы поменять порядок'}
      {...attributes}
      {...listeners}
      aria-label={`Переместить: ${label}`}
    >
      <GripVertical />
    </Button>
  )
  return (
    <HandleContext value={handle}>
      <tbody
        ref={setNodeRef}
        className={cn('relative', isDragging && 'z-10 bg-background opacity-90 shadow-lg')}
        style={{ transform: CSS.Translate.toString(transform), transition }}
      >
        {children}
      </tbody>
    </HandleContext>
  )
}

/** Ручка ⋮⋮ строки, внутри которой стоит ячейка. */
export function DragHandle() {
  return use(HandleContext)
}
