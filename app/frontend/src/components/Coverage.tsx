import { formatDate, formatDateWithWeekday } from '../lib/format'
import { coverage, expectedOffset, yearLabelStep, type Axis } from '../lib/inventory'

// Сколько % ширины нужно подписи «2022» перед первой отметкой года.
const START_LABEL_ROOM = 12

interface BarProps {
  first: string
  last: string
  axis: Axis
  expectedDate: string
}

/** Полоса «за какие даты есть данные» на общей оси инструмента; вертикальная отметка — ожидаемая дата. */
export function CoverageBar({ first, last, axis, expectedDate }: BarProps) {
  const { left, width } = coverage(first, last, axis)
  const label = `Данные с ${formatDate(first)} по ${formatDate(last)}`
  return (
    // Зона наведения выше самой полосы.
    <div className="relative h-4" role="img" aria-label={label} title={label}>
      <span className="absolute inset-x-0 top-[7px] h-0.5 rounded-full bg-border" />
      {/* Отметка ожидаемой даты лежит под полосой и равна ей по высоте: свежий
          набор закрывает её целиком, у отстающего она видна справа от конца полосы. */}
      <span
        className="absolute top-1 h-2 w-px bg-muted-foreground"
        style={{ left: `${expectedOffset(expectedDate, axis)}%` }}
      />
      <span
        className="absolute top-1 h-2 rounded-full bg-coverage"
        style={{ left: `${left}%`, width: `max(4px, ${width}%)` }}
      />
    </div>
  )
}

/** Подписи лет над полосами инструмента. */
export function CoverageAxis({ axis, expectedDate }: { axis: Axis; expectedDate: string }) {
  // На длинной оси подписан каждый step-й год; черты без подписей только шумят.
  const step = yearLabelStep(axis)
  const ticks = axis.years.filter(({ year }) => year % step === 0)
  // Стартовый год подписываем, только если до первой отметки хватает места.
  const showStart = (ticks[0]?.offset ?? 100) >= START_LABEL_ROOM
  return (
    <div className="relative h-[18px] text-xs leading-[18px] text-muted-foreground tabular-nums" aria-hidden="true">
      {showStart && <span className="absolute top-0 left-0">{new Date(axis.start).getUTCFullYear()}</span>}
      {ticks.map(({ year, offset }) => (
        <span key={year} className="absolute inset-y-0 border-l pl-1 whitespace-nowrap" style={{ left: `${offset}%` }}>
          {year}
        </span>
      ))}
      <span
        className="absolute inset-y-0 w-px bg-muted-foreground"
        style={{ left: `${expectedOffset(expectedDate, axis)}%` }}
        title={`Ожидаемая дата: ${formatDateWithWeekday(expectedDate)}`}
      />
    </div>
  )
}
