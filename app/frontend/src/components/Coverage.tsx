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

/** Полоса «за какие даты есть данные» на общей оси группы; вертикальная отметка — ожидаемая дата. */
export function CoverageBar({ first, last, axis, expectedDate }: BarProps) {
  const { left, width } = coverage(first, last, axis)
  const label = `Данные с ${formatDate(first)} по ${formatDate(last)}`
  return (
    <div className="coverage" role="img" aria-label={label} title={label}>
      <span className="coverage__track" />
      <span className="coverage__expected" style={{ left: `${expectedOffset(expectedDate, axis)}%` }} />
      <span className="coverage__bar" style={{ left: `${left}%`, width: `max(4px, ${width}%)` }} />
    </div>
  )
}

/** Подписи лет над полосами группы. */
export function CoverageAxis({ axis, expectedDate }: { axis: Axis; expectedDate: string }) {
  // На длинной оси подписан каждый step-й год; черты без подписей только шумят.
  const step = yearLabelStep(axis)
  const ticks = axis.years.filter(({ year }) => year % step === 0)
  // Стартовый год подписываем, только если до первой отметки хватает места.
  const showStart = (ticks[0]?.offset ?? 100) >= START_LABEL_ROOM
  return (
    <div className="axis" aria-hidden="true">
      {showStart && <span className="axis__start">{new Date(axis.start).getUTCFullYear()}</span>}
      {ticks.map(({ year, offset }) => (
        <span key={year} className="axis__tick" style={{ left: `${offset}%` }}>
          {year}
        </span>
      ))}
      <span
        className="axis__expected"
        style={{ left: `${expectedOffset(expectedDate, axis)}%` }}
        title={`Ожидаемая дата: ${formatDateWithWeekday(expectedDate)}`}
      />
    </div>
  )
}
