import { formatDate, formatDateWithWeekday } from '../lib/format'
import { coverage, expectedOffset, yearLabelStep, type Axis } from '../lib/inventory'

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
  const step = yearLabelStep(axis)
  return (
    <div className="axis" aria-hidden="true">
      {axis.years.map(({ year, offset }) => (
        <span key={year} className="axis__tick" style={{ left: `${offset}%` }}>
          {year % step === 0 ? year : ''}
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
