import { useState } from 'react'
import { formatNumber } from '../lib/format'
import type { SectionView } from '../lib/inventory'
import { isProblem } from '../lib/status'
import InstrumentCard from './InstrumentCard'

interface Props {
  view: SectionView
  expectedDate: string
  defaultOpen: boolean
  filtering: boolean
}

/** Раздел верхнего уровня: Фьючерсы, Акции, Общее. */
export default function SectionBlock({ view, expectedDate, defaultOpen, filtering }: Props) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <details className="section" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="section__summary">
        <h2 className="section__title">{view.section.title}</h2>
        <span className="section__count">{formatNumber(view.instruments.length)}</span>
        <span className="section__desc">{view.section.description}</span>
      </summary>
      {open &&
        view.instruments.map((item) => (
          // При фильтре карточки раскрыты; без фильтра — только те, где есть проблемы.
          <InstrumentCard
            key={item.instrument.id}
            view={item}
            expectedDate={expectedDate}
            defaultOpen={filtering || item.rows.some((row) => isProblem(row.status))}
          />
        ))}
    </details>
  )
}
