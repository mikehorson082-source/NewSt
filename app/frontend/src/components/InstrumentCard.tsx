import { useMemo, useState } from 'react'
import { countByStatus, timeAxis, type InstrumentView } from '../lib/inventory'
import { STATUS, STATUS_ORDER, isProblem } from '../lib/status'
import { CoverageAxis } from './Coverage'
import DatasetRow from './DatasetRow'
import StatusIcon from './StatusIcon'

interface Props {
  view: InstrumentView
  expectedDate: string
  defaultOpen: boolean
}

/** Карточка инструмента: название, сводка статусов, а раскрытая — все его наборы на общей оси лет. */
export default function InstrumentCard({ view, expectedDate, defaultOpen }: Props) {
  const [open, setOpen] = useState(defaultOpen)
  const counts = countByStatus(view.rows)
  const shown = STATUS_ORDER.filter((status) => counts[status] > 0)

  return (
    <details className="instrument" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="instrument__summary">
        <span className="instrument__heading">
          <span className="instrument__title">{view.instrument.title}</span>
          <span className="instrument__subtitle">{view.instrument.subtitle}</span>
        </span>
        <span className="instrument__health">
          {shown.map((status) => (
            <span
              key={status}
              className={isProblem(status) ? 'health health--problem' : 'health'}
              title={STATUS[status].many}
            >
              <StatusIcon status={status} />
              {counts[status]}
              <span className="visually-hidden"> {STATUS[status].many.toLowerCase()}</span>
            </span>
          ))}
        </span>
      </summary>
      {/* Содержимое свёрнутой карточки не рисуется: акций сотни, телефону так легче. */}
      {open && <InstrumentRows view={view} expectedDate={expectedDate} />}
    </details>
  )
}

function InstrumentRows({ view, expectedDate }: { view: InstrumentView; expectedDate: string }) {
  const axis = useMemo(() => timeAxis(view.rows, expectedDate), [view.rows, expectedDate])
  return (
    <ul className="rows">
      {axis && (
        <li className="row row--axis" aria-hidden="true">
          <div className="row__bar">
            <CoverageAxis axis={axis} expectedDate={expectedDate} />
          </div>
        </li>
      )}
      {view.rows.map((row) => (
        <DatasetRow key={row.id} row={row} axis={axis} expectedDate={expectedDate} />
      ))}
    </ul>
  )
}
