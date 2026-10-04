import { useMemo, useState } from 'react'
import { countByStatus, timeAxis, type GroupView } from '../lib/inventory'
import { STATUS, STATUS_ORDER, isProblem } from '../lib/status'
import { CoverageAxis } from './Coverage'
import DatasetRow from './DatasetRow'
import StatusIcon from './StatusIcon'

interface Props {
  view: GroupView
  expectedDate: string
  defaultOpen: boolean
}

/** Группа строк: заголовок со сводкой статусов, ось лет и строки. Свёрнута, если всё в порядке. */
export default function GroupSection({ view, expectedDate, defaultOpen }: Props) {
  const [open, setOpen] = useState(defaultOpen)
  const axis = useMemo(() => timeAxis(view.rows, expectedDate), [view.rows, expectedDate])
  const counts = countByStatus(view.rows)
  const shown = STATUS_ORDER.filter((status) => counts[status] > 0)

  return (
    <details className="group" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="group__summary">
        <span className="group__heading">
          <span className="group__title">{view.group.title}</span>
          <span className="group__count">{view.rows.length}</span>
        </span>
        <span className="group__health">
          {shown.map((status) => (
            <span key={status} className={isProblem(status) ? 'health health--problem' : 'health'}>
              <StatusIcon status={status} />
              {counts[status]} {STATUS[status].many.toLowerCase()}
            </span>
          ))}
        </span>
      </summary>
      <p className="group__desc">{view.group.description}</p>
      <ul className="rows">
        {axis && (
          <li className="row row--axis" aria-hidden="true">
            <div className="row__bar">
              <CoverageAxis axis={axis} expectedDate={expectedDate} />
            </div>
          </li>
        )}
        {view.rows.map((row) => (
          <DatasetRow
            key={row.id}
            row={row}
            axis={axis}
            expectedDate={expectedDate}
            filesLabel={view.group.files_label}
          />
        ))}
      </ul>
    </details>
  )
}
