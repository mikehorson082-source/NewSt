import { cn } from 'cn'
import type { Run } from '@/api/schema'
import { formatDateTime } from '@/lib/format'
import { duration, STATUS_LABEL, summary } from '@/lib/pipeline'
import { RunStatusIcon } from './RunStatus'

interface Props {
  runs: Run[] // новые сверху
  selected: string | null
  onSelect: (runId: string) => void
}

/** Журнал запусков: нажатие показывает лог запуска. */
export default function RunHistory({ runs, selected, onSelect }: Props) {
  if (runs.length === 0) {
    return <p className="text-sm text-muted-foreground">Запусков из пульта ещё не было.</p>
  }
  return (
    <ul className="divide-y overflow-hidden rounded-lg border">
      {runs.map((run) => (
        <li key={run.id}>
          <button
            type="button"
            onClick={() => onSelect(run.id)}
            aria-current={run.id === selected ? 'true' : undefined}
            className={cn(
              'grid w-full grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-x-3 px-4 py-2.5 text-left text-sm outline-none hover:bg-muted/50 focus-visible:ring-3 focus-visible:ring-ring/50',
              run.id === selected && 'bg-muted',
            )}
          >
            <RunStatusIcon status={run.status} className="mt-0.5" />
            <span className="flex min-w-0 flex-col">
              <span className="font-medium [overflow-wrap:anywhere]">
                {run.title}
                <span className="sr-only">: {STATUS_LABEL[run.status]}</span>
              </span>
              <span className="text-xs text-muted-foreground">{summary(run)}</span>
            </span>
            <span className="flex flex-col items-end text-xs text-muted-foreground tabular-nums">
              <span>{formatDateTime(run.started_at)}</span>
              <span>{duration(run.started_at, run.finished_at)}</span>
            </span>
          </button>
        </li>
      ))}
    </ul>
  )
}
