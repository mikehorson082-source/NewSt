import { Play } from 'lucide-react'
import type { Run, Step, StepResult } from '@/api/schema'
import { Button } from '@/components/ui/button'
import { formatDateTime } from '@/lib/format'
import { duration } from '@/lib/pipeline'
import { RunStatusLabel } from './RunStatus'

interface Props {
  step: Step
  last: { run: Run; result: StepResult } | null // последний итог шага по журналу
  blocked: string | null // почему сейчас нельзя запустить; null — можно
  onRun: () => void
}

/** Шаг конвейера: что делает, команда для терминала, куда пишет, последний итог и кнопка запуска. */
export default function StepRow({ step, last, blocked, onRun }: Props) {
  return (
    <li className="grid gap-x-6 gap-y-2 px-4 py-3 sm:grid-cols-[minmax(0,1fr)_auto] lg:grid-cols-[minmax(0,1.4fr)_minmax(12rem,1fr)_auto] lg:items-center">
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="font-medium">{step.title}</span>
        <span className="text-sm text-muted-foreground">{step.description}</span>
        {/* Одно нажатие выделяет команду целиком — удобно скопировать в терминал. */}
        <code className="mt-0.5 font-mono text-xs text-muted-foreground select-all [overflow-wrap:anywhere]">
          {step.command}
        </code>
        {step.writes.length > 0 && (
          <span className="text-xs text-muted-foreground">
            пишет в <span className="font-mono">{step.writes.map((folder) => `data/${folder}`).join(', ')}</span>
          </span>
        )}
      </div>

      <div className="text-sm sm:col-start-1 lg:col-start-2">
        {last ? (
          <div className="flex flex-col gap-0.5">
            <RunStatusLabel status={last.result.status} className="font-medium" />
            <span className="text-xs text-muted-foreground tabular-nums">
              {last.result.started_at
                ? `${formatDateTime(last.result.started_at)} · ${duration(last.result.started_at, last.result.finished_at)}`
                : formatDateTime(last.run.started_at)}
            </span>
            {last.result.note && <span className="text-xs text-muted-foreground">{last.result.note}</span>}
          </div>
        ) : (
          <span className="text-muted-foreground">Из пульта ещё не запускался</span>
        )}
      </div>

      <div className="sm:col-start-2 sm:row-span-2 sm:row-start-1 sm:self-center lg:col-start-3 lg:row-span-1">
        <Button variant="outline" size="sm" onClick={onRun} disabled={blocked !== null} title={blocked ?? undefined}>
          <Play />
          Запустить
        </Button>
      </div>
    </li>
  )
}
