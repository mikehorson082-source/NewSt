import { ArrowRight, ExternalLink, Play } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import type { PipelineState, Run, Step } from '@/api/schema'
import { usePipeline } from '@/api/usePipeline'
import LogView from '@/components/pipeline/LogView'
import RunHistory from '@/components/pipeline/RunHistory'
import { RunStatusLabel } from '@/components/pipeline/RunStatus'
import StepRow from '@/components/pipeline/StepRow'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { formatDateTime } from '@/lib/format'
import { activeRun, duration, lastResult, progress } from '@/lib/pipeline'

/** Экран «Загрузка»: докачка с биржи, склейка и проверки — те же скрипты, что в терминале. */
export default function DataLoad() {
  const { state, error, actionError, start, stop } = usePipeline()
  const [selected, setSelected] = useState<string | null>(null) // чей лог показан; null — последнего запуска

  async function launch(steps: string[]) {
    const run = await start(steps)
    if (run) setSelected(run.id)
  }

  return (
    <section className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold tracking-tight">Загрузка данных</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Докачка с биржи и склейка. Это те же скрипты, что в терминале: команда указана под каждым шагом. Данные пишет
          только один процесс за раз — пульт или терминал.
        </p>
      </header>

      {error && <Notice tone="error">Не удалось получить состояние загрузки: {error}</Notice>}
      {actionError && <Notice tone="error">{actionError}</Notice>}
      {!state && !error && <Skeleton className="h-64 w-full" />}
      {state && <Board state={state} selected={selected} onSelect={setSelected} onLaunch={launch} onStop={stop} />}
    </section>
  )
}

interface BoardProps {
  state: PipelineState
  selected: string | null
  onSelect: (runId: string) => void
  onLaunch: (steps: string[]) => void
  onStop: (runId: string) => void
}

function Board({ state, selected, onSelect, onLaunch, onStop }: BoardProps) {
  const { steps, runs, busy } = state
  const dataRun = activeRun(runs, 'data')
  const checkRun = activeRun(runs, 'check')
  const byId = new Map(steps.map((step) => [step.id, step]))
  const shown = runs.find((run) => run.id === selected) ?? runs[0]

  // Почему кнопка шага сейчас неактивна: загрузка — по одной (и не при чужой), проверка — по одной.
  const dataBlocked = dataRun ? `Идёт: ${dataRun.title}` : busy ? `Данные сейчас пишет ${busy}` : null
  const blockedFor = (step: Step) => (step.kind === 'data' ? dataBlocked : checkRun ? `Идёт: ${checkRun.title}` : null)

  return (
    <>
      {busy && !dataRun && (
        <Notice tone="warning">
          Сейчас данные пишет {busy}. Запуск загрузки станет доступен, когда он закончит.
        </Notice>
      )}

      <UpdateAll
        chain={state.update_all.map((id) => byId.get(id)?.title ?? id)}
        run={dataRun}
        stepTitle={(id) => byId.get(id)?.title ?? id}
        blocked={dataBlocked}
        onStart={() => onLaunch(state.update_all)}
        onStop={onStop}
      />

      <StepGroup
        title="Шаги загрузки"
        steps={steps.filter((step) => step.kind === 'data')}
        runs={runs}
        blockedFor={blockedFor}
        onLaunch={onLaunch}
      />
      <StepGroup
        title="Проверки"
        hint="Только читают данные: можно запускать и во время загрузки."
        steps={steps.filter((step) => step.kind === 'check')}
        runs={runs}
        blockedFor={blockedFor}
        onLaunch={onLaunch}
      />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(18rem,1fr)]">
        <div className="flex min-w-0 flex-col gap-2">
          <h2 className="text-lg font-semibold">Лог</h2>
          {shown ? (
            <>
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-sm">
                <span className="font-medium">{shown.title}</span>
                <RunStatusLabel status={shown.status} />
                <span className="text-muted-foreground tabular-nums">
                  {formatDateTime(shown.started_at)} · {duration(shown.started_at, shown.finished_at)}
                </span>
              </div>
              <LogView key={shown.id} runId={shown.id} />
              <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                <span>
                  Файл:{' '}
                  <code className="font-mono select-all [overflow-wrap:anywhere]">
                    {state.runs_dir}/{shown.id}.log
                  </code>
                </span>
                <a
                  href={`/api/pipeline/log/file?run_id=${encodeURIComponent(shown.id)}`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 font-medium text-foreground underline-offset-4 hover:underline"
                >
                  Открыть лог целиком
                  <ExternalLink className="size-3" aria-hidden="true" />
                </a>
              </p>
            </>
          ) : (
            <p className="text-sm text-muted-foreground">Здесь появится вывод скрипта после первого запуска.</p>
          )}
        </div>
        <div className="flex min-w-0 flex-col gap-2">
          <h2 className="text-lg font-semibold">Журнал запусков</h2>
          <RunHistory runs={runs} selected={shown?.id ?? null} onSelect={onSelect} />
        </div>
      </div>
    </>
  )
}

interface UpdateAllProps {
  chain: string[] // названия шагов «Обновить всё» по порядку
  run: Run | undefined // идущая загрузка
  stepTitle: (id: string) => string
  blocked: string | null
  onStart: () => void
  onStop: (runId: string) => void
}

function UpdateAll({ chain, run, stepTitle, blocked, onStart, onStop }: UpdateAllProps) {
  return (
    <div className="flex flex-col gap-4 rounded-lg border p-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex min-w-0 flex-col gap-1.5">
        <h2 className="text-lg font-semibold">Обновить всё</h2>
        <p className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-sm">
          {chain.map((title, index) => (
            <span key={title} className="inline-flex items-center gap-1.5">
              {index > 0 && <ArrowRight className="size-3.5 text-muted-foreground" aria-label="затем" />}
              {title}
            </span>
          ))}
        </p>
        <p className="text-xs text-muted-foreground">
          Упал шаг — остальные идут дальше. Склейка пропускается, если не докачались минутки: на недокачанных данных
          она дала бы ряд с дырой в конце.
        </p>
      </div>

      {run ? (
        <div className="flex shrink-0 flex-col gap-2 sm:items-end">
          <span className="text-sm tabular-nums" role="status">
            {runProgressText(run, stepTitle)}
          </span>
          <Button variant="outline" onClick={() => onStop(run.id)} disabled={run.stop_requested}>
            {run.stop_requested ? 'Остановлю после текущего шага' : 'Остановить после шага'}
          </Button>
        </div>
      ) : (
        <Button className="shrink-0" onClick={onStart} disabled={blocked !== null} title={blocked ?? undefined}>
          <Play />
          Обновить всё
        </Button>
      )}
    </div>
  )
}

/** «Шаг 3 из 5: Минутные свечи · 4:12». */
function runProgressText(run: Run, stepTitle: (id: string) => string): string {
  const { done, total, current } = progress(run)
  if (!current) return `Шагов позади: ${done} из ${total}`
  return `Шаг ${done + 1} из ${total}: ${stepTitle(current.step)} · ${duration(current.started_at ?? run.started_at, null)}`
}

interface GroupProps {
  title: string
  hint?: string
  steps: Step[]
  runs: Run[]
  blockedFor: (step: Step) => string | null
  onLaunch: (steps: string[]) => void
}

function StepGroup({ title, hint, steps, runs, blockedFor, onLaunch }: GroupProps) {
  if (steps.length === 0) return null
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-baseline gap-x-3">
        <h2 className="text-lg font-semibold">{title}</h2>
        {hint && <span className="text-sm text-muted-foreground">{hint}</span>}
      </div>
      <ul className="divide-y overflow-hidden rounded-lg border">
        {steps.map((step) => (
          <StepRow
            key={step.id}
            step={step}
            last={lastResult(runs, step.id)}
            blocked={blockedFor(step)}
            onRun={() => onLaunch([step.id])}
          />
        ))}
      </ul>
    </div>
  )
}

function Notice({ tone, children }: { tone: 'error' | 'warning'; children: ReactNode }) {
  const style =
    tone === 'error'
      ? 'border-destructive/40 bg-destructive/5 text-destructive'
      : 'border-status-warning/50 bg-status-warning/10'
  return (
    <p className={`rounded-lg border px-3 py-2 text-sm ${style}`} role={tone === 'error' ? 'alert' : 'status'}>
      {children}
    </p>
  )
}
