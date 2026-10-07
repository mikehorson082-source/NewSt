import type { Run, StepResult } from '../api/schema'

export type RunStatus = Run['status']
export type StepStatus = StepResult['status']
export type Kind = Run['kind']

export const STATUS_LABEL: Record<RunStatus | StepStatus, string> = {
  running: 'Идёт',
  ok: 'Готово',
  error: 'Ошибка',
  skipped: 'Пропущен',
  stopped: 'Остановлен',
  interrupted: 'Прерван',
  pending: 'Ждёт',
}

/** Идущий запуск этого вида: загрузка и проверка идут независимо, по одной. */
export function activeRun(runs: Run[], kind: Kind): Run | undefined {
  return runs.find((run) => run.kind === kind && run.status === 'running')
}

/**
 * Последний итог шага по журналу (runs — новые сверху). Шаги, до которых
 * очередь не дошла (ждёт, остановлен), не в счёт: они ничего не сделали.
 */
export function lastResult(runs: Run[], stepId: string): { run: Run; result: StepResult } | null {
  for (const run of runs) {
    const result = run.steps.find((item) => item.step === stepId)
    if (result && result.status !== 'pending' && result.status !== 'stopped') return { run, result }
  }
  return null
}

/** Длительность «0:42», «4:12», «1:02:03»; не закончился — до now. */
export function duration(startIso: string, endIso: string | null, now: number = Date.now()): string {
  const end = endIso ? new Date(endIso).getTime() : now
  const total = Math.max(0, Math.floor((end - new Date(startIso).getTime()) / 1000))
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const seconds = String(total % 60).padStart(2, '0')
  return hours > 0 ? `${hours}:${String(minutes).padStart(2, '0')}:${seconds}` : `${minutes}:${seconds}`
}

/** Ход запуска: сколько шагов позади и какой идёт сейчас. */
export function progress(run: Run): { done: number; total: number; current: StepResult | undefined } {
  return {
    done: run.steps.filter((step) => step.status !== 'pending' && step.status !== 'running').length,
    total: run.steps.length,
    current: run.steps.find((step) => step.status === 'running'),
  }
}

/** Сводка шагов запуска: «4 готово · 1 ошибка». */
export function summary(run: Run): string {
  const order: [StepStatus, string][] = [
    ['ok', 'готово'],
    ['error', 'ошибка'],
    ['skipped', 'пропущено'],
    ['stopped', 'остановлено'],
    ['running', 'идёт'],
    ['pending', 'ждёт'],
  ]
  return order
    .map(([status, word]) => [run.steps.filter((step) => step.status === status).length, word] as const)
    .filter(([count]) => count > 0)
    .map(([count, word]) => `${count} ${word}`)
    .join(' · ')
}
