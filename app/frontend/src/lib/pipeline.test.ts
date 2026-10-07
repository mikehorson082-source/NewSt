import { describe, expect, it } from 'vitest'
import type { Run, StepResult } from '../api/schema'
import { activeRun, duration, lastResult, progress, summary } from './pipeline'

function step(id: string, status: StepResult['status']): StepResult {
  return { step: id, status, started_at: null, finished_at: null, exit_code: null, note: null }
}

function run(id: string, status: Run['status'], steps: StepResult[], kind: Run['kind'] = 'data'): Run {
  return {
    id,
    title: id,
    kind,
    status,
    started_at: '2026-10-06T23:00:00+03:00',
    finished_at: null,
    stop_requested: false,
    steps,
  }
}

describe('журнал запусков', () => {
  const runs = [
    run('new', 'stopped', [step('futures_min', 'ok'), step('continuous', 'stopped')]),
    run('old', 'error', [step('continuous', 'error')]),
  ]

  it('последний итог шага пропускает запуски, где до него не дошло', () => {
    expect(lastResult(runs, 'continuous')?.run.id).toBe('old')
    expect(lastResult(runs, 'futures_min')?.result.status).toBe('ok')
    expect(lastResult(runs, 'futoi')).toBeNull()
  })

  it('идущий запуск ищется по виду', () => {
    const list = [run('check', 'running', [], 'check'), ...runs]
    expect(activeRun(list, 'check')?.id).toBe('check')
    expect(activeRun(list, 'data')).toBeUndefined()
  })

  it('ход и сводка', () => {
    const current = run('now', 'running', [step('a', 'ok'), step('b', 'running'), step('c', 'pending')])
    expect(progress(current)).toEqual({ done: 1, total: 3, current: current.steps[1] })
    expect(summary(current)).toBe('1 готово · 1 идёт · 1 ждёт')
  })
})

describe('длительность', () => {
  it('минуты и часы', () => {
    expect(duration('2026-10-06T23:00:00+03:00', '2026-10-06T23:04:12+03:00')).toBe('4:12')
    expect(duration('2026-10-06T23:00:00+03:00', '2026-10-07T00:02:03+03:00')).toBe('1:02:03')
  })

  it('идёт — до сейчас', () => {
    expect(duration('2026-10-06T23:00:00+03:00', null, new Date('2026-10-06T23:00:42+03:00').getTime())).toBe('0:42')
  })
})
