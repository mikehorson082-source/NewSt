import { useCallback, useEffect, useState } from 'react'
import { apiGet, apiPost } from './client'
import type { PipelineState, Run } from './schema'

// Пока что-то идёт — опрашиваем часто, иначе редко: заметить загрузку, запущенную из терминала.
const ACTIVE_POLL_MS = 1500
const IDLE_POLL_MS = 10_000

export interface Pipeline {
  state: PipelineState | null
  error: string | null // не удалось получить состояние
  actionError: string | null // не удалось запустить или остановить
  start: (steps: string[]) => Promise<Run | null>
  stop: (runId: string) => Promise<void>
}

/** Состояние конвейера с опросом сервера; запуск и остановка шагов. */
export function usePipeline(): Pipeline {
  const [state, setState] = useState<PipelineState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  const refresh = useCallback(() => setTick((value) => value + 1), [])
  const active = state !== null && (state.busy !== null || state.runs.some((run) => run.status === 'running'))

  useEffect(() => {
    const controller = new AbortController()
    apiGet('/api/pipeline', controller.signal)
      .then((result) => {
        setState(result)
        setError(null)
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason))
      })
    return () => controller.abort()
  }, [tick])

  useEffect(() => {
    // Вкладка скрыта (телефон в кармане) — не опрашиваем; вернулись — сразу обновляем.
    const id = setInterval(() => document.hidden || refresh(), active ? ACTIVE_POLL_MS : IDLE_POLL_MS)
    const onVisible = () => document.hidden || refresh()
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      clearInterval(id)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [active, refresh])

  const start = useCallback(
    async (steps: string[]) => {
      setActionError(null)
      try {
        const run = await apiPost('/api/pipeline/runs', { steps })
        refresh()
        return run
      } catch (reason) {
        setActionError(reason instanceof Error ? reason.message : String(reason))
        refresh()
        return null
      }
    },
    [refresh],
  )

  const stop = useCallback(
    async (runId: string) => {
      setActionError(null)
      try {
        await apiPost('/api/pipeline/stop', { run_id: runId })
      } catch (reason) {
        setActionError(reason instanceof Error ? reason.message : String(reason))
      }
      refresh()
    },
    [refresh],
  )

  return { state, error, actionError, start, stop }
}
