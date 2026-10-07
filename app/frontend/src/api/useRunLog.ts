import { useEffect, useState } from 'react'
import { apiGet } from './client'

const POLL_MS = 1000
const RETRY_MS = 3000
const MAX_CHARS = 400_000 // в браузере держим хвост лога: начало длинной загрузки не нужно

interface LogState {
  runId: string | null
  text: string
  offset: number
  complete: boolean
  error: string | null
  polls: number // каждый ответ сервера — новый опрос, даже если лог не вырос
}

const empty = (runId: string | null): LogState => ({ runId, text: '', offset: 0, complete: false, error: null, polls: 0 })

/** Лог запуска: дочитывается с сервера по кусочкам, пока запуск не закончится. */
export function useRunLog(runId: string | null): { text: string; complete: boolean; error: string | null } {
  const [log, setLog] = useState<LogState>(() => empty(runId))
  if (log.runId !== runId) setLog(empty(runId)) // выбрали другой запуск — читаем его лог с начала

  const { offset, complete, polls, error } = log
  useEffect(() => {
    if (runId === null || complete) return
    const controller = new AbortController()
    const delay = polls === 0 ? 0 : error ? RETRY_MS : POLL_MS
    const timer = setTimeout(() => {
      apiGet('/api/pipeline/log', controller.signal, { run_id: runId, offset })
        .then((chunk) =>
          setLog((current) => {
            if (current.runId !== runId) return current
            let text = current.text + chunk.text
            if (text.length > MAX_CHARS) text = '… начало лога не показано\n' + text.slice(-MAX_CHARS)
            return { ...current, text, offset: chunk.offset, complete: chunk.complete, error: null, polls: current.polls + 1 }
          }),
        )
        .catch((reason: unknown) => {
          if (controller.signal.aborted) return
          const message = reason instanceof Error ? reason.message : String(reason)
          setLog((current) => (current.runId === runId ? { ...current, error: message, polls: current.polls + 1 } : current))
        })
    }, delay)
    return () => {
      clearTimeout(timer)
      controller.abort()
    }
  }, [runId, offset, complete, polls, error])

  return { text: log.text, complete: log.complete, error: log.error }
}
