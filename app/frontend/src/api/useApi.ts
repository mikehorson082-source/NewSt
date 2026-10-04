import { useCallback, useEffect, useState } from 'react'
import { apiGet } from './client'
import type { ApiGet } from './schema'

export interface ApiState<T> {
  data: T | null
  error: string | null
  loading: boolean // идёт запрос: первый или повторный
  refresh: () => void
}

/** Данные GET-метода API. При повторном запросе прежние данные остаются на экране. */
export function useApi<P extends keyof ApiGet>(path: P): ApiState<ApiGet[P]> {
  const [data, setData] = useState<ApiGet[P] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    apiGet(path, controller.signal)
      .then((result) => {
        setData(result)
        setError(null)
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason))
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [path, attempt])

  const refresh = useCallback(() => {
    setLoading(true)
    setAttempt((value) => value + 1)
  }, [])

  return { data, error, loading, refresh }
}
