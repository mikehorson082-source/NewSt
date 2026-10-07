import type { ApiGet, ApiPatch, ApiPost } from './schema'

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init: RequestInit): Promise<T> {
  const response = await fetch(path, { ...init, headers: { Accept: 'application/json', ...init.headers } })
  if (!response.ok) {
    // FastAPI объясняет ошибку в поле detail: «… повреждён; поправьте или удалите его».
    const body: unknown = await response.json().catch(() => null)
    const detail = body && typeof body === 'object' && 'detail' in body ? body.detail : null
    const reason = typeof detail === 'string' ? detail : `${response.status} ${response.statusText}`
    throw new ApiError(response.status, `${path}: ${reason}`)
  }
  return (await response.json()) as T
}

/**
 * GET к API; тип ответа берётся из контракта: apiGet('/api/data/status') → InventoryReport.
 * query — параметры строки запроса: apiGet('/api/pipeline/log', signal, { run_id, offset }).
 */
export function apiGet<P extends keyof ApiGet>(
  path: P,
  signal?: AbortSignal,
  query?: Record<string, string | number>,
): Promise<ApiGet[P]> {
  const params = query ? new URLSearchParams(Object.entries(query).map(([key, value]) => [key, String(value)])) : null
  return request(params ? `${path}?${params}` : path, { signal })
}

/** POST к API: тело и ответ — из контракта (ApiPost в schema.ts). */
export function apiPost<P extends keyof ApiPost>(path: P, body: ApiPost[P]['body']): Promise<ApiPost[P]['response']> {
  return request(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
}

/** PATCH к API: тело и ответ — из контракта (ApiPatch в schema.ts). */
export function apiPatch<P extends keyof ApiPatch>(path: P, body: ApiPatch[P]['body']): Promise<ApiPatch[P]['response']> {
  return request(path, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
}
