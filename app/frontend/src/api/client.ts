import type { ApiGet } from './schema'

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/** GET к API; тип ответа берётся из контракта: apiGet('/api/data/status') → InventoryReport. */
export async function apiGet<P extends keyof ApiGet>(path: P, signal?: AbortSignal): Promise<ApiGet[P]> {
  const response = await fetch(path, { headers: { Accept: 'application/json' }, signal })
  if (!response.ok) {
    throw new ApiError(response.status, `${path}: ${response.status} ${response.statusText}`)
  }
  return (await response.json()) as ApiGet[P]
}
