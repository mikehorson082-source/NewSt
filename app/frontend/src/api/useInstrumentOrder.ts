import { useCallback, useRef, useState } from 'react'
import { apiPatch } from './client'
import type { InstrumentOrder } from './schema'
import { useApi } from './useApi'

type Sections = InstrumentOrder['sections']

export interface InstrumentOrderState {
  sections: Sections | null // null — ещё не загрузился или не загрузился вовсе
  error: string | null
  save: (section: string, ids: string[]) => void
}

/**
 * Свой порядок инструментов. Новый порядок виден сразу, а на сервер уходит следом;
 * не сохранился — показываем ошибку и перечитываем порядок с сервера.
 */
export function useInstrumentOrder(): InstrumentOrderState {
  const loaded = useApi('/api/settings/instrument-order')
  const [local, setLocal] = useState<Sections | null>(null)
  const [saveError, setSaveError] = useState<string | null>(null)
  // Ответы могут прийти не по порядку: применяем только ответ на последнее сохранение.
  const latest = useRef(0)
  const sections = local ?? loaded.data?.sections ?? null
  const { refresh } = loaded

  const save = useCallback(
    (section: string, ids: string[]) => {
      if (sections === null) return // не прочитав файл, не пишем в него
      setLocal({ ...sections, [section]: ids })
      setSaveError(null)
      const request = ++latest.current
      apiPatch('/api/settings/instrument-order', { sections: { [section]: ids } })
        .then((result) => {
          if (request === latest.current) setLocal(result.sections)
        })
        .catch((reason: unknown) => {
          if (request !== latest.current) return
          setSaveError(`Порядок не сохранился: ${reason instanceof Error ? reason.message : String(reason)}`)
          setLocal(null)
          refresh()
        })
    },
    [sections, refresh],
  )

  return { sections, error: saveError ?? (sections === null ? loaded.error : null), save }
}
