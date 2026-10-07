import { useLayoutEffect, useRef } from 'react'
import { useRunLog } from '@/api/useRunLog'

// Пользователь у нижнего края (в пределах STICK_PX) — новые строки прокручивают лог; отлистал вверх — не мешаем читать.
const STICK_PX = 24

/** Лог запуска: моноширинный, дочитывается с сервера, пока запуск идёт. */
export default function LogView({ runId }: { runId: string }) {
  const { text, complete, error } = useRunLog(runId)
  const box = useRef<HTMLPreElement>(null)
  const atBottom = useRef(true)

  useLayoutEffect(() => {
    if (box.current && atBottom.current) box.current.scrollTop = box.current.scrollHeight
  }, [text])

  return (
    <div className="flex flex-col gap-1.5">
      <pre
        ref={box}
        onScroll={(event) => {
          const el = event.currentTarget
          atBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < STICK_PX
        }}
        className="max-h-[28rem] min-h-32 overflow-auto rounded-lg border bg-muted/40 p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap [overflow-wrap:anywhere]"
        tabIndex={0}
        aria-label="Лог запуска"
      >
        {text || (complete ? 'Лог пуст.' : 'Жду вывод…')}
      </pre>
      {error && (
        <p className="text-xs text-status-critical" role="status">
          Лог не обновляется: {error}. Повторю через пару секунд.
        </p>
      )}
    </div>
  )
}
