import { useMemo, useState } from 'react'
import { useApi } from '../api/useApi'
import GroupSection from '../components/GroupSection'
import StatusFilter from '../components/StatusFilter'
import { formatDateTime, formatDateWithWeekday } from '../lib/format'
import { countByStatus, groupRows, type Filter } from '../lib/inventory'
import { isProblem } from '../lib/status'

/** Экран 1 «Инвентарь данных»: что лежит в data/, за какие даты и насколько свежее. */
export default function DataCatalog() {
  const { data, error, loading, refresh } = useApi('/api/data/status')
  const [filter, setFilter] = useState<Filter>({ status: null, query: '' })
  const groups = useMemo(() => (data ? groupRows(data, filter) : []), [data, filter])
  const counts = useMemo(() => countByStatus(data?.datasets ?? []), [data])
  const filtering = filter.status !== null || filter.query.trim() !== ''

  return (
    <section className="catalog" aria-busy={loading}>
      <header className="catalog__head">
        <div>
          <h1>Инвентарь данных</h1>
          {data && (
            <p className="catalog__meta">
              Свежесть считается от {formatDateWithWeekday(data.expected_date)} — последнего завершённого
              торгового дня. Проверено {formatDateTime(data.generated_at)}.
            </p>
          )}
        </div>
        <button type="button" className="button" onClick={refresh} disabled={loading}>
          {loading ? 'Проверяю…' : 'Проверить снова'}
        </button>
      </header>

      {error && (
        <div className="callout" role="alert">
          <p>Не удалось получить инвентарь: {error}</p>
          <p>
            Запущен ли API? <code>uv run uvicorn app.api.main:app --port 8000</code>
          </p>
        </div>
      )}

      {!data && !error && <p className="empty">Сканирую data/…</p>}

      {data && (
        <div className={loading ? 'catalog__body catalog__body--refreshing' : 'catalog__body'}>
          <div className="filters">
            <StatusFilter
              counts={counts}
              total={data.datasets.length}
              active={filter.status}
              onChange={(status) => setFilter((current) => ({ ...current, status }))}
            />
            <label className="search">
              <span className="visually-hidden">Поиск по рынку, тикеру или файлу</span>
              <input
                type="search"
                placeholder="Рынок, тикер или файл"
                value={filter.query}
                onChange={(event) => setFilter((current) => ({ ...current, query: event.target.value }))}
              />
            </label>
          </div>

          {groups.length === 0 && <p className="empty">Ничего не найдено.</p>}
          {groups.map((view) => (
            // При фильтре группы раскрыты; без фильтра — только те, где есть проблемы.
            <GroupSection
              key={`${view.group.id}:${filtering}`}
              view={view}
              expectedDate={data.expected_date}
              defaultOpen={filtering || view.rows.some((row) => isProblem(row.status))}
            />
          ))}
        </div>
      )}
    </section>
  )
}
