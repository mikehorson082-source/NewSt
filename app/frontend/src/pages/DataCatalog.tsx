import type { ColumnVisibilityState } from '@tanstack/react-table'
import { cn } from 'cn'
import { RefreshCw, Search, X } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useApi } from '@/api/useApi'
import { useInstrumentOrder } from '@/api/useInstrumentOrder'
import InstrumentTable from '@/components/catalog/InstrumentTable'
import StatusFilter from '@/components/catalog/StatusFilter'
import StatusIcon from '@/components/StatusIcon'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { InputGroup, InputGroupAddon, InputGroupInput } from '@/components/ui/input-group'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { formatDateTime, formatDateWithWeekday, formatNumber } from '@/lib/format'
import { countByStatus, filterTree, instrumentRow, type Filter } from '@/lib/inventory'
import { applyOrder } from '@/lib/order'
import { STATUS, STATUS_ORDER } from '@/lib/status'

const NO_FILTER: Filter = { statuses: [], query: '' }

/** Экран 1 «Инвентарь данных»: что лежит в data/, за какие даты и насколько свежее. */
export default function DataCatalog() {
  const { data, error, loading, refresh } = useApi('/api/data/status')
  const order = useInstrumentOrder()
  const [filter, setFilter] = useState<Filter>(NO_FILTER)
  const [tab, setTab] = useState<string | null>(null)
  const [columnVisibility, setColumnVisibility] = useState<ColumnVisibilityState>({})
  const filtering = filter.statuses.length > 0 || filter.query.trim() !== ''
  const filterKey = `${filter.statuses.join(',')}|${filter.query.trim().toLowerCase()}`
  const counts = useMemo(() => countByStatus(data?.datasets ?? []), [data])

  // Разделы всегда на месте (вкладки не пропадают при поиске); строки — под фильтром и в своём порядке.
  const sections = useMemo(() => {
    if (!data) return []
    const found = new Map(filterTree(data, filter).map((view) => [view.section.id, view.instruments]))
    return data.sections.map((section) => {
      const fullOrder = applyOrder(
        section.instruments.map((item) => item.id),
        order.sections?.[section.id],
      )
      const position = new Map(fullOrder.map((id, index) => [id, index]))
      const rows = (found.get(section.id) ?? [])
        .map(instrumentRow)
        .sort((a, b) => (position.get(a.id) ?? 0) - (position.get(b.id) ?? 0))
      return { section, rows, fullOrder }
    })
  }, [data, filter, order.sections])
  const activeTab = tab ?? sections[0]?.section.id ?? null

  return (
    <section className="flex flex-col gap-6" aria-busy={loading}>
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h1 className="text-2xl font-semibold tracking-tight">Инвентарь данных</h1>
          {data && (
            <p className="text-sm text-muted-foreground">
              Свежесть считается от {formatDateWithWeekday(data.expected_date)} — последнего завершённого торгового
              дня. Проверено {formatDateTime(data.generated_at)}.
            </p>
          )}
        </div>
        <Button variant="outline" onClick={refresh} disabled={loading}>
          <RefreshCw className={cn(loading && 'animate-spin')} />
          {loading ? 'Проверяю…' : 'Проверить снова'}
        </Button>
      </header>

      {error && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4 text-sm" role="alert">
          <p className="font-medium text-destructive">Не удалось получить инвентарь: {error}</p>
          <p className="mt-1 text-muted-foreground">
            Запущен ли API? <code className="font-mono">uv run python -m app</code>
          </p>
        </div>
      )}

      {!data && !error && <LoadingTable />}

      {data && (
        <div className={cn('flex flex-col gap-4 transition-opacity', loading && 'opacity-60')}>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-muted-foreground tabular-nums">
            <span>
              Наборов: <span className="font-medium text-foreground">{formatNumber(data.datasets.length)}</span>
            </span>
            {STATUS_ORDER.filter((status) => counts[status] > 0).map((status) => (
              <span key={status} className="inline-flex items-center gap-1.5">
                <StatusIcon status={status} />
                {STATUS[status].many}: {formatNumber(counts[status])}
              </span>
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <InputGroup className="w-full sm:w-80">
              <InputGroupAddon>
                <Search />
              </InputGroupAddon>
              <InputGroupInput
                type="search"
                placeholder="Инструмент, тикер или файл"
                aria-label="Поиск по инструменту, тикеру или файлу"
                value={filter.query}
                onChange={(event) => setFilter((current) => ({ ...current, query: event.target.value }))}
              />
            </InputGroup>
            <StatusFilter
              counts={counts}
              selected={filter.statuses}
              onChange={(statuses) => setFilter((current) => ({ ...current, statuses }))}
            />
            {filtering && (
              <Button variant="ghost" size="sm" onClick={() => setFilter(NO_FILTER)}>
                Сбросить
                <X />
              </Button>
            )}
          </div>

          {order.error && (
            <p className="rounded-lg border border-status-warning/50 bg-status-warning/10 px-3 py-2 text-sm" role="status">
              {order.error}
            </p>
          )}

          {activeTab && (
            <Tabs value={activeTab} onValueChange={(value) => setTab(String(value))} className="gap-3">
              <TabsList>
                {sections.map(({ section, rows }) => (
                  <TabsTrigger key={section.id} value={section.id} className="flex-none px-2 sm:px-3" title={section.description}>
                    {section.title}
                    <Badge variant="secondary" className="tabular-nums">
                      {formatNumber(rows.length)}
                    </Badge>
                  </TabsTrigger>
                ))}
              </TabsList>
              {sections.map(({ section, rows, fullOrder }) => (
                // keepMounted: сортировка и страница вкладки сохраняются при переключении.
                <TabsContent key={section.id} value={section.id} keepMounted>
                  <InstrumentTable
                    rows={rows}
                    order={order.sections === null ? null : fullOrder}
                    onReorder={(ids) => order.save(section.id, ids)}
                    expectedDate={data.expected_date}
                    filterKey={filterKey}
                    filtering={filtering}
                    columnVisibility={columnVisibility}
                    onColumnVisibilityChange={setColumnVisibility}
                  />
                </TabsContent>
              ))}
            </Tabs>
          )}
        </div>
      )}
    </section>
  )
}

function LoadingTable() {
  return (
    <div className="flex flex-col gap-3" aria-label="Сканирую data/…">
      <Skeleton className="h-8 w-full sm:w-80" />
      <Skeleton className="h-8 w-64" />
      <div className="flex flex-col gap-2 rounded-lg border p-3">
        {Array.from({ length: 8 }, (_, index) => (
          <Skeleton key={index} className="h-9 w-full" />
        ))}
      </div>
    </div>
  )
}
