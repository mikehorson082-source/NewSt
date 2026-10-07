import type { DatasetStatus, Instrument, InventoryReport, Section } from '../api/schema'
import { DAY_MS, parseDate } from './format'
import { STATUS_ORDER, type Status } from './status'

export interface Filter {
  statuses: Status[] // пусто — все статусы
  query: string
}

export interface InstrumentView {
  instrument: Instrument
  rows: DatasetStatus[]
}

export interface SectionView {
  section: Section
  instruments: InstrumentView[]
}

function includes(texts: string[], query: string): boolean {
  return texts.some((text) => text.toLowerCase().includes(query))
}

/**
 * Разделы и инструменты под фильтр.
 * Поиск: совпало название инструмента («нефть», «SBER») — видны все его наборы,
 * иначе — только наборы, в которых нашлось слово. Статус фильтрует наборы.
 * Инструменты и разделы без наборов скрываются.
 */
export function filterTree(report: InventoryReport, filter: Filter): SectionView[] {
  const index = new Map(report.datasets.map((row) => [row.id, row]))
  const query = filter.query.trim().toLowerCase()
  return report.sections
    .map((section) => ({
      section,
      instruments: section.instruments.flatMap((instrument) => {
        const byName = query === '' || includes([instrument.title, instrument.subtitle], query)
        const rows = instrument.datasets
          .map((id) => index.get(id))
          .filter((row): row is DatasetStatus => row !== undefined)
          .filter((row) => filter.statuses.length === 0 || filter.statuses.includes(row.status))
          .filter((row) => byName || includes([row.title, row.description, row.path], query))
        return rows.length > 0 ? [{ instrument, rows }] : []
      }),
    }))
    .filter((view) => view.instruments.length > 0)
}

export function countByStatus(rows: DatasetStatus[]): Record<Status, number> {
  const counts: Record<Status, number> = { ok: 0, lagging: 0, stale: 0, unchecked: 0, missing: 0, error: 0 }
  for (const row of rows) counts[row.status] += 1
  return counts
}

/** Строка таблицы инструментов: сводка по его наборам (под фильтром) для столбцов и сортировки. */
export interface InstrumentRow {
  id: string
  instrument: Instrument
  datasets: DatasetStatus[]
  worst: Status // худший статус наборов: по нему сортируется столбец «Статус»
  firstDate: string | null // самая ранняя дата среди наборов
  lastDate: string | null // самая поздняя
  totalRows: number | null // null — ни у одного набора строки не считаются
  sizeBytes: number
}

export function instrumentRow({ instrument, rows }: InstrumentView): InstrumentRow {
  const firsts = rows.flatMap((row) => (row.first_date ? [row.first_date] : []))
  const lasts = rows.flatMap((row) => (row.last_date ? [row.last_date] : []))
  const counted = rows.filter((row) => row.rows !== null)
  return {
    id: instrument.id,
    instrument,
    datasets: rows,
    // Даты ISO сравниваются как строки: «2024-01-05» < «2024-10-01».
    worst: STATUS_ORDER.find((status) => rows.some((row) => row.status === status)) ?? 'unchecked',
    firstDate: firsts.length > 0 ? firsts.reduce((a, b) => (b < a ? b : a)) : null,
    lastDate: lasts.length > 0 ? lasts.reduce((a, b) => (b > a ? b : a)) : null,
    totalRows: counted.length > 0 ? counted.reduce((sum, row) => sum + (row.rows ?? 0), 0) : null,
    sizeBytes: rows.reduce((sum, row) => sum + row.size_bytes, 0),
  }
}

export interface Axis {
  start: number // мс, полночь UTC первой даты
  end: number // мс, конец последнего дня
  years: { year: number; offset: number }[] // 1 января каждого года внутри оси, % ширины
}

/** Общая ось времени инструмента: от самой ранней даты до последней (не раньше ожидаемой даты). */
export function timeAxis(rows: DatasetStatus[], expectedDate: string): Axis | null {
  const firsts: number[] = []
  const lasts: number[] = []
  for (const row of rows) {
    if (row.first_date && row.last_date) {
      firsts.push(parseDate(row.first_date).getTime())
      lasts.push(parseDate(row.last_date).getTime())
    }
  }
  if (firsts.length === 0) return null
  const start = Math.min(...firsts)
  const end = Math.max(...lasts, parseDate(expectedDate).getTime()) + DAY_MS
  const years: Axis['years'] = []
  for (let year = new Date(start).getUTCFullYear() + 1; Date.UTC(year, 0, 1) < end; year += 1) {
    years.push({ year, offset: offset(Date.UTC(year, 0, 1), start, end) })
  }
  return { start, end, years }
}

/** Будни (пн–пт) среди дней [from, to); границы — полночь UTC. */
function weekdays(from: number, to: number): number {
  let count = 0
  for (let day = from; day < to; day += DAY_MS) {
    const weekday = new Date(day).getUTCDay()
    if (weekday !== 0 && weekday !== 6) count += 1
  }
  return count
}

function offset(time: number, start: number, end: number): number {
  return ((time - start) / (end - start)) * 100
}

/** Полоса покрытия на оси: левый край и ширина в % (последний день входит целиком). */
export function coverage(first: string, last: string, axis: Axis): { left: number; width: number } {
  const from = parseDate(first).getTime()
  const to = parseDate(last).getTime() + DAY_MS
  return { left: offset(from, axis.start, axis.end), width: ((to - from) / (axis.end - axis.start)) * 100 }
}

/** Месяц графика на оси: края в % и высота 0…1. */
export interface MonthStep {
  left: number
  right: number
  value: number
}

/**
 * Ступени графика «сколько данных» на оси: по ступени на месяц из monthly.
 * Высота — строк за будний день (а не за месяц): неполные первый и последний
 * месяцы и короткий февраль не проваливаются и не подскакивают. Будни, а не
 * календарные дни: хвост «1–2 октября» — одни будни, а в полном месяце есть
 * выходные без торгов. 1 — самый плотный месяц набора, 0 — месяц без данных.
 * Пример: данные с пн 18.03.2024, в марте 1 000 строк за 10 будней, в апреле
 * 2 200 за 22 — у обоих по 100 строк за будний день: ступени одной высоты.
 */
export function monthSteps(first: string, last: string, monthly: number[], axis: Axis): MonthStep[] {
  const from = parseDate(first).getTime()
  const to = parseDate(last).getTime() + DAY_MS
  const start = parseDate(first)
  const spans = monthly.map((rows, i) => {
    const monthStart = Date.UTC(start.getUTCFullYear(), start.getUTCMonth() + i, 1)
    const monthEnd = Date.UTC(start.getUTCFullYear(), start.getUTCMonth() + i + 1, 1)
    const lo = Math.max(monthStart, from)
    const hi = Math.min(monthEnd, to)
    return { lo, hi, density: rows / Math.max(1, weekdays(lo, hi)) }
  })
  const peak = Math.max(0, ...spans.map((span) => span.density))
  return spans.map(({ lo, hi, density }) => ({
    left: offset(lo, axis.start, axis.end),
    right: offset(hi, axis.start, axis.end),
    value: peak > 0 ? density / peak : 0,
  }))
}

/** Где на оси кончается ожидаемая дата, %: полоса, не дотянувшая до отметки, отстаёт. */
export function expectedOffset(expectedDate: string, axis: Axis): number {
  return offset(parseDate(expectedDate).getTime() + DAY_MS, axis.start, axis.end)
}

/** Подписывать каждый step-й год, чтобы подписи не налезали друг на друга. */
export function yearLabelStep(axis: Axis, maxLabels = 7): number {
  return Math.max(1, Math.ceil(axis.years.length / maxLabels))
}
