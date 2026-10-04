import type { DatasetStatus, Instrument, InventoryReport, Section } from '../api/schema'
import { DAY_MS, parseDate } from './format'
import type { Status } from './status'

export interface Filter {
  status: Status | null // null — все статусы
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
          .filter((row) => filter.status === null || row.status === filter.status)
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

function offset(time: number, start: number, end: number): number {
  return ((time - start) / (end - start)) * 100
}

/** Полоса покрытия на оси: левый край и ширина в % (последний день входит целиком). */
export function coverage(first: string, last: string, axis: Axis): { left: number; width: number } {
  const from = parseDate(first).getTime()
  const to = parseDate(last).getTime() + DAY_MS
  return { left: offset(from, axis.start, axis.end), width: ((to - from) / (axis.end - axis.start)) * 100 }
}

/** Где на оси кончается ожидаемая дата, %: полоса, не дотянувшая до отметки, отстаёт. */
export function expectedOffset(expectedDate: string, axis: Axis): number {
  return offset(parseDate(expectedDate).getTime() + DAY_MS, axis.start, axis.end)
}

/** Подписывать каждый step-й год, чтобы подписи не налезали друг на друга. */
export function yearLabelStep(axis: Axis, maxLabels = 7): number {
  return Math.max(1, Math.ceil(axis.years.length / maxLabels))
}
