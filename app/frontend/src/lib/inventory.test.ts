import { describe, expect, it } from 'vitest'
import type { DatasetStatus, InventoryReport } from '../api/schema'
import { countByStatus, coverage, expectedOffset, groupRows, matches, timeAxis, yearLabelStep } from './inventory'

function row(overrides: Partial<DatasetStatus>): DatasetStatus {
  return {
    id: 'raw/futures/brent',
    group: 'futures_minutes',
    title: 'brent',
    description: 'Нефть Brent',
    path: 'raw/futures/brent',
    status: 'ok',
    files: 1,
    rows: 10,
    first_date: '2024-01-01',
    last_date: '2025-12-31',
    size_bytes: 100,
    modified_at: null,
    lag_days: 0,
    note: null,
    facts: [],
    ...overrides,
  }
}

const brent = row({})
const gold = row({ id: 'raw/futures/gold', title: 'gold', description: 'Золото', path: 'raw/futures/gold', status: 'stale' })
const margins = row({
  id: 'reference/margin_rates.parquet',
  group: 'reference',
  title: 'margin_rates',
  description: 'Минимальные ставки ГО',
  path: 'reference/margin_rates.parquet',
})

const report: InventoryReport = {
  generated_at: '2026-01-05T10:00:00+03:00',
  expected_date: '2025-12-31',
  data_root: '/data',
  groups: [
    { id: 'reference', title: 'Справочники', description: '', files_label: null },
    { id: 'futures_minutes', title: 'Минутные свечи', description: '', files_label: 'Контрактов' },
    { id: 'algopack', title: 'Algopack', description: '', files_label: 'Файлов' },
  ],
  datasets: [margins, brent, gold],
  summary: { ok: 2, lagging: 0, stale: 1, unchecked: 0, missing: 0, error: 0 },
}

describe('фильтр', () => {
  it('ищет по имени, описанию и пути без учёта регистра', () => {
    expect(matches(gold, { status: null, query: 'золото' })).toBe(true)
    expect(matches(gold, { status: null, query: 'FUTURES/GOLD' })).toBe(true)
    expect(matches(gold, { status: null, query: 'brent' })).toBe(false)
    expect(matches(gold, { status: 'ok', query: '' })).toBe(false)
  })

  it('группы идут в порядке отчёта, пустые скрываются', () => {
    const all = groupRows(report, { status: null, query: '' })
    expect(all.map((view) => view.group.id)).toEqual(['reference', 'futures_minutes'])
    const stale = groupRows(report, { status: 'stale', query: '' })
    expect(stale.map((view) => view.rows.map((item) => item.title))).toEqual([['gold']])
  })

  it('считает строки по статусам', () => {
    expect(countByStatus(report.datasets)).toEqual({ ok: 2, lagging: 0, stale: 1, unchecked: 0, missing: 0, error: 0 })
  })
})

describe('ось времени', () => {
  // 2024-01-01 … 2025-12-31: ровно два года, 731 день (2024 високосный).
  const axis = timeAxis([brent], '2025-12-31')!

  it('годы отмечены 1 января внутри оси', () => {
    expect(axis.years.map((tick) => tick.year)).toEqual([2025])
    expect(axis.years[0].offset).toBeCloseTo((366 / 731) * 100)
  })

  it('полоса покрывает дни целиком: от первого дня до конца последнего', () => {
    expect(coverage('2024-01-01', '2025-12-31', axis)).toEqual({ left: 0, width: 100 })
    const half = coverage('2025-01-01', '2025-12-31', axis)
    expect(half.left).toBeCloseTo((366 / 731) * 100)
    expect(half.width).toBeCloseTo((365 / 731) * 100)
  })

  it('ось доходит до ожидаемой даты, даже если данные кончились раньше', () => {
    const behind = timeAxis([row({ last_date: '2025-06-30' })], '2025-12-31')!
    expect(expectedOffset('2025-12-31', behind)).toBeCloseTo(100)
    expect(coverage('2024-01-01', '2025-06-30', behind).width).toBeLessThan(100)
  })

  it('без дат оси нет', () => {
    expect(timeAxis([row({ first_date: null, last_date: null })], '2025-12-31')).toBeNull()
  })

  it('подписи лет прореживаются', () => {
    const long = timeAxis([row({ first_date: '2010-01-07' })], '2026-10-02')!
    expect(long.years).toHaveLength(16)
    expect(yearLabelStep(long)).toBe(3)
    expect(yearLabelStep(axis)).toBe(1)
  })
})
