import { describe, expect, it } from 'vitest'
import type { DatasetStatus, InventoryReport } from '../api/schema'
import { countByStatus, coverage, expectedOffset, filterTree, timeAxis, yearLabelStep } from './inventory'

function row(overrides: Partial<DatasetStatus>): DatasetStatus {
  return {
    id: 'raw/futures/brent',
    kind: 'futures_minutes',
    title: 'Минутные свечи',
    description: 'файл на контракт',
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
const gold = row({ id: 'raw/futures/gold', path: 'raw/futures/gold', status: 'stale' })
const clearing = row({
  id: 'reference/rates_moex_clearing.parquet',
  kind: 'reference',
  title: 'rates_moex_clearing',
  description: 'Курс клиринга MOEX USD/RUB',
  path: 'reference/rates_moex_clearing.parquet',
})

const report: InventoryReport = {
  generated_at: '2026-01-05T10:00:00+03:00',
  expected_date: '2025-12-31',
  data_root: '/data',
  sections: [
    {
      id: 'futures',
      title: 'Фьючерсы',
      description: '',
      instruments: [
        { id: 'futures/brent', title: 'Нефть Brent', subtitle: 'brent · BR · сырьё', datasets: [brent.id, clearing.id] },
        { id: 'futures/gold', title: 'Золото', subtitle: 'gold · GOLD · сырьё', datasets: [gold.id, clearing.id] },
      ],
    },
    {
      id: 'common',
      title: 'Общее',
      description: '',
      instruments: [{ id: 'common/reference', title: 'Справочники', subtitle: '', datasets: [clearing.id] }],
    },
  ],
  datasets: [brent, gold, clearing],
  summary: { ok: 2, lagging: 0, stale: 1, unchecked: 0, missing: 0, error: 0 },
}

const titles = (filter: { status: DatasetStatus['status'] | null; query: string }) =>
  filterTree(report, filter).map((view) => [
    view.section.id,
    view.instruments.map((item) => `${item.instrument.title}: ${item.rows.map((r) => r.title).join(', ')}`),
  ])

describe('фильтр по инструментам', () => {
  it('без фильтра — все разделы и общий набор у каждого инструмента', () => {
    expect(titles({ status: null, query: '' })).toEqual([
      ['futures', ['Нефть Brent: Минутные свечи, rates_moex_clearing', 'Золото: Минутные свечи, rates_moex_clearing']],
      ['common', ['Справочники: rates_moex_clearing']],
    ])
  })

  it('совпало название инструмента — видны все его наборы', () => {
    expect(titles({ status: null, query: 'нефть' })).toEqual([
      ['futures', ['Нефть Brent: Минутные свечи, rates_moex_clearing']],
    ])
    expect(titles({ status: null, query: 'GOLD' })).toEqual([['futures', ['Золото: Минутные свечи, rates_moex_clearing']]])
  })

  it('иначе ищется по наборам: остаются только совпавшие', () => {
    expect(titles({ status: null, query: 'клиринг' })).toEqual([
      ['futures', ['Нефть Brent: rates_moex_clearing', 'Золото: rates_moex_clearing']],
      ['common', ['Справочники: rates_moex_clearing']],
    ])
  })

  it('статус скрывает инструменты и разделы без таких наборов', () => {
    expect(titles({ status: 'stale', query: '' })).toEqual([['futures', ['Золото: Минутные свечи']]])
  })

  it('считает наборы по статусам', () => {
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
