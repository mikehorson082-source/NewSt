import { describe, expect, it } from 'vitest'
import type { DatasetStatus, InventoryReport } from '../api/schema'
import {
  countByStatus,
  coverage,
  expectedOffset,
  filterTree,
  instrumentRow,
  monthSteps,
  timeAxis,
  yearLabelStep,
  type Filter,
} from './inventory'

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
    rule: null,
    facts: [],
    monthly: [],
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

const titles = (filter: Filter) =>
  filterTree(report, filter).map((view) => [
    view.section.id,
    view.instruments.map((item) => `${item.instrument.title}: ${item.rows.map((r) => r.title).join(', ')}`),
  ])

describe('фильтр по инструментам', () => {
  it('без фильтра — все разделы и общий набор у каждого инструмента', () => {
    expect(titles({ statuses: [], query: '' })).toEqual([
      ['futures', ['Нефть Brent: Минутные свечи, rates_moex_clearing', 'Золото: Минутные свечи, rates_moex_clearing']],
      ['common', ['Справочники: rates_moex_clearing']],
    ])
  })

  it('совпало название инструмента — видны все его наборы', () => {
    expect(titles({ statuses: [], query: 'нефть' })).toEqual([
      ['futures', ['Нефть Brent: Минутные свечи, rates_moex_clearing']],
    ])
    expect(titles({ statuses: [], query: 'GOLD' })).toEqual([['futures', ['Золото: Минутные свечи, rates_moex_clearing']]])
  })

  it('иначе ищется по наборам: остаются только совпавшие', () => {
    expect(titles({ statuses: [], query: 'клиринг' })).toEqual([
      ['futures', ['Нефть Brent: rates_moex_clearing', 'Золото: rates_moex_clearing']],
      ['common', ['Справочники: rates_moex_clearing']],
    ])
  })

  it('статус скрывает инструменты и разделы без таких наборов', () => {
    expect(titles({ statuses: ['stale'], query: '' })).toEqual([['futures', ['Золото: Минутные свечи']]])
  })

  it('можно выбрать несколько статусов', () => {
    expect(titles({ statuses: ['stale', 'ok'], query: 'золото' })).toEqual([
      ['futures', ['Золото: Минутные свечи, rates_moex_clearing']],
    ])
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

describe('строка инструмента', () => {
  it('сводка по наборам: худший статус, крайние даты, суммы', () => {
    const view = filterTree(report, { statuses: [], query: 'золото' })[0].instruments[0]
    const summary = instrumentRow({
      ...view,
      rows: [
        gold,
        { ...clearing, first_date: '2020-03-02', last_date: '2026-01-05', rows: null, size_bytes: 50 },
      ],
    })
    expect(summary).toMatchObject({
      id: 'futures/gold',
      worst: 'stale',
      firstDate: '2020-03-02',
      lastDate: '2026-01-05',
      totalRows: 10,
      sizeBytes: 150,
    })
  })

  it('без дат и строк — пусто, а не ноль', () => {
    const summary = instrumentRow({
      instrument: report.sections[1].instruments[0],
      rows: [row({ first_date: null, last_date: null, rows: null, status: 'unchecked' })],
    })
    expect(summary).toMatchObject({ worst: 'unchecked', firstDate: null, lastDate: null, totalRows: null })
  })
})

describe('график по месяцам', () => {
  const axis = { start: Date.UTC(2024, 2, 1), end: Date.UTC(2024, 5, 1), years: [] }

  it('высота — строки за будний день: неполный первый месяц не проваливается', () => {
    // Март с пн 18-го: 10 будней по 100 строк; апрель 22 будня по 100; май пуст.
    const steps = monthSteps('2024-03-18', '2024-05-31', [1000, 2200, 0], axis)
    expect(steps.map((step) => step.value)).toEqual([1, 1, 0])
  })

  it('хвост из одних будней не подскакивает над полным месяцем', () => {
    // Апрель 2024: 22 будня по 5 строк; 1–2 мая (ср, чт) — тоже по 5.
    const steps = monthSteps('2024-04-01', '2024-05-02', [110, 10], axis)
    expect(steps.map((step) => step.value)).toEqual([1, 1])
  })

  it('ступени стоят на оси от первой даты до конца последнего дня', () => {
    const steps = monthSteps('2024-03-18', '2024-05-31', [1000, 2200, 0], axis)
    expect(steps[0].left).toBeCloseTo((17 / 92) * 100)
    expect(steps[0].right).toBeCloseTo((31 / 92) * 100)
    expect(steps[2].right).toBeCloseTo(100)
  })

  it('без строк — все ступени нулевые', () => {
    expect(monthSteps('2024-03-01', '2024-04-30', [0, 0], axis).map((step) => step.value)).toEqual([0, 0])
  })
})
