import { describe, expect, it } from 'vitest'
import {
  formatBytes,
  formatCount,
  formatDate,
  formatDateTime,
  formatDateWithWeekday,
  formatNumber,
  plural,
  tradingDays,
} from './format'

// Intl ставит неразрывные пробелы; для сравнения заменяем их обычными.
const spaces = (text: string) => text.replace(/\s/g, ' ')

describe('даты', () => {
  it('дата без времени не сдвигается часовым поясом', () => {
    expect(formatDate('2026-10-02')).toBe('02.10.2026')
    expect(formatDateWithWeekday('2026-10-02')).toBe('пт 02.10.2026')
    expect(formatDateWithWeekday('2026-10-04')).toBe('вс 04.10.2026')
  })

  it('момент показывается по Москве', () => {
    expect(formatDateTime('2026-10-04T18:45:00+00:00')).toBe('04.10.2026, 21:45')
  })
})

describe('числа', () => {
  it('разряды и компактная запись больших чисел', () => {
    expect(spaces(formatNumber(32579))).toBe('32 579')
    expect(spaces(formatCount(99_999))).toBe('99 999')
    expect(spaces(formatCount(3_052_595))).toBe('3,1 млн')
  })

  it('размеры файлов', () => {
    expect(spaces(formatBytes(82))).toBe('82 Б')
    expect(spaces(formatBytes(26_828))).toBe('26,2 КБ')
    expect(spaces(formatBytes(253_000))).toBe('247 КБ')
    expect(spaces(formatBytes(2.5 * 1024 ** 3))).toBe('2,5 ГБ')
  })
})

describe('множественное число', () => {
  it.each([
    [1, 'день'],
    [2, 'дня'],
    [5, 'дней'],
    [11, 'дней'],
    [14, 'дней'],
    [21, 'день'],
    [22, 'дня'],
    [112, 'дней'],
  ])('%i %s', (n, word) => {
    expect(plural(n, ['день', 'дня', 'дней'])).toBe(word)
  })

  it('торговые дни', () => {
    expect(tradingDays(1)).toBe('1 торговый день')
    expect(tradingDays(2)).toBe('2 торговых дня')
    expect(tradingDays(10)).toBe('10 торговых дней')
  })
})
