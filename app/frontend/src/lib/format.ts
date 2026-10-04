// Форматирование для интерфейса: русская локаль, время — по Москве, как в данных биржи.

const numberFormat = new Intl.NumberFormat('ru-RU')
const compactFormat = new Intl.NumberFormat('ru-RU', { notation: 'compact', maximumFractionDigits: 1 })
const dateTimeFormat = new Intl.DateTimeFormat('ru-RU', {
  timeZone: 'Europe/Moscow',
  day: '2-digit',
  month: '2-digit',
  year: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
})
const WEEKDAYS = ['вс', 'пн', 'вт', 'ср', 'чт', 'пт', 'сб']
const BYTE_UNITS = ['Б', 'КБ', 'МБ', 'ГБ', 'ТБ']

export const DAY_MS = 86_400_000

/** Дата ISO «2026-10-02» → полночь UTC. Дата без времени: часовые пояса её не сдвигают. */
export function parseDate(iso: string): Date {
  const [year, month, day] = iso.split('-').map(Number)
  return new Date(Date.UTC(year, month - 1, day))
}

/** «2026-10-02» → «02.10.2026». */
export function formatDate(iso: string): string {
  const [year, month, day] = iso.split('-')
  return `${day}.${month}.${year}`
}

/** «2026-10-02» → «пт 02.10.2026». */
export function formatDateWithWeekday(iso: string): string {
  return `${WEEKDAYS[parseDate(iso).getUTCDay()]} ${formatDate(iso)}`
}

/** Момент со смещением «2026-10-04T21:45:00+03:00» → «04.10.2026, 21:45» по Москве. */
export function formatDateTime(iso: string): string {
  return dateTimeFormat.format(new Date(iso))
}

export function formatNumber(value: number): string {
  return numberFormat.format(value)
}

const COMPACT_FROM = 100_000

/** До 100 000 — точно, дальше компактно: «3,1 млн». */
export function formatCount(value: number): string {
  return value < COMPACT_FROM ? numberFormat.format(value) : compactFormat.format(value)
}

/** Число со словом: «32 581 строка», «3,1 млн строк» — после «тыс.» и «млн» всегда родительный падеж. */
export function formatCountNoun(value: number, forms: readonly [string, string, string]): string {
  return `${formatCount(value)} ${value < COMPACT_FROM ? plural(value, forms) : forms[2]}`
}

/** Размер файла: «82 Б», «26,2 КБ», «2,5 ГБ». */
export function formatBytes(size: number): string {
  let value = size
  let unit = 0
  while (value >= 1024 && unit < BYTE_UNITS.length - 1) {
    value /= 1024
    unit += 1
  }
  const digits = unit === 0 || value >= 100 ? 0 : 1
  return `${value.toLocaleString('ru-RU', { maximumFractionDigits: digits })} ${BYTE_UNITS[unit]}`
}

/** Русское множественное число: plural(5, ['день', 'дня', 'дней']) → «дней». */
export function plural(n: number, forms: readonly [string, string, string]): string {
  const n10 = n % 10
  const n100 = n % 100
  if (n10 === 1 && n100 !== 11) return forms[0]
  if (n10 >= 2 && n10 <= 4 && (n100 < 12 || n100 > 14)) return forms[1]
  return forms[2]
}

/** «2 торговых дня». */
export function tradingDays(n: number): string {
  return `${n} ${plural(n, ['торговый день', 'торговых дня', 'торговых дней'])}`
}
