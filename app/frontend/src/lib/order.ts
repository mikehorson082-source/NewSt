// Свой порядок инструментов: id сверху вниз по разделу. Задаётся перетаскиванием,
// хранится в app/settings/instrument_order.json (API /api/settings/instrument-order).

/** Ids в сохранённом порядке; не упомянутые в нём (новые в каталоге) — следом, в порядке каталога. */
export function applyOrder(ids: readonly string[], saved: readonly string[] | undefined): string[] {
  if (!saved?.length) return [...ids]
  const present = new Set(ids)
  const head = saved.filter((id) => present.has(id))
  const placed = new Set(head)
  return [...head, ...ids.filter((id) => !placed.has(id))]
}

/**
 * Перетаскивание среди видимых строк (под фильтром, на странице): `from` встаёт на место `to`.
 * Видимые строки делят между собой те же места полного списка, что и раньше, скрытые не двигаются.
 * Пример: порядок A b C d, видны A и C; C перетащили на A → C b A d.
 */
export function moveVisible(order: readonly string[], visible: readonly string[], from: string, to: string): string[] {
  const start = visible.indexOf(from)
  const end = visible.indexOf(to)
  if (start < 0 || end < 0 || start === end) return [...order]
  const moved = [...visible]
  moved.splice(end, 0, ...moved.splice(start, 1))
  const shown = new Set(visible)
  let next = 0
  return order.map((id) => (shown.has(id) ? moved[next++] : id))
}

/** Инструмент — в начало или в конец всего раздела (на другую страницу перетащить нельзя). */
export function moveToEdge(order: readonly string[], id: string, edge: 'top' | 'bottom'): string[] {
  const rest = order.filter((item) => item !== id)
  return edge === 'top' ? [id, ...rest] : [...rest, id]
}
