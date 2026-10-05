import { describe, expect, it } from 'vitest'
import { applyOrder, moveToEdge, moveVisible } from './order'

const catalog = ['si', 'brent', 'gold', 'cny']

describe('сохранённый порядок', () => {
  it('без сохранённого — порядок каталога', () => {
    expect(applyOrder(catalog, undefined)).toEqual(catalog)
    expect(applyOrder(catalog, [])).toEqual(catalog)
  })

  it('сохранённые сверху, новые инструменты — следом в порядке каталога', () => {
    expect(applyOrder(catalog, ['gold', 'si'])).toEqual(['gold', 'si', 'brent', 'cny'])
  })

  it('исчезнувшие из каталога id пропускаются', () => {
    expect(applyOrder(catalog, ['old', 'cny'])).toEqual(['cny', 'si', 'brent', 'gold'])
  })
})

describe('перетаскивание', () => {
  it('все строки видны — обычная перестановка', () => {
    expect(moveVisible(catalog, catalog, 'gold', 'si')).toEqual(['gold', 'si', 'brent', 'cny'])
    expect(moveVisible(catalog, catalog, 'si', 'gold')).toEqual(['brent', 'gold', 'si', 'cny'])
  })

  it('под фильтром скрытые строки остаются на своих местах', () => {
    // Видны si и gold; gold перетащили на место si.
    expect(moveVisible(catalog, ['si', 'gold'], 'gold', 'si')).toEqual(['gold', 'brent', 'si', 'cny'])
  })

  it('бросили на себя или мимо — порядок не меняется', () => {
    expect(moveVisible(catalog, catalog, 'si', 'si')).toEqual(catalog)
    expect(moveVisible(catalog, ['si'], 'si', 'gold')).toEqual(catalog)
  })

  it('в начало и в конец раздела', () => {
    expect(moveToEdge(catalog, 'gold', 'top')).toEqual(['gold', 'si', 'brent', 'cny'])
    expect(moveToEdge(catalog, 'si', 'bottom')).toEqual(['brent', 'gold', 'cny', 'si'])
  })
})
