import {
  columnVisibilityFeature,
  createPaginatedRowModel,
  createSortedRowModel,
  metaHelper,
  rowExpandingFeature,
  rowPaginationFeature,
  rowSortingFeature,
  tableFeatures,
} from '@tanstack/react-table'

export interface CatalogColumnMeta {
  label: string // имя столбца в меню «Столбцы»
  className?: string // ширина, выравнивание, скрытие на узком экране — и у заголовка, и у ячеек
}

export interface CatalogTableMeta {
  canReorder: boolean // свой порядок загружен и не включена сортировка по столбцу
  moveToEdge: (id: string, edge: 'top' | 'bottom') => void
}

// TanStack Table v9: подключаем только нужные возможности, остальное не попадёт в сборку.
// Фильтр (поиск, статусы) считается до таблицы — lib/inventory.ts, — таблица получает готовые строки.
export const features = tableFeatures({
  columnVisibilityFeature,
  rowExpandingFeature,
  rowPaginationFeature,
  rowSortingFeature,
  paginatedRowModel: createPaginatedRowModel(),
  sortedRowModel: createSortedRowModel(),
  columnMeta: metaHelper<CatalogColumnMeta>(),
  tableMeta: metaHelper<CatalogTableMeta>(),
})

export type CatalogFeatures = typeof features
