// Сгенерировано из схемы OpenAPI (app/api/contract.py). Руками не править:
// uv run python -m app.api.contract  (или npm run gen:api в app/frontend)

/** Набор данных: папка контрактов, файл, срез справочника по инструменту. */
export interface DatasetStatus {
  id: string;
  kind: string;
  title: string;
  description: string;
  path: string;
  status: "ok" | "lagging" | "stale" | "unchecked" | "missing" | "error";
  files: number;
  rows: number | null;
  first_date: string | null;
  last_date: string | null;
  size_bytes: number;
  modified_at: string | null;
  lag_days: number | null;
  note: string | null;
  facts: Fact[];
}

/** Дополнительный факт о наборе: «Роллов: 53», «Контрактов: 77». */
export interface Fact {
  label: string;
  value: number | string;
}

/** Инструмент (нефть, Сбербанк) или тематическая группа раздела «Общее». */
export interface Instrument {
  id: string;
  title: string;
  subtitle: string;
  datasets: string[];
}

/** Весь инвентарь: разделы с инструментами, наборы и сводка по статусам. */
export interface InventoryReport {
  generated_at: string;
  expected_date: string;
  data_root: string;
  sections: Section[];
  datasets: DatasetStatus[];
  summary: Record<"ok" | "lagging" | "stale" | "unchecked" | "missing" | "error", number>;
}

/** Верхний уровень: Фьючерсы, Акции, Общее. */
export interface Section {
  id: string;
  title: string;
  description: string;
  instruments: Instrument[];
}

/** GET-пути API и тип ответа: apiGet('/api/data/status') вернёт InventoryReport. */
export interface ApiGet {
  "/api/health": Record<string, string>;
  "/api/data/status": InventoryReport;
}
