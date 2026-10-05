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

export interface HTTPValidationError {
  detail: ValidationError[];
}

/** Инструмент (нефть, Сбербанк) или тематическая группа раздела «Общее». */
export interface Instrument {
  id: string;
  title: string;
  subtitle: string;
  datasets: string[];
}

/** Порядок инструментов: id раздела → id инструментов сверху вниз. */
export interface InstrumentOrder {
  sections: Record<string, string[]>;
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

export interface ValidationError {
  loc: (string | number)[];
  msg: string;
  type: string;
  input: unknown;
  ctx: Record<string, unknown>;
}

/** GET-пути API и тип ответа: apiGet('/api/data/status') вернёт InventoryReport. */
export interface ApiGet {
  "/api/health": Record<string, string>;
  "/api/data/status": InventoryReport;
  "/api/settings/instrument-order": InstrumentOrder;
}

/** PATCH-пути API: тело запроса и тип ответа. */
export interface ApiPatch {
  "/api/settings/instrument-order": { body: InstrumentOrder; response: InstrumentOrder };
}
