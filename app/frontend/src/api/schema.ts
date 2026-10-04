// Сгенерировано из схемы OpenAPI (app/api/contract.py). Руками не править:
// uv run python -m app.api.contract  (или npm run gen:api в app/frontend)

/** Одна строка инвентаря: файл справочника, папка рынка или ряд. */
export interface DatasetStatus {
  id: string;
  group: string;
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

/** Дополнительный факт о наборе: «Роллов: 53», «Рынков: 24». */
export interface Fact {
  label: string;
  value: number | string;
}

/** Группа строк инвентаря: справочники, минутки, склейки… */
export interface Group {
  id: string;
  title: string;
  description: string;
  files_label: string | null;
}

/** Весь инвентарь: группы, строки и сводка по статусам. */
export interface InventoryReport {
  generated_at: string;
  expected_date: string;
  data_root: string;
  groups: Group[];
  datasets: DatasetStatus[];
  summary: Record<"ok" | "lagging" | "stale" | "unchecked" | "missing" | "error", number>;
}

/** GET-пути API и тип ответа: apiGet('/api/data/status') вернёт InventoryReport. */
export interface ApiGet {
  "/api/health": Record<string, string>;
  "/api/data/status": InventoryReport;
}
