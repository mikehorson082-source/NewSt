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
  rule: string | null;
  facts: Fact[];
  monthly: number[];
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

/** Кусок лога запуска с позиции offset. */
export interface LogChunk {
  text: string;
  offset: number;
  complete: boolean;
}

/** Экран «Загрузка»: шаги, порядок «Обновить всё», журнал запусков и чужая загрузка. */
export interface PipelineState {
  steps: Step[];
  update_all: string[];
  runs: Run[];
  busy: string | null;
  runs_dir: string;
}

/** Запуск из пульта: один шаг или «Обновить всё». */
export interface Run {
  id: string;
  title: string;
  kind: "data" | "check";
  status: "running" | "ok" | "error" | "stopped" | "interrupted";
  started_at: string;
  finished_at: string | null;
  stop_requested: boolean;
  steps: StepResult[];
}

/** Что запустить: id шагов (порядок — всегда как в реестре шагов). */
export interface RunRequest {
  steps: string[];
}

/** Верхний уровень: Фьючерсы, Акции, Общее. */
export interface Section {
  id: string;
  title: string;
  description: string;
  instruments: Instrument[];
}

/** Шаг конвейера: что запускается, куда пишет и от чего зависит. */
export interface Step {
  id: string;
  title: string;
  description: string;
  kind: "data" | "check";
  args: string[];
  command: string;
  writes: string[];
  needs: string[];
}

/** Итог шага в одном запуске. */
export interface StepResult {
  step: string;
  status: "pending" | "running" | "ok" | "error" | "skipped" | "stopped";
  started_at: string | null;
  finished_at: string | null;
  exit_code: number | null;
  note: string | null;
}

/** Какой запуск остановить после текущего шага. */
export interface StopRequest {
  run_id: string;
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
  "/api/pipeline": PipelineState;
  "/api/pipeline/log": LogChunk;
}

/** POST-пути API: тело запроса и тип ответа. */
export interface ApiPost {
  "/api/pipeline/runs": { body: RunRequest; response: Run };
  "/api/pipeline/stop": { body: StopRequest; response: Run };
}

/** PATCH-пути API: тело запроса и тип ответа. */
export interface ApiPatch {
  "/api/settings/instrument-order": { body: InstrumentOrder; response: InstrumentOrder };
}
