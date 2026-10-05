import DataCatalog from './pages/DataCatalog'

/** Оболочка пульта. Экраны «Эксперименты» и «Графики» появятся рядом с «Данными». */
export default function App() {
  return (
    <div className="min-h-svh">
      <header className="border-b bg-background/95 px-4 py-3">
        <div className="mx-auto flex max-w-7xl items-baseline gap-2.5">
          <span className="font-semibold">NewSt</span>
          <span className="text-muted-foreground">Кабинет исследователя</span>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 pt-6 pb-12">
        <DataCatalog />
      </main>
    </div>
  )
}
