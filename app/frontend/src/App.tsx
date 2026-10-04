import DataCatalog from './pages/DataCatalog'

/** Оболочка пульта. Экраны «Эксперименты» и «Графики» появятся рядом с «Данными». */
export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <span className="topbar__brand">NewSt</span>
        <span className="topbar__title">Кабинет исследователя</span>
      </header>
      <main className="page">
        <DataCatalog />
      </main>
    </div>
  )
}
