import { cn } from 'cn'
import { useEffect, useState } from 'react'
import DataCatalog from './pages/DataCatalog'
import DataLoad from './pages/DataLoad'

// Экраны пульта; адрес экрана — в «#» ссылки: его можно добавить в закладки, «Назад» работает.
const PAGES = [
  { hash: '#/', title: 'Данные', page: DataCatalog },
  { hash: '#/load', title: 'Загрузка', page: DataLoad },
] as const

function currentHash(): string {
  return PAGES.some((page) => page.hash === window.location.hash) ? window.location.hash : PAGES[0].hash
}

/** Оболочка пульта. Экраны «Эксперименты» и «Графики» появятся рядом с «Данными». */
export default function App() {
  const [hash, setHash] = useState(currentHash)
  useEffect(() => {
    const onChange = () => setHash(currentHash())
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  const Page = PAGES.find((page) => page.hash === hash)?.page ?? DataCatalog

  return (
    <div className="min-h-svh">
      <header className="border-b bg-background/95 px-4 py-2.5">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2">
          <div className="flex items-baseline gap-2.5">
            <span className="font-semibold">NewSt</span>
            <span className="hidden text-muted-foreground sm:inline">Кабинет исследователя</span>
          </div>
          <nav className="flex gap-1 text-sm" aria-label="Экраны">
            {PAGES.map((page) => (
              <a
                key={page.hash}
                href={page.hash}
                aria-current={page.hash === hash ? 'page' : undefined}
                className={cn(
                  'rounded-md px-2.5 py-1 text-muted-foreground outline-none hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50',
                  page.hash === hash && 'bg-muted font-medium text-foreground',
                )}
              >
                {page.title}
              </a>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 pt-6 pb-12">
        <Page />
      </main>
    </div>
  )
}
