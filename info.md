### Загрузка и обновление данных
```bash
# Справочники инструментов
uv run python src/data/download_reference.py --update

# Дневные свечи
uv run python src/data/download_futures_daily.py --update

# Минутные свечи
uv run python src/data/download_futures_min.py --update

# 5-минутный открытый интерес (FUTOI)
uv run python src/data/download_futoi.py --update
```

### Обработка и склейка
```bash
# Склейка всех фьючерсов в непрерывные серии
uv run python src/data/build_continuous_futures.py
```
