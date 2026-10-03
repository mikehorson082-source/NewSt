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

### Бэктесты
```bash
# Проверка бэктестов
uv run pytest src/backtest
```


| Источник                     | Что берём                                                                                                                                                                                                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| backtesting-py               | Механику одного контракта: порядок бара (broker.next → strategy.next), обрезку данных, bracket SL/TP + SL первым, фил стопа max(open, stop) с гэпом, margin_available, комиссию fixed + relative, equity-лог, список их краевых тестов                                    |
| hse (Vol_Brazil, PhD)        | Тесты и методологию: future-perturbation на лукахед, референс-движок против numba (parity 1e-10), тесты издержек и портфельной агрегации, walk-forward конфиги MOEX, реальные комиссии, SPA/DM                                                                            |
| machine-learning-for-trading | Архитектуру и протокол: «спеки / движок / издержки» раздельно, ContractSpec (k = step_price/step, margin_pct), counterfactual k=1 как демонстрация ошибки, parity-подход «меняем один конфиг и меряем», cost-инструменты ch.18 (спред, impact, capacity), риск-слой ch.19 |