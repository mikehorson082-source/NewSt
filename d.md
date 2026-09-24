# Исследования и статьи: торговые стратегии на фьючерсах MOEX

Собрано по результатам веб-поиска (Brave Search, DuckDuckGo).
Пункты, которые НЕ удалось верифицировать, помечены отдельно.

---

## 1. Smart-Lab — массовый тест 3811 стратегий StockSharp (главный источник)

ОРИГИНАЛ, автор Роман Миронов (MR Trade), 28.08.2026.
Библиотека: GitHub `StockSharp/AlgoTrading` (3811 стратегий). Автор портировал 53 стратегии на Python
и прогнал по 28 инструментам MOEX.

- https://smart-lab.ru/blog/1345567.php

### Условия эксперимента (подтверждено из статьи)
- Период: 01.01.2024 — 25.06.2026 (2.5 года)
- Инструменты: 28 (8 акций, 8 фьючерсов на них, 6 товарных, 2 валютных, 4 индексных)
- Данные: 5-мин бары, ресемпл в 1h/4h
- Прогонов: 1592 связки «стратегия × инструмент» (1497 дошли до метрик)
- Издержки акции: 0.05% на ногу
- Издержки фьючерсы: биржевой сбор + 0.45 ₽ за контракт на ногу
- Проскальзывание: НЕ моделировалось (цифры — оптимистичная граница)
- Период был падающим: Buy&Hold GAZP −38.5%, ROSN −48.3%, CHMF −59.7%, фьючерсы с контанго до −62.9%

### Ключевые результаты
- Медианная чистая доходность связки: −24.5%
- Доля прибыльных связок: 26.6%
- Медианный PF: 0.86; медианный Sharpe: −0.89; медианная просадка: −34.7%
- Положительную медиану по всем 28 инструментам имеют 9 стратегий из 51:
  - 0006 Tripple MA (5m): +36.0%, 70.6% прибыльных, PF 1.26
  - 0019 CCI Breakout (1h): +24.5%, 64.3%, PF 1.13
  - 1837 BW Wise Man 2 (4h): +20.0%, 60.7%, PF 1.23
  - 0432 MACD DMI (1h): +16.1%, 57.1%, PF 1.15
  - 0001 MA CrossOver (5m): +14.7%; 0332 Donchian+Hurst (4h): +10.2%
  - 0007 Keltner Breakout (5m): +5.4%; 0002 N-Day Breakout (5m): +4.1%; 0005 Donchian (5m): +4.1%

### Главный вывод: решает оборот, а не индикатор
- Корреляция Спирмена между числом сделок и доходностью: −0.64 (почти монотонная)
- Q1 (медиана 59 сделок): −4.4%, прибыльных 43.9%
- Q5 (медиана 2737 сделок): −88.7%, прибыльных 3.0%
- Разрыв валовой/чистой доходности: 4h = 5.5 п.п., 1h = 7.4 п.п., 5m = 10.5 п.п., 1m = 32.9 п.п.
- Комиссия съедает весь эдж на минутках.

### Проверка на шум (shuffle test)
Ряд циклически сдвигается по приращениям 200 раз, связь сигнала с будущим разрушается.
- Лучшая по доходности связка MA CrossOver × GAZP (+278.1%) статистически НЕ отличается от шума (+275.6%, p=0.415)
- Donchian × SV (+270.5%, Sharpe 3.67) хуже собственного шума (−13.7 п.п.)
- Реальный эдж: CCI Divergence × NG (перевес +231.6 п.п., p=0.010), CCI Breakout × NVTK_FUT (+151.3 п.п., p=0.000)
- Вся прибыль живёт в верхних 5% сделок: без них 7 из 8 связок уходят в минус
- Эдж не переносится между инструментами: ADX DI × NG +157%, но медиана по 28 инструментам −86.5%
- Затухание: лучшие связки в 2026 году уже в минусе (CCI Breakout × NVTK_FUT: 2024 +80.3%, 2025 +67.4%, 2026 −3.6%)
- Итог автора: до статуса «заслуживает проверки» дошли 2 стратегии — CCI Breakout и MACD DMI на 1h

## 2. Smart-Lab — трендовая на Si, форвард-тест

Автор тот же, продолжение исследования, 01.09.2026. Статья:
- https://smart-lab.ru/blog/1346872.php

- Бэктест с 2024: накопленная +263.8%, MaxDD 17.2%, Sharpe 2.1, PF 2.2
- ~1.2 сделки в день, удержание от часа до нескольких дней
- Инструмент: фьючерс на доллар (Si), стратегия с фильтрами (риск переоптимизации признан автором)
- Первый контрольный отрезок — месяц реальной торговли (~25–30 сделок), эксперимент продолжается
- В комментариях Op_Man показал сопоставимый бэктест трендовой на Si/CNY с 01.01.2024

## 3. MOEX официальное — стратегии использования фьючерсов

- «Стратегии использования фьючерсов на акции»:
  https://www.moex.com/a2086
- «Стратегии использования фьючерсов и опционов на Индекс РТС» (хедж, синтетический фьючерс
  на «второй эшелон», календарный спред, торговля волатильностью, арбитраж):
  https://www.moex.com/a2084
- «Индекс RVI. Фьючерс на волатильность. Торговые стратегии» (PDF):
  https://www.moex.com/ru/documents/8262
- Школа MOEX, курс «Стратегии торговли фьючерсами»:
  https://school.moex.com/courses/strategii-torgovli-fyuchersami
- Мастер-класс «Парный трейдинг»: https://www.moex.com/ru/events/1682
- Семинар «Парный трейдинг»: https://www.moex.com/e3180
- Календарные спреды: https://www.moex.com/s3578
- Вебинары срочного рынка (календарный спред, синтетическая облигация, парный трейдинг):
  https://www.moex.com/ru/events/13704
- Учебные торги фьючерсами/опционами: https://www.moex.com/s127
- ALGOPACK (данные MOEX для алготрейдинга): https://data.moex.com/products/algopack

## 4. Фьючерс на индекс RUONIA (новый инструмент, запуск 19.05.2026)

- MOEX, фьючерс на ставку RUONIA (стратегии, вероятность изменения ключевой ставки из котировок):
  https://www.moex.com/ru/derivatives/money/ruonia/
- MOEX, «Фьючерс на индекс RUONIA — новый инструмент для торговли рублевой ставкой»:
  https://www.moex.com/ru/ruonia
- MOEX, «Фьючерсы на RUONIA: реализация стратегии на ставке ЦБ»:
  https://www.moex.com/a4103
- MOEX пресс-релиз о запуске: https://www.moex.com/n100018
- MOEX, PDF «Фьючерсы на ставку RUONIA»: https://fs.moex.com/f/1143/fjuchers-na-stavku-ruonia.pdf
- MOEX, английская страница контракта: https://www.moex.com/msn/en-ruonia
- Ведомости (14.05.2026): https://www.vedomosti.ru/investments/news/2026/05/14/1197382-indeks-ruonia
- Коммерсантъ (14.05.2026): https://www.kommersant.ru/doc/8655451
- BCS Express, разбор нового фьючерса: https://bcs-express.ru/novosti-i-analitika/novyi-f-iuchers-na-ruonia-to-za-zver-i-kak-ego-priruchit
- Smart-Lab, словарь по RUONIA: https://smart-lab.ru/finansoviy-slovar/RUONIA
- Фьючерсы на процентные ставки (общее): https://www.moex.com/s3812

## 5. Фандинг / вечные фьючерсы (USDRUBF, CNYRUBF, IMOEXF и др.)

- Статистика фандинга USDRUBF и возможные торговые стратегии:
  https://marketpower.pro/post/statistika-fandinga-usdrubf-i-vozmozhnye-torgovye-strategii
- Фандинг USDRUBF в % годовых 2023–2026 помесячно:
  https://marketpower.pro/post/fanding-usdrubf-v-godovykh-2023-2026-gg-pomesiachno
- Анализ фандинга USDRUBF по данным Мосбиржи 2023–2025:
  https://marketpower.pro/post/analiz-fandinga-v-vechnom-fiucherse-usdrubf-po-dannym-mosbirzhi-s-2023-po-2025-pomesiachno
- Контанго-фандинг, ноябрь 2025 – январь 2026:
  https://marketpower.pro/post/analiz-kontango-fandinga-s-noiabria-2025-po-ianvar-2026-gg
- Сводная таблица фандинга вечных фьючерсов (USDRUBF 18%, IMOEXF 20%, GLDRUBF 25% и т.д.):
  https://smart-lab.ru/finansoviy-slovar/funding-eternal-futures
- MOEX, PDF «Фандинг и исполнение вечных фьючерсов на валюту» (авг. 2025):
  https://fs.moex.com/f/17840/prezentacija-vyhod-i-fanding-v-vechnyh-fjuchersa.pdf
- MOEX, вечные фьючерсы на валютные пары: https://www.moex.com/a8141
- MOEX, вечные фьючерсы на индекс (формула фандинга): https://www.moex.com/a8803
- MOEX, вечные фьючерсы (общее): https://www.moex.com/s3581
- MOEX, FAQ «Гид: вечные фьючерсы»:
  https://fs.moex.com/f/22748/faq-vechnye-fjuchersy-20082025.pdf
- Smart-Lab, вечные фьючерсы и фандинг на FORTS:
  https://smart-lab.ru/blog/1070068.php
- vc.ru, история вечных фьючерсов и арбитраж на неэффективностях:
  https://vc.ru/money/563441-kratkaya-istoriya-vkpb-vechnyh-fyuchersov-i-rasskaz-pro-arbitrazh-na-neeffektivnosti

## 6. ВШЭ — парный трейдинг и статистический арбитраж (открытый реестр ВКР)

- Статистический арбитраж на российском фондовом рынке (2016):
  https://www.hse.ru/edu/vkr/182209389
- Эффективность парного трейдинга на российском фондовом рынке:
  https://www.hse.ru/edu/vkr/153016501
- Парный трейдинг: проверка статарбитража среди секторов США:
  https://www.hse.ru/edu/vkr/468570936
- Эмпирический анализ стратегий парного трейдинга на крипторынке:
  https://www.hse.ru/edu/vkr/316604590
- ВКР: прогнозирование цены акций через новостной анализ + deep learning:
  https://www.hse.ru/edu/vkr/471399469
- Общий реестр ВКР ВШЭ: https://www.hse.ru/edu/vkr/

НЕ подтверждены по именам (возможны неточности/галлюцинации в пересказе):
- Дербилова — гибрид «статфильтр + ML-ранжирование (HistGradientBoostingRegressor)»
- Макаров — парный трейдинг на фильтре Калмана
- Немцова — арбитраж на funding rate USDRUBF / Si
- Флегонтов — cash-and-carry на парах MOEX
- ВШЭ — RL-хеджирование валютного риска на Si

## 7. Новости / Telegram как торговый сигнал

- Известия (14.04.2026): «Каждый девятый сигнал из Telegram-каналов влияет на цены акций» (11%):
  https://iz.ru/2078421/2026-04-14/kazhdyi-deviatyi-signal-iz-telegram-kanalov-vliiaet-na-tceny-aktcii
- Новая газета (18.04.2026), разбор исследования (событийный анализ, окно ±15 мин):
  https://novayagazeta.ru/articles/2026/04/18/ne-kotiruetsia
- ВШЭ, сентимент частных инвесторов (нейросетевая тональность):
  https://publications.hse.ru/articles/584949321
- ВШЭ, сентимент инвесторов (PDF):
  https://publications.hse.ru/pubs/share/direct/808361304.pdf
- ВШЭ / Теплова, новости и доходность акций (Журнал НЭА, PDF):
  https://publications.hse.ru/pubs/share/direct/584949166.pdf

## 8. Научные статьи (CyberLeninka) — хеджирование фьючерсами на российском рынке

- Круглова А.Р. (2021). «Фьючерсы на РТС как оптимальный инструмент хеджирования рисков».
  Рассчитан оптимальный коэффициент хеджирования для фьючерса на РТС и индекса РТС = 1.023:
  https://cyberleninka.ru/article/n/fyuchersy-na-rts-kak-optimalnyy-instrument-hedzhirovaniya-riskov
- Лакшина В.В., Лапшина К.А. (2016). «Сравнительный анализ стратегий хеджирования фьючерсами
  портфеля ценных бумаг»:
  https://cyberleninka.ru/article/n/sravnitelnyy-analiz-strategiy-hedzhirovaniya-fyuchersami-portfelya-tsennyh-bumag
- Ищук Т.Л., Жилкин Д.В. (2016). «Срочный рынок Московской биржи как площадка для хеджирования
  ценовых рисков»:
  https://cyberleninka.ru/article/n/srochnyy-rynok-moskovskoy-birzhi-kak-ploschadka-dlya-hedzhirovaniya-tsenovyh-riskov-finansovyh-i-tovarno-denezhnyh-operatsiy
- Джапарова И.А., Манина Е.А. (2020). «Особенности производных финансовых инструментов
  в хеджировании рисков»:
  https://cyberleninka.ru/article/n/osobennosti-proizvodnyh-finansovyh-instrumentov-v-hedzhirovanii-riskov

## 9. Академические исследования по фьючерсам (методология, не MOEX)

- Speculative Futures Trading under Mean Reversion (arXiv, OU/CIR/XOU, транзакционные издержки):
  https://arxiv.org/abs/1601.04210
- Momentum and Mean-Reversion in Commodity Spot and Futures Markets:
  https://www.sciencedirect.com/science/article/pii/S2405851315300416
- Regime-Switching Model of Returns with Momentum and Mean Reversion:
  https://www.sciencedirect.com/science/article/pii/S0264999323000494
- Regime-Conditioned Mean Reversion Framework (SSRN, 2026):
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6087107
- Quantitative Trading Strategies: Empirical Performance (NYU Stern, PDF, единый Python-бэктест):
  https://www.stern.nyu.edu/sites/default/files/2025-05/Glucksman_Lahanis.pdf
- An Application of the Ornstein-Uhlenbeck Process to Pairs Trading (обзор):
  https://pith.science/paper/2412.12458

## 10. Прочее по MOEX / практика

- Smart-Lab, корреляции 21 акции MOEX за 12 лет:
  https://smart-lab.ru/blog/1267608.php
- Smart-Lab, линейная регрессия (S&C 2007) на фьючерсах MOEX, 2025 (Si, комиссия 0.04%):
  https://smart-lab.ru/blog/1153389.php
- Smart-Lab, стратегия: сигнал от акций, исполнение на фьючерсе (комиссия 0.45 руб),
  FVG + фильтры, портфель:
  https://smart-lab.ru/tag/бэктестинг
- Smart-Lab, управление рисками и капиталом (переворот раз в день по каждому тикеру):
  https://smart-lab.ru/blog/1297952.php
- Smart-Lab, трендовая стратегия Элерса на данных MOEX (Python + aiomoex):
  https://smart-lab.ru/blog/1126711.php
- TradingView, идеи и скрипты по MOEX: https://ru.tradingview.com/scripts/moex/
- TradingView, торговые идеи по фьючерсу RTS (RI1!): https://ru.tradingview.com/symbols/RUS-RI1!/ideas/
- Habr, как делать грамотный бэктест (метрики, сигналы, сделки):
  https://habr.com/ru/articles/935196/
- Habr, арбитраж ставок финансирования (funding rate, скрипт):
  https://habr.com/ru/articles/965072/
- fkviking.com, подключение алготрейдеров к MOEX, арбитражный робот (парный трейдинг):
  https://fkviking.com/moex
- fkviking.com, вечные фьючерсы — простой взгляд:
  https://fkviking.com/eternal_futures
- Дневник исследований MT5/MOEX (визуальный бэктест):
  https://spread-sniper.com/learn/futures-arbitrage

---

## Итоговые заметки

1. Массовый тест StockSharp (п.1) подтверждён ПОЛНОСТЬЮ и детально: главный урок — сортировка
   стратегий по доходности бесполезна, решает оборот (комиссия) и проверка на шум. Реальный
   кандидат после всех проверок — только CCI Breakout и MACD DMI на 1h, и то «заслуживают проверки».
2. Парный трейдинг / статистический арбитраж — наиболее подтверждён академически (ВШЭ, CyberLeninka).
   Классика (z-score/OU) теряет устойчивость после комиссий; перспективны гибриды с ML (не подтверждено).
3. Funding rate / вечные фьючерсы MOEX — рабочая ниша с официальными данными и помесячной статистикой.
4. Фьючерс на индекс RUONIA (запуск 19.05.2026) — новый инструмент для торговли рублевой ставкой
   и арбитража между денежным, долговым и срочным рынками.
5. Хеджирование фьючерсом на РТС: оптимальный коэффициент хеджирования ~1.023 (Круглова, 2021).
6. Новостные/соцсетевые сигналы дают краткосрочный эффект, который легко съедается издержками.
