Загрузка / докачка дневных свечей по фьючерсам:
.venv/bin/python src/data/download_futures_daily.py --update

Загрузка / докачка минутных свечей по фьючерсам:
.venv/bin/python src/data/download_futures_min.py --update

Загрузка / докачка пятиминутных futoi по фьючерсам:
uv run python src/data/download_futoi.py --update

Загрузка / докачка справочников:
uv run python src/data/download_reference.py --update




