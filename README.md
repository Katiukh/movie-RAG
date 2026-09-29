# movie-rag

Сбор movie-постов публичного VK-сообщества через Playwright для semantic search / RAG.
Python 3.12+, зависимости устанавливаются через `uv sync`, браузер — через
`uv run playwright install chromium`.

```bash
uv run python -m movie_rag.ingestion.vk_browser
uv run python -m movie_rag.ingestion.parser
```

Scraper по умолчанию собирает до 500 постов; parser записывает результат в
`data/processed/movies.jsonl`. Контракт `fetch_posts(URL)` — список словарей
с `url`, отображаемой `date` и исходным `text`. Parser не изменён.

Для диагностики и коротких запусков:

```bash
uv run python -m movie_rag.ingestion.vk_browser --target-movies 50 --debug
uv run pytest -q
```

Доступны также `--max-scrolls` (по умолчанию 1000) и `--scroll-pause-ms`
(1200). `--debug` показывает scrollY, scrollHeight, количество уникальных
permalink-ссылок и movie-постов в текущем DOM, последние URL.

## Почему сбор останавливался

Лента VK виртуализирована: `article[data-index]` имеют абсолютные позиции,
старые карточки удаляются из DOM. Высота страницы включает ещё не отрисованную
область. В диагностическом запуске прыжки к `document.body.scrollHeight`
увеличили высоту с 8095 до 32739 px, но сохранили те же 5 permalink-ссылок
(4 movie-поста). Прокрутка небольшими шагами через `mouse.wheel` загрузила
56 уникальных movie-постов за 64 шага; одновременно в DOM оставалось около 8.

Scraper проходит ленту шагами меньше высоты экрана и сохраняет посты на каждом
шаге в Python-словаре. Прогресс определяется новыми URL всех постов, включая
обычные записи, а не высотой страницы или только количеством фильмов.
Ссылки с `reply`/`thread` исключаются, `vk.com` и `vk.ru` приводятся к одному URL.
Всплывающее приглашение войти закрывается штатной кнопкой «Закрыть».

Остановка происходит при достижении цели, лимита шагов, после 6 повторных
ожиданий внизу без новых постов или 30 шагов без новых постов. Последние два
случая означают конец доступной ленты **либо** остановку загрузки; браузер
не может гарантировать, что VK показал всю историю сообщества. Причина
остановки выводится в лог. Ошибка первоначальной загрузки не маскируется
пустым результатом.

Тесты запускают настоящий Chromium на локальных HTML-страницах и не обращаются
к VK. Они проверяют виртуализацию, промежуток обычных записей, комментарии,
дедупликацию, точный лимит, конец ленты и растущий пустой участок.

Проверка на реальной странице: scraper собрал 500 уникальных movie-постов за
690 шагов при паузе 400 мс. Штатный запуск `uv run python -m
movie_rag.ingestion.parser` с паузой 1200 мс сохранил 500 фильмов
с уникальными URL, непустыми датами и разделами ЧТО / О ЧЁМ / ЗАЧЕМ.

## BM25 retrieval baseline

```bash
uv sync
uv run python -m movie_rag.retrieval.bm25
uv run python -m movie_rag.retrieval.bm25 "роуд-муви" --top-k 5
uv run python -m movie_rag.retrieval.bm25 "хоррор" --corpus data/processed/movies.jsonl
```

Без query CLI прогоняет восемь тестовых запросов и печатает название, год,
BM25 score, «Что», «Зачем» и URL. Результаты ручной проверки на 500 фильмах:
[таблицы top-5 и ограничения](docs/bm25-baseline.md).

```python
from pathlib import Path
from movie_rag.retrieval.bm25 import BM25Retriever, load_movies

retriever = BM25Retriever(load_movies(Path("data/processed/movies.jsonl")))
results = retriever.search("неглупый ужастик", top_k=5)
```

Один документ на фильм: `title + what + about + why`. `not_for`, URL, год
и дата остаются metadata. Индекс строится в памяти через
[rank-bm25](https://github.com/dorianbrown/rank_bm25), `BM25Okapi` с параметрами
по умолчанию. Общий tokenizer для документов и query приводит текст к lowercase
и выделяет русские/латинские слова и цифры; дефисы разделяют слова.
Лемматизации, удаления стоп-слов и расширения синонимами нет; `ё` и `е` различаются.

Поиск возвращает копии movie-словарей с дополнительным `score`, отсортированные
по убыванию score. При равенстве сохраняется порядок корпуса. Документы без
общих токенов исключаются: результатов может быть меньше `top_k`, вплоть до
пустого списка. Пустой корпус и пустой query также дают `[]`, отрицательный
`top_k` — `ValueError`. Scores не являются уверенностью или вероятностью;
на маленьких корпусах библиотечный IDF может давать нулевые/отрицательные scores,
поэтому совпадения не отбрасываются по порогу `score > 0`.

Все тесты: `uv run pytest -q`; только retrieval: `uv run pytest -q tests/test_bm25.py`.

## Dense semantic search

BM25 выполняет lexical retrieval по совпадающим токенам. Dense retrieval
использует [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) через
`sentence-transformers` для поиска по смысловому сходству.

```bash
uv sync
uv run python -m movie_rag.retrieval.dense
uv run python -m movie_rag.retrieval.dense "что-нибудь странное про интернет" --top-k 5
uv run streamlit run app.py
```

Без аргумента query dense CLI прогоняет шесть примеров и печатает top-5.
В Streamlit по умолчанию выбран **Dense — BGE-M3**; sidebar позволяет выбрать
**BM25** и 3/5/10 результатов (по умолчанию 5). Карточки содержат название,
год, поля ЧТО / О ЧЁМ / ЗАЧЕМ, score и ссылку на исходный пост. История чата
сохраняется только в текущей сессии для просмотра: каждый запрос — самостоятельный
поиск, без учёта предыдущих сообщений. Текст карточек берётся из корпуса.

Первый запуск скачивает веса модели из Hugging Face (около 2,3 ГБ) и кодирует
корпус. CUDA выбирается автоматически, если доступна; иначе используется CPU.
`Embedding device` показан в sidebar, а CLI пишет устройство в лог.
На CPU первоначальная индексация может занять заметное время.

Один документ на фильм: `title + what + about + why`, без `not_for`, служебных
маркеров и искусственных инструкций. Нормализованные embeddings сравниваются
через NumPy dot product — это cosine similarity. Отрицательные similarity
допустимы; score не является вероятностью релевантности. Scores BM25 и Dense
находятся в разных шкалах и напрямую не сравниваются.

Embeddings корпуса вычисляются один раз при создании retriever, затем
`search()` кодирует только query. Streamlit переиспользует retriever через
`st.cache_resource`. Для перезапуска процесса есть дисковый кэш:

```text
data/embeddings/
├── bge_m3_embeddings.npy
└── bge_m3_metadata.json
```

Metadata хранит модель, dimension, ordered URL, fingerprint текстов и URL,
флаг нормализации и checksum матрицы. Изменения индексируемого текста, URL,
порядка документов или модели инвалидируют кэш; повреждённый кэш пересоздаётся.
Изменение только `not_for`/даты не требует нового кодирования. Кэш embeddings
исключён из Git. Принудительная перестройка:

```bash
uv run python -m movie_rag.retrieval.dense --rebuild-cache
```

Для небольшого GPU можно уменьшить batch size: `--batch-size 4`. Другой корпус
и каталог кэша задаются через `--corpus` и `--cache-dir`. Пустой query возвращает
`[]`; отрицательный `top_k` вызывает `ValueError`. Исходные movie-словари не меняются.

```python
from pathlib import Path
from movie_rag.retrieval.bm25 import load_movies
from movie_rag.retrieval.dense import DenseRetriever

retriever = DenseRetriever(load_movies(Path("data/processed/movies.jsonl")))
results = retriever.search("неглупый ужастик", top_k=5)
```

Проверки: `uv run pytest -q`. Тесты Dense и UI используют fake-модели/retrievers;
они не скачивают BGE-M3 и не требуют GPU.

[Результаты реального сравнения Dense/BM25: top-5, scores и наблюдения](docs/dense-baseline.md).
Проверено на CUDA: 500 нормализованных float32-векторов размерности 1024;
повторный запуск в отдельном процессе использовал сохранённый кэш.

Ненужная фоновая конвертация весов Transformers в safetensors отключена при загрузке модели: она могла удерживать CLI после завершения поиска. Используются оригинальные веса BGE-M3.
