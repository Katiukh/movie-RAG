# Технические заметки

[Описание проекта и результаты](../README.md). Здесь сохранены подробности
реализации, запуска и проверки отдельных подходов.

## Локальные рекомендации с Ollama / Qwen3 8B

Streamlit использует последовательность: Hybrid top-20 → BGE reranker → top-10
→ локальный Ollama (`qwen3:8b`) → все выбранные фильмы с исходными описаниями из corpus.
Количество ответов зависит от соответствия запросу: допустимы 0–10 фильмов. Название, год и ссылка
берутся из корпуса; внутренние rankings, scores и отклонённые кандидаты в UI
не показываются. Каждый запрос оценивается отдельно от истории чата.

Установите [Ollama](https://ollama.com/download). Запустите сервер в отдельном
терминале с отключёнными облачными функциями:

```bash
OLLAMA_NO_CLOUD=1 ollama serve
```

Если Ollama уже работает как приложение или служба, задайте `OLLAMA_NO_CLOUD=1`
в окружении сервера и перезапустите его. Это [настройка сервера Ollama](https://docs.ollama.com/faq#how-do-i-disable-ollama-cloud-features),
а не `.env` приложения; она блокирует облачный inference, в том числе через aliases моделей.

Загрузите модель и запустите приложение:

```bash
ollama pull qwen3:8b
uv sync
uv run streamlit run app.py
```

Inference выполняется локально; API key не нужен. По умолчанию используются
`http://localhost:11434` и `qwen3:8b`. Настройки из
[.env.example](../.env.example) необязательны: их можно задать через environment
variables или `.env` в корне проекта. Здесь находятся только параметры локального
Ollama. Уже заданные environment variables имеют приоритет над `.env`.

```dotenv
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=qwen3:8b
LLM_TIMEOUT_SECONDS=60
```

Локальный API-вызов находится в `src/movie_rag/generation/recommender.py`; он получает
готовых кандидатов и не запускает retrieval. В модель передаются query,
`movie_id` (существующий `url`), `title`, `what`, `about`, `why`, `not_for` и schema.
Факты берутся только из `what`, `about`, `why`, `not_for`; название и год служат
идентификации и не разрешают использовать знания о сюжете из памяти модели.
Поле `not_for` используется моделью при оценке релевантности; документы
и passage retrieval/reranker остаются прежними.

Модель используется только как relevance filter и возвращает список IDs:

```json
{"selected_ids": ["URL из переданных кандидатов", "другой URL из кандидатов"]}
```

Пустой список `{"selected_ids": []}` валиден. Количество выбранных фильмов
не фиксировано: 0, 1, несколько или все 10. В [Ollama `/api/chat`](https://docs.ollama.com/api/chat)
передаётся эта schema через `format`, с `stream: false`.

Компактный prompt требует оценивать каждый фильм независимо и выбирать его,
только если описание достаточно подтверждает все существенные условия query.
Для «романтика но грустная» нужны и романтика, и грустный/меланхоличный/трагический
характер. Для «мало стрельбы но она есть» нужны стрельба и её ограниченная роль.
Для «фантастика без хоррора» нужны фантастика и подтверждение отсутствия хоррора;
простое отсутствие слова «хоррор» не доказывает отрицательное условие.
Недостаток информации или противоречие означает отказ. Модель не должна выбирать
«лучший из неподходящих», использовать память о фильме или жанровые догадки.

Локальная validation принимает только объект с `selected_ids`: список строк,
каждая из которых принадлежит переданным top-10. Неизвестные IDs, дубликаты,
неправильные типы и дополнительные поля отклоняют ответ целиком. Старые ответы
с evaluations, requirements, evidence или reason не принимаются.
Смысловая релевантность остаётся решением модели; код проверяет контракт и IDs.

Приложение сопоставляет выбранные IDs с исходными Movie objects и возвращает
их копии в прежнем порядке BGE reranker. Модель не управляет порядком и ничего
не добавляет к описаниям. UI показывает название и год, `what` как краткое
описание, «О чём» (`about`), «Почему стоит посмотреть» (`why`),
«Кому может не подойти» (`not_for`) и ссылку на источник. Пустые поля пропускаются.
Поля отображаются дословно как обычный текст, без интерпретации Markdown.
Весь текст о фильме приходит из human-written corpus.

Reason generation, evidence/quote extraction, requirements/status schema,
rankings и scores от модели удалены. В UI нет сгенерированных объяснений,
внутренних scores, JSON и отклонённых кандидатов. Один запрос делает один
локальный LLM-вызов; второго вызова и agent workflow нет.

История прежнего формата очищается при обновлении. После изменения generation
полностью перезапустите Streamlit: `Ctrl+C`, затем `uv run streamlit run app.py`
и обновите страницу браузера, чтобы сервер загрузил актуальные Python-модули.

В Ollama отключено thinking (`think: false`);
UI использует только выбранные corpus movies после validation IDs. Ошибки конфигурации, timeout,
API, пустой или оборванный ответ и невалидный JSON показываются понятным
сообщением, без traceback, деталей ответа сервера и fallback на retrieval.
Пустой список кандидатов не вызывает API. Один запрос делает один API-вызов
без автоматических повторов; connect timeout — 5 секунд, read timeout задаётся
`LLM_TIMEOUT_SECONDS`.

`LLM_BASE_URL` допускает только локальный loopback-адрес: `localhost`,
`127.0.0.1` или `::1` (например, другой порт Ollama). HTTP redirects и
использование HTTP proxies из окружения отключены. Внешний LLM API и fallback
на него отсутствуют. Имя локально установленной модели задаётся `LLM_MODEL`;
default — `qwen3:8b`. Явные теги `:cloud` и `-cloud` отклоняются до HTTP-вызова.
Для inference заданы `num_ctx=8192`, `num_predict=4096`
и `temperature=0.2`. При долгой первоначальной загрузке или работе на CPU
можно увеличить `LLM_TIMEOUT_SECONDS`, например до `180`.

Если приложение не может подключиться к Ollama, сначала проверьте запуск
сервера командой `ollama list`. В WSL сервер должен быть доступен из того же
окружения, где запущен Streamlit. Увеличение `LLM_TIMEOUT_SECONDS` помогает
при долгом inference, но не устраняет отсутствие подключения к серверу.

Тесты `tests/test_generation.py` и `tests/test_app.py` подменяют HTTP API
и модели retrieval. Они проверяют valid/unknown/duplicate IDs, пустой выбор,
0/1/несколько/10 результатов, безопасные ошибки, бюджет 20 → 10, неизменность
corpus fields, буквальное отображение исходных текстов и отсутствие генерации
пользовательского текста. Prompt tests закрепляют corpus-only отбор и все условия
составных запросов; unit tests не оценивают смысловой отбор самой модели.
Запущенный Ollama для unit tests не нужен. Retrieval benchmark, его gold,
candidate budget и метрики остаются прежними.

## Retrieval benchmark

Независимый benchmark содержит 50 запросов по текущему корпусу: 10 near_exact,
25 paraphrase, 15 semantic; 9 semantic-запросов имеют по 2–5 positives.
Разметка использует существующие URL документов;
основания релевантности сохранены в `notes`. Кандидаты и проверенный gold лежат
в `data/eval/`. [Методика, ограничения и обновление](../data/eval/README.md).
[Сравнение BM25, Dense, Hybrid, reranker и LLM: overall и все типы запросов](../reports/retrieval-benchmark-comparison.md).
[Оценка LLM-фильтра и сравнение с остальными режимами](../reports/llm-benchmark-comparison.md).

```bash
uv run python -m movie_rag.evaluation --validate-only
uv run python -m movie_rag.evaluation --retriever bm25
uv run python -m movie_rag.evaluation --retriever dense \
  --output artifacts/eval/dense_results.jsonl \
  --summary-output artifacts/eval/dense_summary.json
uv run python -m movie_rag.evaluation --retriever hybrid \
  --output artifacts/eval/hybrid_results.jsonl \
  --summary-output artifacts/eval/hybrid_summary.json
uv run python -m movie_rag.evaluation --retriever reranker \
  --output artifacts/eval/reranker_results.jsonl \
  --summary-output artifacts/eval/reranker_summary.json

# Ollama должен быть запущен, qwen3:8b — загружена локально.
uv run python -m movie_rag.evaluation --retriever llm
```

CLI печатает Recall@1/3/5/10/20 overall и по каждому типу query, сохраняет ranking
и метрики каждого запроса для error analysis, а также summary с хешами входных
данных. Без параметров запускается BM25. Общие флаги: `--dataset`, `--corpus`,
`--ks`; Dense/Hybrid используют существующий кэш через `--cache-dir` и
`--batch-size`, Hybrid — прежние `--candidate-k 20`, `--rrf-k 60`.
Режим `reranker` оценивает существующий Hybrid + BGE reranker на 20 кандидатах.
Его inference-флаги: `--reranker-batch-size 4`, `--reranker-max-length 512`,
`--reranker-device cpu|cuda` (по умолчанию автоматический выбор).
Retrieval pipeline и gold не изменяются при оценке.

Режим `llm` оценивает выдачу приложения целиком: Hybrid top-20 → BGE reranker
top-10 → локальный Ollama relevance filter. На каждый непустой набор кандидатов
делается один вызов `generation/recommender.py`; модель возвращает только
`selected_ids`, а порядок остаётся от reranker. Неизвестные/повторные IDs и ошибки
Ollama прерывают запуск, как ошибки других backends; они не считаются пустой выдачей.
Пустой корректный выбор даёт Recall 0. Количество результатов — от 0 до 10,
поэтому Recall@20 равен Recall@10. Даже с `--ks 1` модель получает все top-10,
а `--candidate-k` для этого режима должен оставаться равным 20.

LLM использует те же `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT_SECONDS`, что и UI:
переменные окружения имеют приоритет над `.env` проекта, default — локальный
`qwen3:8b`. API key и внешний fallback не нужны. По умолчанию результаты
сохраняются в `artifacts/eval/llm_results.jsonl` и `artifacts/eval/llm_summary.json`,
не затрагивая прежние benchmark artifacts; пути можно переопределить обычными
`--output` и `--summary-output`. Summary дополнительно содержит настройки Ollama,
бюджет top-10, хеш prompt и хеш всех передаваемых corpus fields, включая `not_for`.
Метрики и разбивка по типам query общие для всех пяти режимов. Это оценка Recall
по прежнему gold, а не проверка каждого семантического решения модели.
Unit tests LLM-eval мокают HTTP и загрузку моделей; Ollama им не требуется.

```python
from pathlib import Path
from movie_rag.evaluation import evaluate_retriever, load_eval_dataset
from movie_rag.retrieval.bm25 import BM25Retriever, load_movies

movies = load_movies(Path("data/processed/movies.jsonl"))
dataset = load_eval_dataset(Path("data/eval/retrieval_eval.jsonl"), movies)
results = evaluate_retriever(BM25Retriever(movies), dataset, ks=[1, 3, 5, 10, 20])
# results["overall"], results["by_query_type"], results["per_query"]
```

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
[таблицы top-5 и ограничения](../docs/bm25-baseline.md).

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
В Streamlit Dense входит в Hybrid pipeline; пользователь видит финальные
рекомендации после BGE reranker и Qwen3 8B. История чата сохраняется только
в текущей сессии для просмотра и не передаётся модели при следующем запросе.

Первый запуск скачивает веса модели из Hugging Face (около 2,3 ГБ) и кодирует
корпус. CUDA выбирается автоматически, если доступна; иначе используется CPU.
CLI пишет устройство в лог.
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

[Результаты реального сравнения Dense/BM25: top-5, scores и наблюдения](../docs/dense-baseline.md).
Проверено на CUDA: 500 нормализованных float32-векторов размерности 1024;
повторный запуск в отдельном процессе использовал сохранённый кэш.

Ненужная фоновая конвертация весов Transformers в safetensors отключена при загрузке модели: она могла удерживать CLI после завершения поиска. Используются оригинальные веса BGE-M3.


## Hybrid retrieval

BM25 находит точные лексические совпадения, BGE-M3 — смысловое сходство.
Reciprocal Rank Fusion (RRF) объединяет их rankings по позициям документов,
не складывая несопоставимые BM25 scores и cosine similarity.

```text
Query
 ├── BM25 ──── top-N ──┐
 └── BGE-M3 ── top-N ──┤
                      RRF
                       ↓
                     Top-K
```

Для документа `d`: `RRF(d) = 1/(rrf_k + rank_bm25(d)) + 1/(rrf_k + rank_dense(d))`.
Ранги начинаются с 1; отсутствующий документ даёт нулевой вклад от этого поиска.
По умолчанию `candidate_k=20`, `top_k=5`, `rrf_k=60`. Если `top_k > candidate_k`,
каждый retriever запрашивается с глубиной `max(candidate_k, top_k)`.
Кандидаты объединяются по уникальному `url`, а не по названию. Повторный URL
в одном ranking учитывается только по первому вхождению. Сортировка:
RRF по убыванию, затем dense rank, BM25 rank и URL по возрастанию;
отсутствующий ранг считается бесконечным.

```python
from pathlib import Path
from movie_rag.retrieval.bm25 import BM25Retriever, load_movies
from movie_rag.retrieval.dense import DenseRetriever
from movie_rag.retrieval.hybrid import HybridRetriever

movies = load_movies(Path("data/processed/movies.jsonl"))
bm25 = BM25Retriever(movies)
dense = DenseRetriever(movies)
hybrid = HybridRetriever(bm25, dense, candidate_k=20, rrf_k=60)
results = hybrid.search("неглупый ужастик", top_k=5)
```

Hybrid использует переданные retrievers: сам не загружает модель, не строит
BM25 index и не вычисляет embeddings корпуса. Алгоритмы существующих BM25 и
Dense не изменены. Результат — копия metadata фильма с `rrf_score`,
`bm25_rank`, `bm25_score`, `dense_rank`, `dense_score`; отсутствующий источник
обозначается `None`. Неоднозначного общего поля `score` у Hybrid нет.
Исходные словари не меняются.

Пустой query или `top_k=0` возвращает `[]` без вызовов retrievers.
Если один ranking пуст, используется второй; если оба пусты, результат `[]`.
`candidate_k < 1`, `rrf_k < 0` и `top_k < 0` вызывают `ValueError`.
Ошибка backend не маскируется пустой выдачей: CLI сообщает исключение,
а Streamlit показывает существующее сообщение об ошибке поиска.

В Streamlit Hybrid используется для получения кандидатов перед reranking
и LLM filtering. Кэшированные экземпляры BM25 и Dense переиспользуются
для той же версии корпуса; RRF и исходные оценки доступны через CLI.

Сравнение всех восьми запросов BM25 baseline или одного своего запроса:

```bash
uv run python -m movie_rag.retrieval.hybrid
uv run python -m movie_rag.retrieval.hybrid "хоррор" --top-k 5 --candidate-k 20 --rrf-k 60
uv run pytest -q
uv run ruff check app.py src/movie_rag/retrieval/hybrid.py tests/test_hybrid.py tests/test_app.py
```

CLI также принимает `--corpus`, `--cache-dir`, `--batch-size`, `--rebuild-cache`.
Модель и индексы создаются один раз за запуск, каждый query кодируется один раз;
три выдачи используют одни и те же rankings. Unit tests используют fake
retrievers без загрузки BGE-M3; интеграционные тесты проверяют UI и повторное
использование ресурсов.

### Ручное сравнение BM25 / Dense / Hybrid

Запуск 2026-09-29: 500 фильмов, BGE-M3 на CUDA, существующий кэш embeddings,
`candidate_k=20`, `rrf_k=60`, `top_k=5`. Ниже фактическая выдача команды
`uv run python -m movie_rag.retrieval.hybrid`. Ранги B/D в колонке Hybrid —
позиции в top-20 BM25/Dense; «—» означает отсутствие в этом списке.
Это качественная проверка восьми примеров, а не новая quantitative evaluation.

Наблюдения по выдаче и описаниям фильмов в корпусе:

- **«неглупый ужастик»**: BM25 пуст, Hybrid полностью повторяет Dense.
  «Пакетоголовый» подходит как необычный мамблкор-хоррор; далее встречаются
  комедии и документальная сатира. Отсутствие пустого ответа исправлено,
  но вся пятёрка не стала релевантной.
- **«хоррор»**: Hybrid сохраняет два фильма из BM25 top-5 —
  «Коко-ди Коко-да» и «Господин Дьявол». «Пир» выходит на первое место благодаря
  попаданию в оба rankings. Жанровая выдача сохраняется, исходный top-5 — нет.
- **«мрачный криминальный триллер»**: первым становится «Комната страха»
  (BM25 #4, Dense #6); в top-5 также «Советник» и «Стрингер».
  Пересечение rankings заметно важнее первого места только в одном поиске.
- **«уютное осеннее кино»**: «Девятые врата» остаётся первым за счёт двух
  вкладов; «Осенняя Соната» поднимается на второе место. Надёжного улучшения
  по настроению нет: тяжёлая семейная драма не обязательно уютная,
  а «Осиное гнездо» — боевик про осаду склада.
- **«одиночество»**: сильный exact match «Мы все идём на всемирную выставку»
  сохраняется на втором месте, хотя не входит в Dense top-20. При одинаковом
  RRF первым идёт Dense #1 по объявленному правилу разрешения равенств.
- **«роуд-муви»**: «Вне игры» остаётся первым; четыре фильма из BM25 top-5
  сохраняются. «Взрослые игры» вытесняет дорожный триллер «Неистовый».
  Сильная лексическая выдача в основном сохранена, но не полностью.
- **«программисты и хакеры»**: «Сеть», прямо описанная как история
  программистки и хакеров, приходит на второе место. Однако первым становится
  нерелевантная криминальная драма «В погоне за Бонни и Клайдом»
  (BM25 #18, Dense #17): RRF усиливает случайное пересечение. Шум BM25 остаётся.
- **«программист хакер»**: BM25 пуст; Hybrid повторяет Dense с «Сетью» первой.
  Семантический поиск преодолевает отсутствие точных словоформ, но следующие
  фильмы не обязательно о программистах или хакерах.

При `rrf_k=60` и глубине 20 даже два вклада с позиции #20 дают `0.025`,
что больше одного вклада с позиции #1 (`≈0.016393`). Поэтому RRF поощряет
согласие поисков, включая ошибочное, и не гарантирует улучшения каждого запроса.
Весов, reranker и изменений токенизации в этой реализации нет.

<details>
<summary>неглупый ужастик — top-5</summary>

| # | BM25 (score) | Dense (cosine) | Hybrid (RRF; B/D ranks) |
|---|---|---|---|
| 1 | — | Пакетоголовый (2008)<br>0.517469 | Пакетоголовый (2008)<br>0.016393; —/1 |
| 2 | — | Непокой (2025)<br>0.508432 | Непокой (2025)<br>0.016129; —/2 |
| 3 | — | Ослеплённый желаниями (2000)<br>0.507095 | Ослеплённый желаниями (2000)<br>0.015873; —/3 |
| 4 | — | Несносный дед (2013)<br>0.497462 | Несносный дед (2013)<br>0.015625; —/4 |
| 5 | — | Ништяк, браток (2020)<br>0.495795 | Ништяк, браток (2020)<br>0.015385; —/5 |

</details>

<details>
<summary>хоррор — top-5</summary>

| # | BM25 (score) | Dense (cosine) | Hybrid (RRF; B/D ranks) |
|---|---|---|---|
| 1 | Адский ад (2020)<br>4.619912 | Комната страха (2002)<br>0.545985 | Пир (2005)<br>0.029469; 11/5 |
| 2 | Господин Дьявол (2019)<br>4.412393 | Страх (1983)<br>0.542000 | Коко-ди Коко-да (2019)<br>0.029469; 5/11 |
| 3 | Клоуны-убийцы из космоса (1987)<br>3.526612 | Гость (2013)<br>0.534315 | Страх (1983)<br>0.029287; 16/2 |
| 4 | Монстры Юга (2015)<br>3.372514 | Дежавю (2006)<br>0.529550 | Господин Дьявол (2019)<br>0.029116; 2/17 |
| 5 | Коко-ди Коко-да (2019)<br>3.231319 | Пир (2005)<br>0.528722 | Похоронены, но не мертвы (1981)<br>0.028439; 7/14 |

</details>

<details>
<summary>мрачный криминальный триллер — top-5</summary>

| # | BM25 (score) | Dense (cosine) | Hybrid (RRF; B/D ranks) |
|---|---|---|---|
| 1 | Всё ради неё (2008)<br>6.868926 | Тёмный полдень (1995)<br>0.612929 | Комната страха (2002)<br>0.030777; 4/6 |
| 2 | Советник (2013)<br>5.864102 | «Иллюзия убийства» (2025)<br>0.605947 | Серые люди (2024)<br>0.028778; 10/9 |
| 3 | Сеть (1995)<br>5.756305 | Наблюдающий (2022)<br>0.605736 | Советник (2013)<br>0.028629; 2/20 |
| 4 | Комната страха (2002)<br>5.686615 | Стрингер (2013)<br>0.597769 | Стрингер (2013)<br>0.028125; 20/4 |
| 5 | Дом на Турецкой улице (2002)<br>5.618593 | Кровавое лето Сэма (1999)<br>0.582396 | Она (2015)<br>0.025316; 19/19 |

</details>

<details>
<summary>уютное осеннее кино — top-5</summary>

| # | BM25 (score) | Dense (cosine) | Hybrid (RRF; B/D ranks) |
|---|---|---|---|
| 1 | Девятые врата (1999)<br>2.701585 | Осенняя Соната (1978)<br>0.586101 | Девятые врата (1999)<br>0.030092; 1/13 |
| 2 | Метрополис (1927)<br>2.593038 | Последний киносеанс (1971)<br>0.527231 | Осенняя Соната (1978)<br>0.016393; —/1 |
| 3 | Мальтийский сокол (1941)<br>2.531999 | Осиное гнездо (2001)<br>0.524960 | Последний киносеанс (1971)<br>0.016129; —/2 |
| 4 | Любовь актрисы Сумако (1947)<br>2.492878 | Был месяц май (1970)<br>0.521554 | Метрополис (1927)<br>0.016129; 2/— |
| 5 | Память (2021)<br>2.473767 | Яркий летний день (1991)<br>0.514293 | Осиное гнездо (2001)<br>0.015873; —/3 |

</details>

<details>
<summary>одиночество — top-5</summary>

| # | BM25 (score) | Dense (cosine) | Hybrid (RRF; B/D ranks) |
|---|---|---|---|
| 1 | Мы все идём на всемирную выставку (2021)<br>6.102558 | Одинокая белая женщина (1992)<br>0.529645 | Одинокая белая женщина (1992)<br>0.016393; —/1 |
| 2 | — | Джорджино (1994)<br>0.507234 | Мы все идём на всемирную выставку (2021)<br>0.016393; 1/— |
| 3 | — | Обнаженная (1993)<br>0.500727 | Джорджино (1994)<br>0.016129; —/2 |
| 4 | — | Воячек (1999)<br>0.498830 | Обнаженная (1993)<br>0.015873; —/3 |
| 5 | — | Молчание моря (1949)<br>0.493263 | Воячек (1999)<br>0.015625; —/4 |

</details>

<details>
<summary>роуд-муви — top-5</summary>

| # | BM25 (score) | Dense (cosine) | Hybrid (RRF; B/D ranks) |
|---|---|---|---|
| 1 | Вне игры (2002)<br>10.897140 | Вне игры (2002)<br>0.486967 | Вне игры (2002)<br>0.032787; 1/1 |
| 2 | Сёстры на всю голову (2024)<br>8.908580 | Каникулы в Мексике (2023)<br>0.444898 | Сёстры на всю голову (2024)<br>0.032002; 2/3 |
| 3 | Последний взмах флага (2017)<br>8.293469 | Сёстры на всю голову (2024)<br>0.438233 | Каникулы в Мексике (2023)<br>0.031514; 5/2 |
| 4 | Взрослые игры (2017)<br>8.242651 | Неистовый (2020)<br>0.409285 | Последний взмах флага (2017)<br>0.031258; 3/5 |
| 5 | Каникулы в Мексике (2023)<br>7.766743 | Последний взмах флага (2017)<br>0.391336 | Неистовый (2020)<br>0.015625; —/4 |

</details>

<details>
<summary>программисты и хакеры — top-5</summary>

| # | BM25 (score) | Dense (cosine) | Hybrid (RRF; B/D ranks) |
|---|---|---|---|
| 1 | Без чувств (1998)<br>8.049383 | Сеть (1995)<br>0.517128 | В погоне за Бонни и Клайдом (2019)<br>0.025808; 18/17 |
| 2 | Ночи в стиле буги (1997)<br>2.887738 | Система безопасности (1964)<br>0.509825 | Сеть (1995)<br>0.016393; —/1 |
| 3 | Фанни и Александр (1982)<br>2.886027 | Первому игроку приготовиться (2018)<br>0.490486 | Без чувств (1998)<br>0.016393; 1/— |
| 4 | Клеопатра (1963)<br>2.864344 | На крючке (2008)<br>0.480953 | Система безопасности (1964)<br>0.016129; —/2 |
| 5 | Джорджино (1994)<br>2.863435 | Апгрейд (2018)<br>0.476756 | Ночи в стиле буги (1997)<br>0.016129; 2/— |

</details>

<details>
<summary>программист хакер — top-5</summary>

| # | BM25 (score) | Dense (cosine) | Hybrid (RRF; B/D ranks) |
|---|---|---|---|
| 1 | — | Сеть (1995)<br>0.485923 | Сеть (1995)<br>0.016393; —/1 |
| 2 | — | Система безопасности (1964)<br>0.479283 | Система безопасности (1964)<br>0.016129; —/2 |
| 3 | — | Кабельщик (1996)<br>0.460334 | Кабельщик (1996)<br>0.015873; —/3 |
| 4 | — | Первому игроку приготовиться (2018)<br>0.458414 | Первому игроку приготовиться (2018)<br>0.015625; —/4 |
| 5 | — | Дежавю (2006)<br>0.453078 | Дежавю (2006)<br>0.015385; —/5 |

</details>

## Reranking

Hybrid retrieval собирает кандидатов по лексическим и семантическим сигналам.
Затем [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
совместно оценивает каждую пару «query — passage» и переставляет кандидатов
по relevance score. Это multilingual cross-encoder: он не строит embeddings
и не просматривает весь корпус.

```text
Query
 ├── BM25 ─────────┐
 └── BGE-M3 dense ─┤
                 RRF
                  ↓
                top-20
                  ↓
         BGE reranker v2 m3
                  ↓
                top-5
```

Используется официальный вариант inference через Transformers:
`AutoTokenizer` + `AutoModelForSequenceClassification`, `eval()` и
`torch.inference_mode()`. Sigmoid применяется к logits обученной модели;
`reranker_score` находится в диапазоне 0..1, но не является калиброванной
вероятностью. CUDA выбирается автоматически с FP16; без CUDA используется
CPU с FP32. По умолчанию batch size = 4, максимальная длина пары = 512 токенов
с padding/truncation. Очень длинные описания могут быть обрезаны.

Passage совпадает с Dense: `title + what + about + why`, без `not_for`.
В результатах остаются все metadata Hybrid, включая `rrf_score`, исходные
BM25/Dense ранги и scores; добавляются `reranker_score` и `hybrid_rank`
(позиция до перестановки, начиная с 1). Сортировка — reranker score по убыванию,
при равенстве сохраняется порядок Hybrid. Повторные URL учитываются один раз,
по первому вхождению; исходные candidate dicts не изменяются.

```python
from pathlib import Path
from movie_rag.retrieval.bm25 import BM25Retriever, load_movies
from movie_rag.retrieval.dense import DenseRetriever
from movie_rag.retrieval.hybrid import HybridRetriever
from movie_rag.retrieval.reranker import Reranker, RerankedHybridRetriever

movies = load_movies(Path("data/processed/movies.jsonl"))
hybrid = HybridRetriever(BM25Retriever(movies), DenseRetriever(movies))
reranker = Reranker()
retriever = RerankedHybridRetriever(hybrid, reranker, candidate_k=20)
results = retriever.search("дорожное кино про дружбу", top_k=5)
```

`candidate_k` ограничивает объём reranking: при `top_k > candidate_k`
возвращается не более `candidate_k` результатов. Поиск остаётся двухэтапным;
модель не ищет пропущенные Hybrid фильмы. Пустые кандидаты, пустой запрос
или `top_k=0` дают `[]` без scoring. Отрицательный `top_k`, неположительный
`candidate_k` и некорректные параметры inference вызывают `ValueError`.
Ошибки модели не маскируются старой Hybrid-выдачей.

В Streamlit Hybrid + Reranker возвращает top-10 для Qwen3 8B; карточки
показывают выбранные фильмы и исходные поля corpus. Отдельные
backends и их scores остаются доступны в CLI и retrieval benchmark.
Reranker кэшируется через отдельный `st.cache_resource`, независимо от
fingerprint корпуса, а Hybrid переиспользует существующие BM25/Dense ресурсы.
Повторные запросы и rerun не загружают модели заново.

```bash
uv sync
uv run streamlit run app.py

# Все 12 запросов из ТЗ: Hybrid top-5, Reranked top-5, перемещения и latency
uv run python -m movie_rag.retrieval.reranker

# Один запрос, свой бюджет кандидатов и размер batch
uv run python -m movie_rag.retrieval.reranker "программист хакер" --candidate-k 20 --top-k 5 --batch-size 4

# CPU для reranker (Dense сохраняет собственный автоматический выбор устройства)
uv run python -m movie_rag.retrieval.reranker "хоррор" --device cpu

# Машиночитаемый отчёт: кандидаты, результаты и отдельные времена этапов
uv run python -m movie_rag.retrieval.reranker --output-json /tmp/reranker-comparison.json

uv run pytest -q
uv run ruff check .
```

CLI принимает также `--corpus`, `--cache-dir`, `--max-length`.
Загрузка обеих моделей и инициализация индексов измеряются отдельно (`setup`).
Hybrid выполняется один раз на запрос; его top-20 используется и для сравнения,
и для reranking. Время reranker включает токенизацию, inference, перенос scores
на CPU и сортировку. Первый запрос может включать прогрев CUDA; эти числа
не следует считать универсальной производительностью сервиса.

Unit tests используют fake scorer и не скачивают модель. Проверяются
сортировка, лимиты, исходные ранги и metadata, отсутствие мутаций, дедупликация,
пустой/одиночный candidate, текст passage, бюджет кандидатов, CPU/CUDA dtype,
батчи, CLI и повторное использование ресурсов в UI.

### Результаты проверки reranker

[Полный отчёт по 12 запросам: Hybrid/Reranked top-5, перемещения, latency и failure cases](../reports/reranker-comparison.md).
Проверено на 500 фильмах, CUDA, RTX 5060 Ti 16 ГБ, FP16; `candidate_k=20`,
`top_k=5`, batch size 4, max length 512. Медиана: Hybrid **14,6 мс**,
reranker **107,3 мс**, сумма этапов **120,3 мс**, без загрузки моделей.
Также прошли реальный CPU inference в FP32 и Streamlit AppTest с настоящими моделями.

«Дом храбрых» сохранил **Hybrid #1 → Reranked #1** для запроса о послевоенной
адаптации. Для «роуд-муви» «Взрослые игры» поднялись **6 → 5**, и в top-5
оказались все пять фильмов сильного BM25 baseline. Но для «неглупый ужастик»
первой стала драма «Обнаженная», а для «дорожное кино про дружбу» — романтическая
история «Влюблённые». Reranking работает технически, но не гарантирует улучшения
релевантности; низкие scores не интерпретируются как вероятность и не используются
для отсечения результатов.
