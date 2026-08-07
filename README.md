# AI Helper

Telegram-бот с диалоговым AI-агентом на LangGraph. Агент обращается к LLM через
универсальный интерфейс провайдера; сейчас реализован провайдер OpenRouter.

## Возможности

- работа в разрешённых Telegram-чатах;
- обработка сообщений только с `@упоминанием` бота;
- отдельная UUID-сессия для каждого чата;
- общий контекст для всех участников одного группового чата;
- ручной сброс контекста и полное удаление старой сессии;
- отправка готовых случайных и тематических мемов через Humor API;
- самостоятельный выбор агентом между случайным мемом и поиском по теме;
- интерактивный статус подготовки ответа с обновлением каждые 5 секунд;
- преобразование Markdown от LLM в нативное форматирование Telegram;
- безопасное разделение длинных ответов с учётом лимита Telegram;
- системный промпт из отдельного YAML-файла;
- подробные логи обработки запросов.

## Требования

- Python `3.14`;
- Poetry;
- токен Telegram-бота от BotFather;
- API-ключ OpenRouter.
- API-ключ Humor API.

## Рекомендуемые требования к VPS

LLM запускается на стороне OpenRouter, поэтому серверу не нужны GPU и высокая
вычислительная мощность. VPS обслуживает только Telegram long polling, хранение
контекста в оперативной памяти и сетевые запросы к API.

| Параметр | Минимум | Рекомендуется |
|---|---:|---:|
| CPU | 1 vCPU | 2 vCPU |
| RAM | 1 ГБ | 2 ГБ |
| Диск | 10 ГБ SSD | 20 ГБ SSD |
| Swap | 512 МБ | 1 ГБ |
| ОС | Linux x86_64/ARM64 | Ubuntu 24.04 LTS или Debian 13 |

Минимальной конфигурации достаточно для личного бота или небольшой группы. Для
нескольких активных чатов, параллельных запросов, обновления Docker-образов без
дефицита места и более стабильной работы рекомендуется конфигурация с 2 vCPU,
2 ГБ RAM и 20 ГБ SSD.

Серверу необходим стабильный исходящий HTTPS-доступ на порт `443` к Telegram API,
OpenRouter, Humor API и реестру Docker-образов. Входящие порты для работы бота
открывать не нужно, поскольку используется long polling.

> [!WARNING]
> **VPS с российским IP для этого проекта не подходит: OpenRouter с российских
> IP не работает.** Выбирайте сервер в юрисдикции, где доступны OpenRouter и
> выбранный поставщик модели. Перед размещением проверьте актуальные региональные
> ограничения конкретной модели.

OpenRouter указывает, что некоторые поставщики моделей запрещают доступ из
определённых стран и что такие ограничения нельзя обходить через VPN или прокси.
Используйте сервис только в соответствии с
[условиями OpenRouter](https://openrouter.ai/terms) и правилами выбранной модели.

## Установка

Все команды ниже выполняются из корня проекта.

Создайте и активируйте локальное виртуальное окружение:

```bash
python3.14 -m venv venv
source venv/bin/activate
```

Установите Poetry в это окружение и установите зависимости проекта:

```bash
python -m pip install poetry
poetry install --no-root
```

Проект рассчитан на использование локального каталога `venv`. Не устанавливайте
зависимости в системный Python.

## Настройка

Скопируйте пример конфигурации:

```bash
cp .env.example .env
```

Заполните `.env`:

```dotenv
# OpenRouter
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=your_model_name
OPENROUTER_SITE_URL=
OPENROUTER_APP_NAME=ai-helper
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_TIMEOUT=60

# Humor API
HUMOR_API_API_KEY=your_humor_api_key
HUMOR_API_RANDOM_URL=https://api.humorapi.com/memes/random
HUMOR_API_SEARCH_URL=https://api.humorapi.com/memes/search
HUMOR_API_TIMEOUT=15
HUMOR_API_USER_AGENT=ai-helper/0.1

# Telegram
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_ALLOWED_CHAT_IDS=-1001234567890,123456789
TELEGRAM_CALENDAR_DEFAULT_TIMEZONE=Europe/Moscow

# PostgreSQL
POSTGRES_DB=ai_helper
POSTGRES_USER=ai_helper
POSTGRES_PASSWORD=replace-with-a-strong-password
DATABASE_URL=postgresql+asyncpg://ai_helper:replace-with-a-strong-password@postgres:5432/ai_helper
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=5
DATABASE_POOL_TIMEOUT=30

# Reminder worker
REMINDER_POLL_INTERVAL=10
REMINDER_BATCH_SIZE=20
REMINDER_LEASE_TIMEOUT_SECONDS=300
REMINDER_MAX_ATTEMPTS=5
REMINDER_RETRY_BASE_DELAY_SECONDS=30
REMINDER_RETRY_MAX_DELAY_SECONDS=3600
REMINDER_RETRY_JITTER_RATIO=0.1
REMINDER_SHUTDOWN_TIMEOUT=30
```

`POSTGRES_*` используются контейнером PostgreSQL при первичной инициализации,
а `DATABASE_*` — приложением и Alembic. Пароль в `DATABASE_URL` должен совпадать
с `POSTGRES_PASSWORD`. Если пароль содержит специальные символы URL, их нужно
закодировать.

`TELEGRAM_CALENDAR_DEFAULT_TIMEZONE` — IANA-таймзона для чата, пока у него нет
сохранённой настройки календаря. После первого календарного действия настройка
чата хранится в PostgreSQL и не зависит от контекста LangGraph.

Параметры `REMINDER_*` управляют встроенным worker-ом: частотой опроса, размером
пачки, временем lease, количеством попыток, backoff и временем graceful shutdown.
Worker использует PostgreSQL как источник истины и после перезапуска восстанавливает
зависшие доставки по истечении lease.

### Разрешённые чаты

`TELEGRAM_ALLOWED_CHAT_IDS` — список Telegram `chat_id`, разделённых запятыми.

- ID пользователя или личного чата обычно положительный: `123456789`.
- ID группы или супергруппы обычно отрицательный: `-1001234567890`.
- Сообщения из остальных чатов бот молча игнорирует.

## Запуск Telegram-бота

Из корня проекта с активированным `venv`:

```bash
python -m bot
```

Или без активации окружения:

```bash
venv/bin/python -m bot
```

## Запуск через Docker Compose

Для контейнерного запуска нужны Docker и Docker Compose. Убедитесь, что `.env`
создан и заполнен, затем выполните из корня проекта:

```bash
docker compose up --build -d
```

Compose запускает отдельный PostgreSQL-контейнер с постоянным именованным volume,
дожидается готовности БД, применяет миграции Alembic одноразовым сервисом
`migrate` и только затем запускает бота. Бот работает через long polling, поэтому
публиковать порты приложения и PostgreSQL не нужно. Compose передаёт переменные
из `.env` внутрь контейнеров. Сам `.env` исключён из контекста сборки через
`.dockerignore` и не попадает в Docker-образ.

Worker напоминаний запускается внутри процесса бота отдельной фоновой задачей. Он
не зависит от активного диалога или контекста LangGraph: сообщение формируется из
сохранённых события и `message_text`, после чего отправляется напрямую через
Telegram Bot API.

Просмотр логов в реальном времени:

```bash
docker compose logs -f bot
```

Проверка состояния контейнера:

```bash
docker compose ps
```

Остановка и удаление контейнера:

```bash
docker compose down
```

Эта команда сохраняет PostgreSQL volume. Для удаления данных БД требуется явно
добавить `--volumes`; используйте это только если данные действительно больше не
нужны.

После изменения исходного кода или зависимостей пересоберите образ:

```bash
docker compose up --build -d
```

### Миграции PostgreSQL

При обычном запуске Compose миграции применяются автоматически сервисом
`migrate`. Локально их можно запустить из существующего `venv`, указав доступный
с хоста адрес PostgreSQL в `DATABASE_URL`:

```bash
DATABASE_URL=postgresql+asyncpg://user:password@127.0.0.1:5432/database \
  venv/bin/python -m alembic upgrade head
```

Текущая ревизия и история:

```bash
venv/bin/python -m alembic current
venv/bin/python -m alembic history
```

### Тестовая PostgreSQL-среда

Тестовая БД запускается отдельно, использует порт `55432` только на localhost и
хранит данные в `tmpfs`. Она не использует production volume:

```bash
docker compose -f docker-compose.test.yaml up -d --wait
```

Применение миграций и запуск всех тестов с проверкой реальной схемы:

```bash
DATABASE_URL=postgresql+asyncpg://ai_helper_test:ai_helper_test@127.0.0.1:55432/ai_helper_test \
  venv/bin/python -m alembic upgrade head

DATABASE_TEST_URL=postgresql+asyncpg://ai_helper_test:ai_helper_test@127.0.0.1:55432/ai_helper_test \
  venv/bin/python -m unittest discover -v
```

Остановка и удаление тестовой среды:

```bash
docker compose -f docker-compose.test.yaml down
```

Контейнер запускается от непривилегированного пользователя и автоматически
перезапускается после сбоя, пока он не остановлен вручную.

После запуска в терминале отображаются:

- подключение к Telegram;
- разрешённые обращения;
- UUID текущей сессии;
- начало запроса к LLM;
- периодический статус долгого запроса;
- время получения и отправки ответа;
- запросы мемов и отправка изображений;
- ротация сессии после сброса;
- ошибки Telegram, OpenRouter и Humor API.

## Использование в Telegram

### Обращение к боту

Перед сообщением обязательно укажите username бота:

```text
@bot_username Объясни, как работает LangGraph
```

Без `@упоминания` сообщение не будет обработано.

После получения вопроса бот сразу отправляет статус. Пока LLM готовит ответ,
статус периодически редактируется. Когда ответ готов, это же сообщение заменяется
финальным ответом.

### Мемы

Агент сам решает, когда использовать инструмент отправки мемов. Явную просьбу без
темы, например `пришли мем`, он обрабатывает как запрос случайного мема. Если
пользователь указывает тему, например `пришли мем про программиста`, агент ищет
готовый мем по одному английскому ключевому слову. За один запрос Humor API
возвращается не более одного результата. Изображение отправляется без подписи.

### Контекст чата

Каждому разрешённому чату соответствует отдельная UUID-сессия LangGraph. В
групповом чате контекст общий: сообщения одного участника влияют на последующие
ответы другим участникам этого же чата.

Контексты разных чатов не смешиваются.

### Сброс контекста

Для сброса истории отправьте:

```text
@bot_username /reset
```

При сбросе бот:

1. Удаляет все checkpoints старой сессии из `InMemorySaver`.
2. Перестаёт использовать старый UUID.
3. Создаёт для чата новую пустую UUID-сессию.

Контексты других чатов не затрагиваются. Без ручного сброса история активной
сессии продолжает расти до перезапуска процесса.

## Архитектура

```text
ai_helper/
├── agent/
│   ├── agent.py              # LangGraph-агент и управление контекстом
│   ├── providers.py          # провайдер OpenRouter
│   ├── settings.py           # настройки OpenRouter и Humor API
│   ├── tools.py              # инструменты агента и получение мемов
│   └── prompts/
│       ├── __init__.py       # загрузка и валидация промптов
│       └── system.yaml       # системный промпт агента
├── bot/
│   ├── __main__.py           # запуск через python -m bot
│   ├── bot.py                # aiogram handlers и управление сессиями
│   └── settings.py           # настройки Telegram
├── .env.example
├── .dockerignore
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
└── poetry.lock
```

Поток обработки сообщения:

```text
Telegram message
    → проверка chat_id
    → проверка @упоминания
    → UUID-сессия текущего чата
    → LangGraph
    → OpenRouterProvider
    → вызов выбранного моделью инструмента (если нужен)
    → текст: преобразование Markdown и редактирование статусного сообщения
    → мем: отправка изображения в Telegram без подписи
```

`Agent` зависит не от OpenRouter напрямую, а от протокола `LLMProvider`. Для
подключения другой LLM достаточно создать провайдер с методами `generate()` и
`agenerate()` и передать его в конструктор `Agent`.

## Системный промпт

Системный промпт находится в:

```text
agent/prompts/system.yaml
```

Он загружается и валидируется при создании агента. Текст системного промпта не
захардкожен в `agent.py`.

## Проверки перед коммитом

Запустить все проверки:

```bash
pre-commit run --all-files
```

В проекте настроены Ruff, проверка форматирования, проверка YAML/TOML и проверка
актуальности `poetry.lock`.
