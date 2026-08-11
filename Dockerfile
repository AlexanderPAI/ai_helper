FROM python:3.14.6-slim-trixie

ARG POETRY_VERSION=2.4.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN python -m pip install --no-cache-dir "poetry==${POETRY_VERSION}" \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app

COPY pyproject.toml poetry.lock README.md ./

RUN poetry install --only main --no-root --no-ansi \
    && python -m pip cache purge

COPY --chown=app:app agent ./agent
COPY --chown=app:app bot ./bot
COPY --chown=app:app calendar_app ./calendar_app
COPY --chown=app:app reminder_app ./reminder_app
COPY --chown=app:app database ./database
COPY --chown=app:app config.py ./config.py
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app alembic.ini ./alembic.ini

USER app

CMD ["python", "-m", "bot"]
