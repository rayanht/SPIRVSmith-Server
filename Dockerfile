FROM python:3.10-buster

RUN pip install poetry

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-dev

COPY . .
ARG GOOGLE_APPLICATION_CREDENTIALS
RUN echo $GOOGLE_APPLICATION_CREDENTIALS | base64 -d > /app/credentials.json 
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
ENV APP_ENV "prod"

CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]