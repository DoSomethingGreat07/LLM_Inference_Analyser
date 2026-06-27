FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

WORKDIR ${APP_HOME}

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-benchmark.txt .
RUN pip install --no-cache-dir -r requirements-benchmark.txt

COPY src ./src
COPY examples ./examples
COPY docker ./docker
RUN chmod +x ./docker/benchmark-entrypoint.sh ./docker/pull-models.sh

ENTRYPOINT ["./docker/benchmark-entrypoint.sh"]
CMD ["--help"]
