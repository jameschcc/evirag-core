ARG BASE_IMAGE=docker.1ms.run/library/python:3.11-slim
FROM ${BASE_IMAGE}
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ENV PIP_INDEX_URL=$PIP_INDEX_URL PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" --upgrade pip && pip install --no-cache-dir --index-url "$PIP_INDEX_URL" .
COPY src ./src
COPY tests ./tests
ARG KB_GIT_TAG=""
ARG KB_GIT_COMMIT=""
ENV KB_GIT_TAG=$KB_GIT_TAG KB_GIT_COMMIT=$KB_GIT_COMMIT
EXPOSE 8000
CMD ["uvicorn", "kb.main:app", "--host", "0.0.0.0", "--port", "8000"]
