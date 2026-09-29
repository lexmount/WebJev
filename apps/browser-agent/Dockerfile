FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 BROWSER_AGENT_DATA=/srv/runs
WORKDIR /srv
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && useradd --create-home --uid 10001 app
COPY backend/ backend/
COPY vendor/ vendor/
COPY --from=frontend /build/dist/ frontend/dist/
RUN mkdir -p /srv/runs && chown app:app /srv/runs
USER app
EXPOSE 8787
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8787"]
