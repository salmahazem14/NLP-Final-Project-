# --------- 1. Slim base image (size optimization) ----------
FROM python:3.11-slim AS base

# --------- 2. Environment settings ----------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# --------- 3. System dependencies (only if needed) ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


RUN pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu


COPY requirements.txt .


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/logs /app/data


COPY . .

RUN if [ ! -f /app/data/feedback.json ]; then echo "{}" > /app/data/feedback.json; fi

EXPOSE 7860
CMD ["uvicorn", "RAG.api.app:app", "--host", "0.0.0.0", "--port", "7860"]

