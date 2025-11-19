FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install gdown

RUN mkdir -p /app/model_anti_fraud && \
    gdown --fuzzy "https://drive.google.com/uc?id=1x_khHgWVYpxuAIZiIqWK5NOpOREEmyQT" -O /tmp/model.zip && \
    unzip /tmp/model.zip -d /tmp/model && \
    find /tmp/model -type f -exec cp {} /app/model_anti_fraud/ \; && \
    rm -rf /tmp/model /tmp/model.zip

COPY . /app

ENV MODEL_PATH=/app/model_anti_fraud \
    MAPPING_JSON=/app/model_anti_fraud/category_mapping_full.json \
    SCAM_THRESHOLD=0.4

CMD ["python", "main_final.py"]


