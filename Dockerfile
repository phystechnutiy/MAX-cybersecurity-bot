FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

ENV MODEL_PATH=/app/model_anti_fraud \
    MAPPING_JSON=/app/model_anti_fraud/category_mapping_full.json \
    SCAM_THRESHOLD=0.5

CMD ["python", "main_final.py"]
