# MaxMinds Anti-Fraud Chat Bot

Проект представляет собой чат-бота для мессенджера MAX для анализа входящих сообщений и определения вероятности мошенничества.  

---

## Структура проекта

Структура репозитория:
```bash
.
├── Dockerfile
├── main_final.py          # точка входа в приложение (запуск бота)
├── requirements.txt       # Python-зависимости
├── .env.example           # шаблон .env без секретов
└── README.md
```
```text
Требования
Python 3.9+ (рекомендуется)
Docker (для контейнерного запуска)
Локальный .env файл

Пример содержимого .env:

MAXAPI_BOT_TOKEN=/token_here/
MODEL_PATH=./model_anti_fraud
MAPPING_JSON=./model_anti_fraud/category_mapping_full.json
SCAM_THRESHOLD=0.4
```

#Запуск программы через Docker

```text
Убедитесь в наличии локального файла .env

```

Формирование образа
```bash
docker build -t antifraud-bot .
```

Запуск контейнера
```bash
docker run --env-file .env antifraud-bot
```
