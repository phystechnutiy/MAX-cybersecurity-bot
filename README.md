# MaxMinds Anti-Fraud Chat Bot

Чат-бот для мессенджера MAX, предназначенный для анализа входящих сообщений и определения вероятности мошенничества с использованием локальной ML-модели.

---

## Структура проекта

Структура репозитория:
```bash
.
├── Dockerfile
├── main_final.py          # точка входа в приложение (запуск бота)
├── requirements.txt       # Python-зависимости
└── README.md
```
```text
Требования
Python 3.9+ (рекомендуется)
Docker (для контейнерного запуска)
Локальный .env файл
Доступ к Google Drive-архиву модели (скачивание происходит автоматически при сборке образа)

Пример содержимого .env:

MAXAPI_BOT_TOKEN=your_token_here
SCAM_THRESHOLD=0.4


```

## Запуск программы через Docker

> [!TIP]
> Убедитесь в наличии локального файла .env и активности Docker

1. Формирование образа
```bash
docker build -t maxminds-antifraud-bot .
```

2. Запуск контейнера
```bash
docker run --rm --env-file .env maxminds-antifraud-bot
```

3. Использование (для начала работы необходимо написать команду /start)

```URL
https://max.ru/t85_hakaton_bot
```
