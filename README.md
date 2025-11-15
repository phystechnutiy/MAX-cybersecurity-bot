# MaxMinds Anti-fraud Bot

Анти-мошеннический бот для платформы MAX.  
Бот принимает обычные и пересланные сообщения, прогоняет их через модель классификации и:

- оценивает вероятность, что сообщение мошенническое;
- показывает категорию сообщения;
- при желании позволяет добавить контакт мошенника в локальную базу (`SQLite`), чтобы использовать её отдельно.

## Стек

- Python 3.11
- [maxapi](https://github.com/love-apples/maxapi)
- transformers, torch
- SQLAlchemy + SQLite
- python-dotenv

## Структура проекта

.
├── main_final.py                 # основной скрипт бота
├── model_anti_fraud/          # директория с моделью
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ... (остальные файлы модели)
├── model_anti_fraud/category_mapping_full.json
├── scam_contacts.db           # локальная база мошенников (создаётся автоматически)
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
