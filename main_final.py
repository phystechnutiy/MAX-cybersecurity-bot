import os
import asyncio
import logging
import json
from typing import Any, Dict, List

from dotenv import load_dotenv
from maxapi import Bot, Dispatcher
from maxapi.types import MessageCreated, Command

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker


######################################################################
# BASIC SETUP
######################################################################
logging.basicConfig(level=logging.INFO)
load_dotenv()

BOT_TOKEN = os.getenv("MAXAPI_BOT_TOKEN")

bot = Bot(BOT_TOKEN)
dp = Dispatcher()

######################################################################
# DATABASE
######################################################################
DB_PATH = "scam_contacts.db"
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Base = declarative_base()


class ScamContact(Base):
    __tablename__ = "scam_contacts"
    id = Column(Integer, primary_key=True)
    phone = Column(String, nullable=False)
    name = Column(String, nullable=False)


Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)


######################################################################
# MODEL
######################################################################
MODEL_PATH = os.getenv("MODEL_PATH", "./model_anti_fraud")
MAPPING_PATH = os.getenv("MAPPING_JSON", "./model_anti_fraud/category_mapping_full.json")
THRESHOLD = float(os.getenv("SCAM_THRESHOLD", "0.4"))

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

id2cat = {int(k): v for k, v in mapping["id2cat"].items()}
scam_categories = set(mapping["scam_categories"])

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()


######################################################################
# STATE MACHINE
######################################################################
STATE_WAITING_PHONE = "waiting_phone"
STATE_WAITING_NAME = "waiting_name"

dialog_states: Dict[int, dict] = {}


######################################################################
# HELPERS
######################################################################
def get_chat_id(event: MessageCreated):
    return event.message.recipient.chat_id


def extract_all_text(msg: Any) -> str:
    """
    Расширенная версия: вытаскивает текст из самого сообщения, пересылок,
    reply и вложенных аттачей, как в старом рабочем коде.
    """
    texts: List[str] = []

    def _iter(node: Any):
        if not node:
            return

        d = node.model_dump() if hasattr(node, "model_dump") else {}

        body = d.get("body")
        if body:
            text = body.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())

            for att in body.get("attachments", []) or []:
                _iter(att)

        link = d.get("link")
        if link and link.get("type") == "forward":
            fwd_msg = link.get("message")
            if fwd_msg:
                if hasattr(fwd_msg, "model_dump"):
                    _iter(fwd_msg)
                else:
                    fwd_text = fwd_msg.get("text") if isinstance(fwd_msg, dict) else None
                    if isinstance(fwd_text, str) and fwd_text.strip():
                        texts.append(fwd_text.strip())

        for att in d.get("attachments", []) or []:
            _iter(att)

        for fwd_key in ("fwd_messages", "forwarded", "forwards", "forward_messages"):
            fwd_list = d.get(fwd_key, [])
            for sub in fwd_list:
                _iter(sub)

        reply = d.get("reply_message")
        if reply:
            _iter(reply)

    _iter(msg)

    seen = set()
    uniq: List[str] = []
    for t in texts:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)

    return "\n\n---\n\n".join(uniq)


def predict(text: str):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]

    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    best = int(np.argmax(probs))
    cat = id2cat[best]
    prob = float(probs[best])
    return cat, prob, (cat in scam_categories and prob >= THRESHOLD)


######################################################################
# COMMAND: /start
######################################################################
@dp.message_created(Command("start"))
async def start(event: MessageCreated):
    await event.message.answer(
        "👋Привет! Это анти-мошеннический бот\n"
        "Отправь любое сообщение (включая пересланное) — я проверю его.\n\n"
        "Если я обнаружу мошенничество, предложу добавить подозрительный контакт в базу командой /add"
    )


######################################################################
# COMMAND: /add
######################################################################
@dp.message_created(Command("add"))
async def add_begin(event: MessageCreated):
    chat_id = get_chat_id(event)

    dialog_states[chat_id] = {"state": STATE_WAITING_PHONE}
    await event.message.answer(
        "Введите номер мошенника (телефон/логин/карта).\nДля отмены: /cancel"
    )


######################################################################
# COMMAND: /cancel
######################################################################
@dp.message_created(Command("cancel"))
async def cancel(event: MessageCreated):
    chat_id = get_chat_id(event)

    if chat_id in dialog_states:
        dialog_states.pop(chat_id)

    await event.message.answer("⭕️Добавление в базу отменено.\n\n🔎Возвращаюсь к анализу.")


######################################################################
# MAIN DETECTION
######################################################################
@dp.message_created()
async def detect(event: MessageCreated):
    chat_id = get_chat_id(event)
    text = event.message.body.text or ""

    ##################################################################
    # 1. WE ARE IN ADD MODE
    ##################################################################
    if chat_id in dialog_states:
        state = dialog_states[chat_id]["state"]

        if state == STATE_WAITING_PHONE:
            dialog_states[chat_id]["phone"] = text
            dialog_states[chat_id]["state"] = STATE_WAITING_NAME
            await event.message.answer("Введите имя мошенника:")
            return

        if state == STATE_WAITING_NAME:
            phone = dialog_states[chat_id]["phone"]
            name = text

            session = SessionLocal()
            c = ScamContact(phone=phone, name=name)
            session.add(c)
            session.commit()
            session.close()

            dialog_states.pop(chat_id)

            await event.message.answer(
                f"🗄️Добавлено в базу:\n\nТелефон: {phone}\nИмя: {name}"
            )
            return

    ##################################################################
    # 2. NORMAL DETECTION
    ##################################################################
    full_text = extract_all_text(event.message)
    logging.info("Извлечён текст: %s", full_text)

    if not full_text:
        await event.message.answer(
            "Нет текста для анализа.\n\nПришлите текстовое сообщение или пересланное."
        )
        return

    category, prob, is_scam = predict(full_text)

    if is_scam:
        await event.message.answer(
            f"🚨 Похоже на мошенничество.\n"
            f"Категория: {category}\nВероятность: {prob:.1%}\n\n"
            "Чтобы добавить отправителя в базу, введите команду:\n/add"
        )
    else:
        await event.message.answer(
            f"✅Признаков мошенничества нет.\n"
            f"Категория сообщений: {category} ({prob:.1%})"
        )

async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
