import json
import random
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Завантаження моделі та обʼєктів ===
model = load_model('chat_bot/chatbot_model.h5')

with open('chat_bot/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('chat_bot/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('chat_bot/final_intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# === Очистка тексту ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zа-яіїєґ0-9\\s]', '', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

# === Глобальний контекст ===
active_context = None

# === Отримати відповідь ===
def get_response(user_input, show_top=3):
    global active_context
    cleaned = clean_text(user_input)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=model.input_shape[1], padding='post')
    prediction = model.predict(padded, verbose=0)[0]

    # Вибір тегів з урахуванням context_filter
    tag_probs = []
    for idx in range(len(prediction)):
        full_tag = label_encoder.inverse_transform([idx])[0]
        tag = full_tag.split("__")[-1]
        # Перевіряємо context_filter
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                context_filter = intent.get("context_filter")
                if not context_filter or context_filter == active_context:
                    tag_probs.append((tag, prediction[idx], intent))
                break

    tag_probs.sort(key=lambda x: x[1], reverse=True)
    tag_probs = tag_probs[:show_top]

    if not tag_probs or tag_probs[0][1] < 0.3:
        return "⚠️ Вибач, я не зрозумів. Можеш повторити?"

    top_tag, top_conf, top_intent = tag_probs[0]

    # Оновлюємо контекст
    if "context_set" in top_intent:
        active_context = top_intent["context_set"]

    # Вивід діагностики
    print("\\n🧠 Top {} передбачень:".format(show_top))
    for tag, conf, intent in tag_probs:
        print(f" → [{tag}] ({conf*100:.1f}%): {random.choice(intent['responses'])}")

    return f"✅ Відповідь: {random.choice(top_intent['responses'])}"

# === Головний цикл ===
if __name__ == "__main__":
    print("🤖 Fredbear's Pizza Бот запущено з підтримкою контексту!")
    print("Введи запит або 'вихід' для завершення.\\n")
    while True:
        msg = input("👤 Ти: ")
        if msg.lower() in ["вихід", "вийти", "exit", "quit"]:
            print("🤖 Бот: Бувай! 🍕")
            break
        response = get_response(msg)
        print("🤖 Бот:", response)
