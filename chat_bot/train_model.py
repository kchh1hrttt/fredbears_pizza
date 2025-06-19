import json
import re
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Очистка текста
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^а-яa-z0-9їієґ\'’\\s]', '', text)  # українська підтримка
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Загрузка intents
with open('chat_bot/8.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

texts = []
labels = []

# Сбор паттернов и меток
for intent in data['intents']:
    context = intent.get('context_set', '')
    tag = intent['tag']
    combined_label = f"{context}__{tag}" if context else tag
    for pattern in intent['patterns']:
        texts.append(clean_text(pattern))
        labels.append(combined_label)

# Токенизация
vocab_size = 5000
tokenizer = Tokenizer(oov_token="<OOV>", num_words=vocab_size)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_len = max(len(s) for s in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Кодировка меток
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)

# Построение модели
embedding_dim = 64
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Обучение
model.fit(padded_sequences, categorical_labels, epochs=100, batch_size=8, verbose=1)

# Сохранение модели и инструментов
model.save('chat_bot/chatbot_model.h5')
with open('chat_bot/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('chat_bot/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Модель обучена и сохранена.")
