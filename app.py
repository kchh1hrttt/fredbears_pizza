from flask import Flask, jsonify, render_template, request, redirect, session, url_for, render_template_string
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.stem import PorterStemmer
import numpy as np
import tensorflow as tf
import pickle
import random
import re

app = Flask(__name__)
app.secret_key = 'qwererwerr234r423r23r423'
USERS_FILE = 'users.json'
AVATAR_FOLDER = 'static/avatars'

os.makedirs(AVATAR_FOLDER, exist_ok=True)





app = Flask(__name__)

# === Завантаження моделі та обʼєктів ===
model = load_model('chat_bot/chatbot_model.h5')

with open('chat_bot/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('chat_bot/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('chat_bot/8.json', 'r', encoding='utf-8') as f:
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

    return f"{random.choice(top_intent['responses'])}"


@app.route("/chat_hints")
def get_chat_hints():
    hint_phrases = []
    for intent in intents.get("intents", []):
        patterns = intent.get("patterns", [])
        if patterns:
            hint_phrases.extend(random.sample(patterns, min(2, len(patterns))))

    unique_hints = list(dict.fromkeys(hint_phrases))
    selected_hints = random.sample(unique_hints, min(10, len(unique_hints)))

    return jsonify(selected_hints)


# === Роут для HTML форми ===
@app.route("/")
def index():
    return render_template("index.html")

# === Роут для обробки запитів від користувача ===
@app.route("/get", methods=["GET"])
def chat_response():
    user_input = request.args.get("msg")
    response = get_response(user_input)
    return jsonify({"response": response})


pizzas = [
    {'name': 'Рустика', 'description': 'Тiсто, курка з беконом, помідорами черрі та червоною цибулею на тонкому тісті з моцарелою та базиліком.', 'price': '500', 'image': 'рустика.png'},
    {'name': 'Літня', 'description': 'Тісто, томатний соус, сир моцарелла, куряче філе, бекон, помідори чері, червона цибуля, болгарський перець, листя базиліку.', 'price': '500', 'image': 'літня.png'},
    {'name': 'Делікатесна', 'description': 'Тісто, томатний соус, сир моцарелла, шинка, гриби, артишоки, оливки, зелень.', 'price': '500', 'image': 'делікатесна.png'},
    {'name': 'Літня', 'description': 'Тісто, томатний соус, сир моцарелла, куряче філе, бекон, помідори чері, червона цибуля, болгарський перець, листя базиліку.', 'price': '500', 'image': 'літня.png'},
    {'name': 'Маргарита', 'description': 'Тісто, томатний соус, сир моцарелла, помідори чері, оливки, базилік.', 'price': '500', 'image': 'маргарита.png'},
    {'name': 'Пепероні', 'description': 'Тісто, томатний соус, сир моцарелла, пепероні.', 'price': '500', 'image': 'пепероні.png'},
    {'name': 'Щедра', 'description': 'істо, томатний соус, сир моцарелла, пепероні, гриби, помідори чері, яйця, оливки, шинка, цибуля, зелень.', 'price': '500', 'image': 'щедра.png'},
    {'name': 'Літня', 'description': 'Тісто, томатний соус, сир моцарелла, куряче філе, бекон, помідори чері, червона цибуля, болгарський перець, листя базиліку.', 'price': '500', 'image': 'літня.png'},
    {'name': 'Делікатесна', 'description': 'Тісто, томатний соус, сир моцарелла, шинка, гриби, артишоки, оливки, зелень.', 'price': '500', 'image': 'делікатесна.png'},
    {'name': 'Літня', 'description': 'Тісто, томатний соус, сир моцарелла, куряче філе, бекон, помідори чері, червона цибуля, болгарський перець, листя базиліку.', 'price': '500', 'image': 'літня.png'},
    {'name': 'Маргарита', 'description': 'Тісто, томатний соус, сир моцарелла, помідори чері, оливки, базилік.', 'price': '500', 'image': 'маргарита.png'},
    {'name': 'Пепероні', 'description': 'Тісто, томатний соус, сир моцарелла, пепероні.', 'price': '500', 'image': 'пепероні.png'},
    {'name': 'Щедра', 'description': 'істо, томатний соус, сир моцарелла, пепероні, гриби, помідори чері, яйця, оливки, шинка, цибуля, зелень.', 'price': '500', 'image': 'щедра.png'}
]

drinks = [
    {'name': 'Кола 0.5л', 'description': '', 'price': '30', 'image': 'кока_кола.png'},
    {'name': 'Пиво "Ара"', 'description': '', 'price': '55', 'image': 'ара.png'},
    {'name': 'Пепсі 0.5л', 'description': '', 'price': '30', 'image': 'пепсі.png'},
    {'name': 'Спрайт 0.5', 'description': '', 'price': '42', 'image': 'спрайт.png'},
    {'name': 'Кола 1л', 'description': '', 'price': '50', 'image': 'кока_кола.png'},
    {'name': 'Пепсі 1л', 'description': '', 'price': '55', 'image': 'пепсі.png'},
    {'name': 'Спрайт 1л', 'description': '', 'price': '65', 'image': 'спрайт.png'},
]

sweets = [
    {'name': 'Містер Кекс', 'description': '', 'price': '99999', 'image': 'містер_кекс.png'},
    {'name': 'Містер Кекс', 'description': '', 'price': '99999', 'image': 'містер_кекс.png'},
    {'name': 'Містер Кекс', 'description': '', 'price': '99999', 'image': 'містер_кекс.png'},
    {'name': 'Містер Кекс', 'description': '', 'price': '99999', 'image': 'містер_кекс.png'},
    {'name': 'Містер Кекс', 'description': '', 'price': '99999', 'image': 'містер_кекс.png'},
    {'name': 'Містер Кекс', 'description': '', 'price': '99999', 'image': 'містер_кекс.png'},
    {'name': 'Містер Кекс', 'description': '', 'price': '99999', 'image': 'містер_кекс.png'},
    {'name': 'Містер Кекс', 'description': '', 'price': '99999', 'image': 'містер_кекс.png'},
    {'name': 'Містер Кекс', 'description': '', 'price': '99999', 'image': 'містер_кекс.png'},
    {'name': 'Містер Кекс', 'description': '', 'price': '99999', 'image': 'містер_кекс.png'},
    {'name': 'Містер Кекс', 'description': '', 'price': '99999', 'image': 'містер_кекс.png'},
]


@app.route("/q")
def indexq():
    return render_template('index.html')


@app.route("/menu")
def menu():
    return render_template('menu.html', pizzas=pizzas, drinks=drinks, sweets=sweets)


@app.route('/get_cart_count')
def get_cart_count():
    cart = session.get('cart', [])
    total_count = sum(item.get('quantity', 1) for item in cart)
    return jsonify({'cart_count': total_count})



@app.route("/cart")
def cart():
    cart = session.get('cart', [])
    total_price = sum(
        float(item['total_price']) for item in cart if isinstance(item, dict)
    )
    return render_template('cart.html', cart=cart, total_price=total_price)


def find_product(name):
    # Ищем по всем категориям — пицца, напитки, десерты
    for product in pizzas + drinks + sweets:
        if product['name'] == name:
            return product
    return None


@app.route('/add_to_cart_ajax', methods=['POST'])
def add_to_cart_ajax():
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data'}), 400

    name = data.get('name')
    if not name:
        return jsonify({'success': False, 'error': 'No name'}), 400

    product = find_product(name)
    if not product:
        return jsonify({'success': False, 'error': 'Product not found'}), 404

    # Ініціалізація корзини
    if 'cart' not in session or not isinstance(session['cart'], list):
        session['cart'] = []

    cart = session['cart']

    # Перевірка, чи вже є товар у корзині — якщо так, збільшуємо кількість
    for item in cart:
        if isinstance(item, dict) and item.get('name') == name:
            item['quantity'] += 1
            item['total_price'] = round(item['quantity'] * float(product['price']), 2)
            break
    else:
        # Додаємо новий товар
        cart.append({
            'name': product['name'],
            'price': product['price'],
            'quantity': 1,
            'total_price': float(product['price']),
            'image': product['image']
        })

    session['cart'] = cart
    session.modified = True

    return jsonify({
        'success': True,
        'cart_count': sum(item['quantity'] for item in cart)
    })


@app.route('/clear_cart', methods=["POST"])
def clear_cart():
    session.pop('cart', None)
    return redirect(url_for('cart'))


# Тимчасова заглушка AI
def ai_response(message):
    if "привіт" in message.lower():
        return "Привіт! Чим можу допомогти?"
    elif "замовлення" in message.lower():
        return "Хочете зробити замовлення? Розкажіть деталі."
    else:
        return "Вибач, я поки не розумію це повідомлення 😢"


@app.route("/support")
def chat_page():
    # Витягуємо по 1–2 патерни з кожного інтенту
    hint_phrases = []
    for intent in intents.get("intents", []):
        patterns = intent.get("patterns", [])
        if patterns:
            hint_phrases.extend(random.sample(patterns, min(2, len(patterns))))

    # Унікальні, максимум 10–12 штук
    unique_hints = list(dict.fromkeys(hint_phrases))
    selected_hints = random.sample(unique_hints, min(10, len(unique_hints)))

    return render_template("support.html", hints=selected_hints)


@app.route("/support", methods=["POST"])
def chatq():
    data = request.get_json()
    message = data.get("message", "")
    reply = ai_response(message)
    return jsonify({"reply": reply})


@app.route('/add_from_cart_ajax', methods=['POST'])
def add_from_cart_ajax():
    data = request.get_json()
    name = data.get('name')
    if not name:
        return jsonify({'success': False, 'error': 'No name'}), 400

    product = find_product(name)
    if not product:
        return jsonify({'success': False, 'error': 'Product not found'}), 404

    cart = session.get('cart', [])

    for item in cart:
        if item.get('name') == name:
            item['quantity'] += 1
            item['total_price'] = round(item['quantity'] * float(item['price']), 2)
            break
    else:
        price = float(product['price'])
        cart.append({
            'name': product['name'],
            'price': price,
            'quantity': 1,
            'total_price': round(price, 2),
            'image': product['image']
        })

    session['cart'] = cart
    session.modified = True
    return jsonify({'success': True})


@app.route('/remove_from_cart_ajax', methods=['POST'])
def remove_from_cart_ajax():
    data = request.get_json()
    name = data.get('name')
    if not name:
        return jsonify({'success': False, 'error': 'No name'}), 400

    cart = session.get('cart', [])

    for item in cart:
        if item.get('name') == name:
            item['quantity'] -= 1
            if item['quantity'] <= 0:
                cart.remove(item)
            else:
                item['total_price'] = round(item['quantity'] * float(item['price']), 2)
            break

    session['cart'] = cart
    session.modified = True
    return jsonify({'success': True})


@app.route('/order_form', methods=['GET', 'POST'])
def order_form():
    if request.method == 'POST':
        name = request.form.get('name')
        address = request.form.get('address')
        phone = request.form.get('phone')
        payment_type = request.form.get('payment_type')

        # Тут може бути логіка збереження замовлення

        # 🧹 Очищаємо корзину після замовлення
        session['cart'] = []
        session.modified = True

        # Редирект на сторінку подяки
        if payment_type == 'cash':
            
            session['cart'] = []
            session.modified = True

            return redirect('/thank_you')
        else:
            return '', 204  # JS сам зробить редирект

    # GET-запит — відображення форми
    user_data = None
    username = session.get('username')  # беремо логін з сесії

    if username:
        with open('users.json', 'r', encoding='utf-8') as f:
            users = json.load(f)
            for user in users.values():
                if user.get('name') == username:
                    user_data = user
                    break

    pizza = None
    cart = session.get('cart', [])
    if cart:
        pizza = cart[0]

    return render_template('order_form.html', user=user_data, pizza=pizza)


# --- Завантаження користувачів ---
def load_users():
    if not os.path.exists("users.json"):
        with open("users.json", "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=4)  # <-- Заміна списку на словник
        return {}
    with open("users.json", "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            # Якщо дані не словник, наприклад список, повернути пустий словник
            if not isinstance(data, dict):
                return {}
            return data
        except json.JSONDecodeError:
            with open("users.json", "w", encoding="utf-8") as f2:
                json.dump({}, f2, ensure_ascii=False, indent=4)  # <-- теж словник
            return {}


# --- Збереження користувачів ---
def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        address = request.form['address']
        phone = request.form['phone']

        users = load_users()
        if username in users:
            return 'Користувач уже існує'

        hashed_password = generate_password_hash(password)

        users[username] = {
            'password': hashed_password,
            'name': name,
            'address': address,
            'phone': phone,
            'avatar': None  # или если добавишь загрузку аватара — тут filename
        }

        save_users(users)
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()

        if username in users and check_password_hash(users[username]['password'], password):
            session['username'] = username
            return redirect(url_for('profile'))
        return 'Невірний логін або пароль'

    return render_template('login.html')


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'username' not in session:
        return redirect(url_for('login'))

    users = load_users()
    username = session['username']
    user = users.get(username)

    if request.method == 'POST':
        avatar_file = request.files.get('avatar')
        if avatar_file and avatar_file.filename:
            avatar_filename = secure_filename(username + '_' + avatar_file.filename)
            avatar_file.save(os.path.join(AVATAR_FOLDER, avatar_filename))
            user['avatar'] = avatar_filename
            save_users(users)

    return render_template('profile.html', user={'username': username, 'avatar': user.get('avatar')})

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')


if __name__ == "__main__":
    app.run(debug=True)
