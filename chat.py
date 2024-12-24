import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import json
import re

# Expanded training data
conversations = [
    "Hello there.", "Hi! How can I help you?",
    "What's your name?", "I am your chatbot assistant.",
    "Who are you?", "I'm a friendly chatbot here to help you.",
    "How are you doing today?", "I'm functioning well and ready to assist you!",
    "What can you do?", "I can chat with you and answer questions.",
    "Tell me a joke.", "Why don't scientists trust atoms? Because they make up everything!",
    "Goodbye.", "See you later! Have a great day!",
    "Hi", "Hello! Nice to meet you!",
    "What's up?", "Just here to chat and help out!",
    "How are you?", "I'm doing great, thanks for asking!",
    "What is your purpose?", "I'm designed to chat and assist users like you."
]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s?!]', '', text)  # Keep question marks and exclamation points
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Create input-output pairs
pairs = []
for i in range(0, len(conversations), 2):
    input_text = preprocess_text(conversations[i])
    target_text = preprocess_text(conversations[i + 1])
    pairs.append((input_text, target_text))

# Prepare tokenizer and sequences
input_texts, target_texts = zip(*pairs)
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(input_texts + target_texts)

input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

vocab_size = len(tokenizer.word_index) + 1
max_seq_length = max(len(seq) for seq in input_sequences + target_sequences)

# Pad sequences
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_seq_length, padding='post')

# Prepare training data
input_data = input_sequences
target_data = np.expand_dims(target_sequences, -1)

# Enhanced model architecture
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_seq_length),
    LSTM(256, return_sequences=True),
    Dropout(0.2),
    LSTM(256, return_sequences=True),
    Dropout(0.2),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train with more epochs and smaller batch size
history = model.fit(
    input_data, target_data,
    epochs=100,
    batch_size=4,
    validation_split=0.2
)

def generate_response(input_text):
    input_text = preprocess_text(input_text)
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_seq_length, padding='post')
    
    # Direct response lookup for exact matches
    for i, (query, response) in enumerate(pairs):
        if input_text == query:
            return response
    
    # Generate response for non-exact matches
    prediction = model.predict(input_seq, verbose=0)
    predicted_sequence = []
    
    for i in range(max_seq_length):
        pred_id = np.argmax(prediction[0, i, :])
        if pred_id == 0:
            break
        predicted_sequence.append(pred_id)
    
    response_words = [tokenizer.index_word.get(idx, '') for idx in predicted_sequence if idx > 0]
    response = ' '.join(response_words).strip()
    
    return response if response else "I'm here to help! Could you please rephrase that?"

# Save the model and tokenizer
model.save('improved_chatbot.keras')
with open('improved_tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)

print("Improved chatbot is ready! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye! Take care!")
        break
    response = generate_response(user_input)
    print(f"Chatbot: {response}")