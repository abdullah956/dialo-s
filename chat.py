import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
import re
import json

# Download NLTK data
nltk.download('punkt')

# Load and preprocess dataset
def load_cornell_data(filepath):
    conversations = []
    with open(filepath, 'r', encoding='iso-8859-1') as file:
        for line in file:
            conversations.append(line.strip())
    return conversations

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Load conversations (Cornell Movie Dialogs dataset)
data_path = 'movie_lines.txt'  # Update the path to your dataset
conversations = load_cornell_data(data_path)

# Tokenization and preprocessing
pairs = []
for i in range(len(conversations) - 1):
    input_text = preprocess_text(conversations[i])
    target_text = preprocess_text(conversations[i + 1])
    pairs.append((input_text, target_text))

# Prepare the tokenizer
input_texts, target_texts = zip(*pairs)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_texts + target_texts)

input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

max_seq_length = max(len(seq) for seq in input_sequences + target_sequences)

input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_seq_length, padding='post')

vocab_size = len(tokenizer.word_index) + 1

# Build the model
embedding_dim = 128
units = 256

def create_model(vocab_size, max_seq_length, embedding_dim, units):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_seq_length),
        LSTM(units, return_sequences=True),
        Dropout(0.2),
        LSTM(units),
        Dropout(0.2),
        Dense(units, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model(vocab_size, max_seq_length, embedding_dim, units)
model.summary()

# Prepare input and target for training
input_data = np.array(input_sequences)
target_data = np.array(target_sequences)

# Train the model
epochs = 10
batch_size = 64
model.fit(input_data, target_data, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Save the model and tokenizer
model.save('chatbot_model.h5')
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)

# Chatbot inference
def chatbot_response(input_text):
    input_text = preprocess_text(input_text)
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_seq_length, padding='post')
    prediction = model.predict(input_seq)
    predicted_seq = np.argmax(prediction, axis=1)
    response = ' '.join(tokenizer.index_word.get(index, '') for index in predicted_seq if index != 0)
    return response

# Test the chatbot
print("Chatbot is ready! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print(f"Chatbot: {response}")