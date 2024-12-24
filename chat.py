# Import required libraries for neural network, text processing and data handling
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import json
import re

# Training dataset: pairs of input messages and corresponding responses
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

# Text preprocessing function to standardize input
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s?!]', '', text)  # Keep question marks and exclamation points
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Split conversations into input-output pairs for training
pairs = []
for i in range(0, len(conversations), 2):
    input_text = preprocess_text(conversations[i])
    target_text = preprocess_text(conversations[i + 1])
    pairs.append((input_text, target_text))

# Convert text to numerical sequences for model processing
input_texts, target_texts = zip(*pairs)
tokenizer = Tokenizer(oov_token="<OOV>")  # OOV handles unknown words
tokenizer.fit_on_texts(input_texts + target_texts)

# Convert text to integer sequences
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

# Calculate vocabulary size and maximum sequence length
vocab_size = len(tokenizer.word_index) + 1
max_seq_length = max(len(seq) for seq in input_sequences + target_sequences)

# Pad sequences to uniform length
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_seq_length, padding='post')

# Prepare final training data format
input_data = input_sequences
target_data = np.expand_dims(target_sequences, -1)

# Define neural network architecture
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_seq_length),  # Word embedding layer
    LSTM(256, return_sequences=True),  # First LSTM layer
    Dropout(0.2),  # Dropout for regularization
    LSTM(256, return_sequences=True),  # Second LSTM layer
    Dropout(0.2),  # Additional dropout
    Dense(vocab_size, activation='softmax')  # Output layer
])

# Configure model training parameters
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    input_data, target_data,
    epochs=100,
    batch_size=4,
    validation_split=0.2  # Use 20% data for validation
)

# Response generation function
def generate_response(input_text):
    input_text = preprocess_text(input_text)
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_seq_length, padding='post')
    
    # Check training data for exact matches first
    for i, (query, response) in enumerate(pairs):
        if input_text == query:
            return response
    
    # Generate new response if no exact match found
    prediction = model.predict(input_seq, verbose=0)
    predicted_sequence = []
    
    # Convert model output to word sequence
    for i in range(max_seq_length):
        pred_id = np.argmax(prediction[0, i, :])
        if pred_id == 0:  # Stop at padding token
            break
        predicted_sequence.append(pred_id)
    
    # Convert word IDs back to text
    response_words = [tokenizer.index_word.get(idx, '') for idx in predicted_sequence if idx > 0]
    response = ' '.join(response_words).strip()
    
    return response if response else "I'm here to help! Could you please rephrase that?"

# Save trained model and tokenizer
model.save('improved_chatbot.keras')
with open('improved_tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)

# Main chat loop
print("Improved chatbot is ready! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye! Take care!")
        break
    response = generate_response(user_input)
    print(f"Chatbot: {response}")