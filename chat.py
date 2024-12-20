import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import nltk
import json
import re

nltk.download('punkt')

# Preprocess the conversations data
conversations = [
    "Hello there.",
    "Hi! How can I help you?",
    "What's your name?",
    "I am your chatbot.",
    "How are you doing today?",
    "I'm just a program, but I'm here to help you.",
    "What can you do?",
    "I can chat with you and answer questions.",
    "Tell me a joke.",
    "Why don't scientists trust atoms? Because they make up everything!",
    "Goodbye.",
    "See you later!"
]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Prepare input-output pairs
pairs = []
for i in range(len(conversations) - 1):
    input_text = preprocess_text(conversations[i])
    target_text = preprocess_text(conversations[i + 1])
    if input_text and target_text:
        pairs.append((input_text, target_text))

# Prepare tokenizer and sequences
input_texts, target_texts = zip(*pairs)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_texts + target_texts)

input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

# Calculate vocab size
vocab_size = len(tokenizer.word_index) + 1

# Pad input sequences
max_seq_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='post')

# Prepare target data - shift sequences by one position
target_sequences = [seq + [0] for seq in target_sequences]  # Add padding token
target_sequences = pad_sequences(target_sequences, maxlen=max_seq_length, padding='post')

# For sparse_categorical_crossentropy, the target shape should match output shape
# Thus, remove the last token in target data to match the target output shape
input_data = input_sequences
target_data = np.expand_dims(target_sequences, -1)

# Build the model
model = Sequential([
    Embedding(vocab_size, 256, input_length=max_seq_length),
    LSTM(512, return_sequences=True),
    LSTM(512, return_sequences=True),
    Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Training callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(input_data, target_data, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Save the model and tokenizer
model.save('chatbot_model.keras')
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)

# Function to generate a response
def generate_response(input_text, max_length=None):
    if max_length is None:
        max_length = max_seq_length - 1
    
    # Preprocess input
    input_text = preprocess_text(input_text)
    print(f"Processed input text: {input_text}")
    
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_length, padding='post')
    print(f"Input sequence: {input_seq}")
    
    # Generate prediction
    predicted_sequence = []
    for _ in range(max_length):
        prediction = model.predict(input_seq, verbose=0)
        predicted_id = np.argmax(prediction[0, -1, :])
        predicted_sequence.append(predicted_id)
        print(f"Predicted id: {predicted_id}")
        
        if predicted_id == 0:  # Stop if padding token is predicted
            break
            
        # Update input sequence for next prediction
        input_seq = np.append(input_seq[:, 1:], [[predicted_id]], axis=1)
        print(f"Updated input sequence: {input_seq}")
    
    # Convert ids to words
    response_words = []
    for idx in predicted_sequence:
        if idx > 0:  # Skip padding token
            word = tokenizer.index_word.get(idx, '')
            if word:
                response_words.append(word)
    
    response = ' '.join(response_words)
    print(f"Generated response: {response}")
    return response

# Interactive chat loop
print("Chatbot is ready! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = generate_response(user_input)
    print(f"Chatbot: {response}")