from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load the tokenizer
with open('improved_tokenizer.json', 'r') as f:
    tokenizer_json = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_json)

# Load the trained model
model = load_model('improved_chatbot.keras')  # Replace with the actual path to your model
max_seq_length = 100

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s?!]', '', text)  # Keep question marks and exclamation points
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_response(input_text):
    input_text = preprocess_text(input_text)
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_seq_length, padding='post')

    prediction = model.predict(input_seq, verbose=0)
    predicted_sequence = [np.argmax(prediction[0, i, :]) for i in range(max_seq_length)]
    response_words = [tokenizer.index_word.get(idx, '') for idx in predicted_sequence if idx > 0]
    response = ' '.join(response_words).strip()
    return response if response else "I'm here to help! Could you please rephrase that?"

def home(request):
    return render(request, 'home.html')

def chat_response(request):
    if request.method == 'POST':
        user_message = request.POST.get('message', '')
        response = generate_response(user_message)
        return JsonResponse({'response': response})
    return JsonResponse({'error': 'Invalid request'}, status=400)
