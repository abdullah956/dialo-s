from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# User input
user_input = "Who are you?"

# Tokenize input
inputs = tokenizer(user_input, return_tensors="pt")

# Generate response
outputs = model.generate(**inputs, max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Chatbot: {response}")