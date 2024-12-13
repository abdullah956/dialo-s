from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

print("Chatbot: Hi! Type 'bye' to exit.")

# Loop for continuous interaction
while True:
    # User input
    user_input = input("You: ")

    # Exit condition
    if user_input.lower() == "bye":
        print("Chatbot: Goodbye!")
        break

    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt")

    # Generate response
    outputs = model.generate(**inputs, max_length=50)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display response
    print(f"Chatbot: {response}")
