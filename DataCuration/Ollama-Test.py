from ollama import Client
import os

ollamaClient = Client("http://localhost:11434")
# Specify the path to your image
image_path = './abo-images-small/images/small/2a/2a291ccd.jpg'  # Replace with your image path

# Check if the image file exists
if not os.path.isfile(image_path):
    raise FileNotFoundError(f"The image file was not found at the specified path: {image_path}")

# Define your prompt
prompt = "the image contains a product, describe it as much as possible"

model_name = 'moondream' 
# model_name = 'llava:7b-v1.5-q3_K_S' 

# Create the message payload
messages = [
    {
        'role': 'user',
        'content': prompt,
        'images': [image_path]
    }
]

# Send the request to the model
response = ollamaClient.chat(model=model_name, messages=messages)

# Print the model's response
print("Model's Response:")
print(response['message']['content'])
