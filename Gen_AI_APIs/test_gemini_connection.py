# Test script to verify the Gemini API key setup and connection
# Sends a basic request to ensure the API is working correctly

import os # Provides access to environment variables
from dotenv import load_dotenv  # Loads variables from the .env file
from google import genai # Gemini API client

# Load environment variables from the .env file
load_dotenv()

# Retrieve the Gemini API key and model name from environment variables
gemini_key = os.getenv('GEMINI_API_KEY')
model = os.getenv('GEMINI_MODEL')

# Initialize the Gemini API client using the API key
client = genai.Client(api_key=gemini_key)

# Send a basic request to the Gemini API to generate text content
response = client.models.generate_content(
    model=model, # Specify the model to use
    contents="Explain how AI works in a few words" # Prompt text
)

# Print the generated response
print(response.text)