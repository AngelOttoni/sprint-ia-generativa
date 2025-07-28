# Test script to verify the OpenAI API key setup and connection
# Sends a basic request to ensure the API is working correctly

import os # Access environment variables
from dotenv import load_dotenv # Load variables from .env file
from openai import OpenAI # OpenAI Python SDK client

# Load environment variables from .env into the system
load_dotenv() 

openai_key = os.getenv('OPENAI_API_KEY') # Get API key from environment
model = os.getenv('OPENAI_MODEL') # Get model name from environment

# Initialize OpenAI client with API key
client = OpenAI(api_key=openai_key) 

response = client.responses.create(
    model=model,  # Specify the model to use
    input="Write a one-sentence bedtime story about a unicorn." # Text prompt
)

print(response.output_text) # Print the generated text response