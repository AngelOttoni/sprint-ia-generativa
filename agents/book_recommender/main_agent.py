import os
import asyncio
import openai
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the .env file
env_path = os.path.join(os.path.dirname(__file__), '../../Gen_AI_APIs/.env')
load_dotenv(env_path)

# Path to the search_engine.py script, which will be registered as a tool
search_engine_path = Path(__file__).parent / 'book_search_tool.py'

# Retrieve API keys and model name from environment variables
key_openai = os.getenv('OPENAI_API_KEY')
key_apify = os.getenv('APIFY_API_KEY')
openai_model_name = os.getenv('OPENAI_MODEL')

# Setup MultiServerMCPClient with two tools:
# 1. 'books_search_tool' runs the local search_engine.py script via stdio
# 2. 'apify' connects to Apify's goodreads-book-scraper actor via SSE with authorization
mcp_client = MultiServerMCPClient(
    {
        'books_search_tool': {
            'command': 'python',
            'args': [str(search_engine_path)],
            'transport': 'stdio',
        },
        'goodreads_scraper_tool': {
            'transport': 'sse',
            'url': 'https://mcp.apify.com/sse?actors=runtime/goodreads-book-scraper',
            'headers': {'Authorization': f'Bearer {key_apify}'},
        },
    }
)


async def chat():
    # Initialize the OpenAI chat model with specified parameters
    llm = ChatOpenAI(model=openai_model_name, top_p=0.5, api_key=key_openai)

    # Fetch tools registered in the MCP client
    tools = await mcp_client.get_tools()

    # Create a REACT agent with the LLM and tools, including an in-memory checkpoint saver
    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=InMemorySaver(),
        prompt=(
            'You are an agent who helps users find books based on their requests. '
            'You can use tools to fetch book information. Remember to translate the book genre to English before searching. '
            'Consider books with up to 130 pages as short books. '
            'Consider books with more than 400 pages as long books. '
            'Choose the top 5 books based on user ratings when responding. '
            'For specific book information, pass the book title in English exactly as the user wrote it, using UTF-8 encoding, to the name search tool. '
            'If the user searches for books by a specific term, use the Apify goodreads-book-scraper tool and pass the search term in English.'
        ),
    )

    print(
        "Enter your request (e.g. 'I want a short romance novel'). Use 'quit' to exit.\n"
    )

    # Interactive loop to process user inputs
    while True:
        user_input = input("> ")

        if user_input.lower() == 'quit':
            print("Leaving...")
            break

        config = {'configurable': {'thread_id': '1'}}

        # Send the user's message to the agent asynchronously and wait for the response
        response = await agent.ainvoke(
            {'messages': [{'role': 'user', 'content': user_input}]}, config=config
        )

        print("\nAgent Response:\n")
        print(response['messages'][-1].content)
        print("\n")


if __name__ == '__main__':
    asyncio.run(chat())
