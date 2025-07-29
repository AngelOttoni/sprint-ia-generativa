from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import asyncio
import openai
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pathlib import Path
from dotenv import load_dotenv
import os

env_path = os.path.join(os.path.dirname(__file__), '../../Gen_AI_APIs/.env')
load_dotenv(env_path)

search_engine_path = Path(__file__).parent / "search_engine.py"

client = MultiServerMCPClient(
    {
        "books_search_engine": {
            "command": "python",
            "args": [str(search_engine_path)],
            "transport": "stdio",
        }
    }
)

openai_api_key = os.getenv("OPENAI_API_KEY")  # Get API key from environment
model = os.getenv("OPENAI_MODEL")  # Get model name from environment


async def chat():
    llm = ChatOpenAI(model=model, top_p=0.5, api_key=openai_api_key)
    tools = await client.get_tools()
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt="Você é um agente que ajuda usuários a encontrar livros com base em suas solicitações. "
        "Você pode usar tools para buscar informações sobre livros. Lembre-se de traduzir o gênero do livro para o inglês antes de fazer a busca na tool."
        "Conside livros curtos os livros que tem no máximo 130 páginas."
        "Considere livros longos os livros que tem mais de 400 páginas."
        "Na hora de responder escolha os 5 melhores livros baseados nas avaliações dos usuários."
        "Para encontrar informações de um livro específico, passe o nome em inglês, do jeito que o usuário escreveu e com codificação UTF-8 para a tool de buscar por nome."
        "Se o usuário pesquisar livros com algum termo específico, use a ferramenta do goodreads-book-scraper do apify e passe o search term em inglês.",
    )

    print("Digite a sua solicitação...")

    while True:
        user_input = input("> ")

        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]}
        )
        print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(chat())
