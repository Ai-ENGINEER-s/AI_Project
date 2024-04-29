from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults




dotenv_dir = r"C:\devpy\playground\Advanced_Rag_Implementation_with_mistral_langchain_Engineers\.env"
load_dotenv(dotenv_dir)

web_search_tool = TavilySearchResults(k=3)