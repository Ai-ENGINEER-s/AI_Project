from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults




dotenv_dir = r"C:\Users\BARRY\Desktop\AI-WorkSpace\advanced-rag\.env"
load_dotenv(dotenv_dir)

web_search_tool =  TavilySearchResults(k=3, api_key="TAVILY_API_KEY")