from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv 
from langchain_community.tools.tavily_search import TavilySearchResults

dotenv_path=r"C:\devpy\playground\advanced-rag\.env"
load_dotenv(dotenv_path)

web_search_tool = TavilySearchResults(k=3)


openai_llm  = ChatOpenAI(api_key="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")
openai_embeddings = OpenAIEmbeddings( api_key="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")