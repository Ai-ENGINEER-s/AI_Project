print("--------L----A-----N-----G---G-----R----A-----P------H------")

from dotenv import load_dotenv 
from langchain_openai import ChatOpenAI
dotenv_dir = r"C:\Users\BARRY\Desktop\AI-WorkSpace\langgraph-full-course\.env"

load_dotenv(dotenv_dir)


llm = ChatOpenAI()

print(llm.invoke("tell me about BURKINA FASO ").content )

