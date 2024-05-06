"""
key concepts : 

1- What is a prompt ? 
2- How many types of prompts do we have ? 
3- How to implement a prompt  ? 
 

"""


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv 
from langchain_groq import ChatGroq 
from langchain.prompts import ChatPromptTemplate


"""
LLMs accept strings as input or objects that can be coerced to string prompts , including List[baseMessage] and promptValue . 

Chat models in LangChain work with different message types such as AIMessage, HumanMessage, SystemMessage, FunctionMessage, and ChatMessage (with an arbitrary role parameter). Generally, HumanMessage, AIMessage, and SystemMessage are the most frequently used.
"""
load_dotenv()

openai_llm =ChatOpenAI()
groq_llm= ChatGroq()

response = groq_llm.invoke("List the seven wonders of the world .")

# chat models work with different type of messages 
# such as HumanMessage AiMessage SystemMessage FunctionMessage 

from langchain.schema import HumanMessage , SystemMessage



