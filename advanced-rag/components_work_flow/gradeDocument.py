from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader 
from langchain_community.vectorstores import Chroma 
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere import ChatCohere 
from langchain_core.pydantic_v1 import BaseModel , Field 
from dotenv import load_dotenv 
from langchain import hub 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate 
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from langchain.schema import Document

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END , StateGraph 
from pprint import pprint
from langchain.schema import Document 
from langchain.tools import tool 
import os



class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents ."""
    binary_score:str = Field(description="Documents are relevant to the question , 'yes' or 'no' ")
