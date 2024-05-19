"---------Chat models import---------"
from langchain_openai import ChatOpenAI 
from langchain_groq import ChatGroq 
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere 

"--------Embeddings models import -----------"

from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)

"---------Others modules ---------------"

from langchain_community.document_loaders import PyPDFLoader 
from langchain_chroma import Chroma 
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_community.tools.tavily_search import TavilySearchResults
from config.settings import openai_embeddings


doc_path = r"C:\devpy\playground\advanced-rag\data\BABOK_Guide_v3_Member_2015.pdf"

web_search_tool = TavilySearchResults(k=3)

doc_content_load = PyPDFLoader(doc_path).load()

# text_splitter_function : function responsable de la scission du document 

text_splitter_function= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

#  doc_split_content : la variable contenant qui stocke 450 characters de note document pdf 

doc_split_content = text_splitter_function.split_documents(doc_content_load)



# embeddings_function : fonction responsable de transformer doc_split_content en un vecteur numerique qui seront stockés dans le vectore store 


embeddings_function = openai_embeddings

vectorestore = Chroma.from_documents(
    embedding=embeddings_function, 
    collection_name="rag-chroma",
    persist_directory='chroma/db',
    documents=doc_split_content
)


#retriever : variable responsable de la recuperations des documents pertinents   

retriever = vectorestore.as_retriever()
