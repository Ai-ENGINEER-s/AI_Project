"----------- Let's put here all things we want ------------"



# c'est quoi langgraph : langgraph est deja un concept d'agent developper par langchain 
# pour permettre aux utilisateur d'avoir plus de controle sur la sortie des LLMs 



"---------Chat models import---------"
#pip install sentence-transformers
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
from langchain.pydantic_v1 import BaseModel , Field, validator 
from langchain_groq import ChatGroq
from langchain.prompts import (ChatPromptTemplate, PromptTemplate)
from langchain import hub
from dotenv import load_dotenv
from typing_extensions import TypedDict 
from typing import  List 
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph , END 
from langchain.output_parsers import PydanticOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


load_dotenv()
web_search_tool = TavilySearchResults(k=3, api_key="TAVILY_API_KEY")


doc_path = r"C:\Users\BARRY\Desktop\AI-WorkSpace\Draft\data\analyseoptimiséedusystèmeRAG.pdf"



doc_content = PyPDFLoader(doc_path).load()
doc_splitter_function = RecursiveCharacterTextSplitter(chunk_size = 450,chunk_overlap = 100 )
doc_split_content = doc_splitter_function.split_documents(doc_content)

embeddings_sentence = SentenceTransformerEmbeddings()
openai_embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    embedding=embeddings_sentence,
    documents= doc_split_content,
    persist_directory="/barry_db"
)


doc_relevant_retriever = vectorstore.as_retriever()

question =""""Exemple de Documentation avec Métadonnées:
○ Section 1 (Traitement des commandes"""

# for d in doc_content: 
#      relevant =doc_relevant_retriever.invoke(question)

    #  print(relevant)

groq_llm = ChatGroq()

openai_llm = ChatOpenAI(api_key="sk-proj-1bELHfK7R8tA5wzNZiXrT3BlbkFJlsWmAgFKgsCM3swObNuh")


class GradeDocuments(BaseModel):
     """Binary score for relevance check on retrieved documents """

     binary_score:str = Field(description="Documents are relevant to the question , 'yes' or 'no'")
 


parser = PydanticOutputParser(pydantic_object=GradeDocuments)

# structured_llm_grader =openai_llm.with_structured_output(GradeDocuments)

# Prompt 
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt |groq_llm| parser
question = "agent memory"
docs = doc_relevant_retriever.invoke(question)
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))