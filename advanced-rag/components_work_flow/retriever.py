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

from components_work_flow.constantes import *
from components_work_flow.gradeDocument import GradeDocuments
from  components_work_flow.gradeAnswer import GradeAnswer 
from components_work_flow.gradeDocument import GradeDocuments
from components_work_flow.gradeHallucination import GradeHallucination 

loader = PyPDFLoader("C:\devpy\playground\Advanced_Rag_Implementation_with_mistral_langchain_Engineers\data\BABOK_Guide_v3_Member_2015.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 250, chunk_overlap=100
)
docs_split = text_splitter.split_documents(pages)

# Add those splits things to  vectorstore 

vectorstore = Chroma.from_documents(
    documents= docs_split,
    collection_name="rag-chroma",
    embedding=CohereEmbeddings()
)

retriever = vectorstore.as_retriever()
## Retrieval Grader 
llm = ChatCohere()  

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system=  """  You are a grader assessing relevance of a retrieved document to a user question . \n
              If the document contains keyword(s) or semantic meaning related to the question , grade it as relevant .
              Give a binary score 'yes' or 'no' score to indicate wether the document is relevant to the question .
              Note : You have to study well each word of the question related to the question and try to undersdand the question
              according to the user need about the documnent 
              By example :when user ask you to resume the document , any word of this is not in the document but the itself is 
              asking you to do a job about the content of the content . 

           """

grade_prompt = ChatPromptTemplate.from_messages(
    [
    ("system", system),
    ("human","Retrieved document:\n\n User question : {question}")
    ]
)

retrieval_grader= grade_prompt | structured_llm_grader 


question=  """
                  Stratégies de sécurisation de l'application
           """

docs = retriever.invoke(question)
#print(docs)
for doc in docs: 
    doc_text = doc.page_content 
 
    print(retrieval_grader.invoke({"question":question,"document":doc_text}))


prompt=PromptTemplate(input_variables=['context', 'question'], template="Vous etes un assistant pour les taches de reponses aux questions . Si la question est hors portée du contexte du contenu du document demander a l'utilisateur de reformuler sa question ou suggerer lui des questions qui sont relatives aux contenu du document . Note :Tu es responsable de la qualité de reponse que tu fournis a l'utilisateur . En cas de mauvaise reponse donnée , tu peux etre licencié pour cela .Tu dois aussi prendre en compte l'option de la langue tu dois repondre a la question de l'utilisateur en fonction de la langue de sa question si la question a été posée en anglais tu lui repond en anglais ainsi de suite  .\nQuestion: {question} \nContext: {context} \nAnswer: "
                      
  )

rag_chain = prompt | llm | StrOutputParser()

generation = rag_chain.invoke({"context":docs, "question":question})
#print(generation)

# Let's create halucination Grader 

structured_llm_grader = llm.with_structured_output(GradeHallucination)

#Prompt 

system =""" You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
       [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
hallucination_grader = hallucination_prompt | structured_llm_grader
hallucination_grader.invoke({"documents": docs, "generation": generation})
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt 
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
answer_grader.invoke({"question": question,"generation": generation})
