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
from langchain.pydantic_v1 import BaseModel , Field
from langchain_groq import ChatGroq
from langchain.prompts import (ChatPromptTemplate, PromptTemplate)
from langchain import hub
from dotenv import load_dotenv
from typing_extensions import TypedDict 
from typing import  List 
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph , END 

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document


load_dotenv()
web_search_tool = TavilySearchResults(k=3, api_key="TAVILY_API_KEY")



doc_path = r"C:/Users/BARRY/Desktop/AI-WorkSpace/langgraph-implementations-projects/data/BABOK_Guide_v3_Member_2015.pdf"

doc_content_load = PyPDFLoader(doc_path).load()

text_splitter_function= RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=0)

doc_split_content = text_splitter_function.split_documents(doc_content_load)

# for d in doc_content_load : 
#     print(d)

embeddings_function = SentenceTransformerEmbeddings()

vectorestore = Chroma.from_documents(
    embedding=embeddings_function, 
    collection_name="rag-chroma",
    documents=doc_split_content
)

retriever = vectorestore.as_retriever()
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved ."""
    binary_score :str =Field(description="Documents are relevant to the question , 'yes' or 'no' ")

llm = ChatGroq(temperature=0, )
structure_llm_grader = llm.with_structured_output(GradeDocuments) 

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_promt= ChatPromptTemplate.from_messages(
    [
        ("system", system), 
        ("human", "Retrieved document :\n\n User question :{question}")

    ]
)

retrieval_grader = grade_promt | structure_llm_grader

question=""" 
 Barrysanoussa19@gmail.com  
  """ 


docs = retriever.invoke(question)


for d in docs : 
    d.page_content
    # print(f"This is the content of the relevant docs to the question : {d}\n")


# llm= ChatGroq()


template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"

prompt = PromptTemplate(
   input_variables=['context', 'question'] ,
    template= template
)

rag_chain = prompt | llm 

generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents 
    """
    question : str
    generation : str
    web_search : str
    documents : List[str]


def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("websearch", web_search)  # web search

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

from pprint import pprint
inputs = {"question": """  Give me a resume of the concept of  Business Analysis Planning and Monitoring in the document """}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])