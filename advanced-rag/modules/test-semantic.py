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

"---------Others moDdules ---------------"

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
from pprint import pprint
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere
from typing_extensions import TypedDict
from typing import List
from config.settings import openai_llm
"-------------"
from grade_classes import GradeAnswer , GradeDocuments ,GradeHallucinations
from modules.prompts import retrieval_grader,hallucination_grader,answer_grader,rag_chain
from config.settings import openai_llm ,web_search_tool
from modules.document_retrieval import retriever
# Data model


question = """ give me all the point about Table 10.16.1: Simple Decision Matrix in the document"""
docs = retriever.invoke(question)




# graph state 

class GraphState(BaseModel):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents 
    """
    question: str
    generation: str
    web_search: str
    documents: List[Document]



    ### Nodes

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
    Sets web search flag based on document relevance.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered documents, updated web_search state, potentially web search results
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    relevant_count = 0
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            relevant_count += 1
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")

    # Update web search flag based on relevance count
    if relevant_count == 0:
        print("---NO RELEVANT DOCUMENTS FOUND---")
        web_search = "Yes"
        # Call web_search function here if needed (assuming it returns additional documents)
        # filtered_docs.extend(web_search(state))

    # No early return - process all documents
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


### Edges

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
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"



 
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve) # retrieve
workflow.add_node("grade_documents", grade_documents) # grade documents
workflow.add_node("generate", generate) # generatae
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
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)

# Compile
app = workflow.compile()


from pprint import pprint
inputs = {"question": """Office Productivity Tools and Technology """}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])