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
from pprint import pprint
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere
load_dotenv()
web_search_tool = TavilySearchResults(k=3, api_key="TAVILY_API_KEY")



# initialisons le chemin de notre fichier 

doc_path = r"D:\devpy\playground\advanced-rag\data\BABOK_Guide_v3_Member_2015.pdf"


doc_content_load = PyPDFLoader(doc_path).load()

# text_splitter_function : function responsable de la scission du document 

text_splitter_function= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

#  doc_split_content : la variable contenant qui stocke 450 characters de note document pdf 

doc_split_content = text_splitter_function.split_documents(doc_content_load)



# embeddings_function : fonction responsable de transformer doc_split_content en un vecteur numerique qui seront stockés dans le vectore store 


embeddings_function = OpenAIEmbeddings( api_key="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")

vectorestore = Chroma.from_documents(
    embedding=embeddings_function, 
    collection_name="rag-chroma",
    persist_directory='chroma/db',
    documents=doc_split_content
)


#retriever : variable responsable de la recuperations des documents pertinents   


retriever = vectorestore.as_retriever()


#  Partie I GradeDocuments notre modele de sortie , dont le role est de nous retourner un resultat binaire 'yes' or 'no' 

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved ."""
    binary_score :str =Field(description="Documents are relevant to the question , 'yes' or 'no' ")


# initialisation de notre llm 

llm = ChatOpenAI(temperature=0, api_key="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")

#  structure_llm_grader : retourne une valeur binaire en fonction de la pertinance de la question 


structure_llm_grader_gradeDocument = llm.with_structured_output(GradeDocuments) 



# Initialisation de notre grade_promt  


system = """ You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_promt= ChatPromptTemplate.from_messages(
    [
        ("system", system), 
        ("human", "Retrieved document :\n\n User question :{question}")

    ]
)

# Construction de notre chaine qui prend le prompt et la sortie binaire 

retrieval_grader = grade_promt | structure_llm_grader_gradeDocument



question ="Ask your question about your document"

# docs : recupere les documents en fonction de la question demandéee 

docs = retriever.invoke(question)


#  construction de notre chain pour avoir la reponse du LLM .Ici nous lui passons le context quis est les docs et aussi la question . Donc il nous produira une reponse a travers les docs recus et la question . 

template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"

prompt = PromptTemplate(
   input_variables=['context', 'question'] ,
    template= template
)

rag_chain = prompt|llm 

generation = rag_chain.invoke({"context": docs, "question": question})
# print(generation)



# Partie II Grade Hallucination  Ici on donne les docs et la generation du LLM et on procedera a une evaluation en fonctions des documents recuperés et la reponse du LLM  


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

structured_llm_grader_hallucination = llm.with_structured_output(GradeHallucinations)

# Prompt 

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)


hallucination_grader = hallucination_prompt | structured_llm_grader_hallucination
hallucination_grader.invoke({"documents": docs, "generation": generation})




# Partie III : GRADE ANSWER  ici on donne la question et la generation puis on fait une evaluation de la pertinence de la  reponse du LLM  par rapport a la question de l'utilisateur  : 


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")


structured_llm_grader_answer = llm.with_structured_output(GradeAnswer)

# Prompt 
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader_answer
answer_grader.invoke({"question": question,"generation": generation})




# GraphState : l'objectif de cette clase c'est de nous retourner une sortie dictionnaire dont les valeurs des clés sont : la question , la generation du LLM , le contenu deu web_search , et la liste des documents recuperés .

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


# retrieve : Cette fonction est un etat qui retourne les documents recupérés en fonction de la question et aussi la question elle meme . Nous la passerons au 1er node . 

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


# generate : cette fonction est un etat qui retourne la reponse du LLM , c'est a dire qu'elle prend les documents recuperés et question et genere une reponse . Au final nous retournons la question , les documents recuperes et la reponse  du LLM 


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
    print("-------THIS IS THE CONTENT OF THE LLM -------: ")
    print(generation)
    return {"documents": documents, "question": question, "generation": generation}


# grade_documents :  cette fonction prend les documents recuperés et procedera a  une evaluation de la pertinence du document recuperé par rapport a la question et si un seul document n'est par relevant nous declachons le web search tool . Dans le cas contraire nous retournous la liste des documents filtrés . Elle nous retourne la liste des documents filtrés , la question , et l'etat deu web_search 

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
            print("-----THIS IS THE CONTENT OF THE RELEVANT DOCUMENT --- :")
            print(f"{d.page_content}\n")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

# web_search : cette fonction nous retourne le resultat des recherches effectués par le web search tool dans une variables ici nommée documents . Au final nous retournous les documents , la question et aussi l'etat du web_search soit oui soit non .


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


#  grade_generation_v_documents_and_question  : 

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

# decide_to_generate : cette fonction prend la question , la liste des documents filtrés , et l'etat du web_search tool , si le resultat du web_search tool est non donc la fonction retourn la reponse du LLM dans le cas contraire on declenche le web_search tool . 

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
        
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


# construction de notre workflow avec l'initialisation StateGraph .


workflow = StateGraph(GraphState)

# Definissons les nodes : les nodes prennenent des etats ici ce sont nos fonctions . 


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
inputs = {"question":" List the Business Process Management (BPM) frameworks mentioned in the passage.  "}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])









# class AdaptiveRagTool(BaseModel):
    
#     def call_function():
#         workflow.add_node("retrieve", retrieve)  # retrieve
#         workflow.add_node("grade_documents", grade_documents)  # grade documents
#         workflow.add_node("generate", generate)  # generatae
#         workflow.add_node("websearch", web_search)  # web search

#         # Build graph
#         workflow.set_entry_point("retrieve")
#         workflow.add_edge("retrieve", "grade_documents")
#         workflow.add_conditional_edges(
#             "grade_documents",
#             decide_to_generate,
#             {
#                 "websearch": "websearch",
#                 "generate": "generate",
#             },
#         )
#         workflow.add_edge("websearch", "generate")
#         workflow.add_edge("generate", END)
#         app = workflow.compile()
        
#         inputs = {"question": """  Give me a resume of the concept of  Business Analysis Planning and Monitoring in the document """}
#         for output in app.stream(inputs):
#             for key, value in output.items():
#                 pprint(f"Finished running: {key}:")
                
#         return value["generation"]

