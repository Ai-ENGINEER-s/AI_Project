"---------Chat models import---------"
from langchain_openai import ChatOpenAI 
from langchain_groq import ChatGroq 
from langchain_cohere import ChatCohere 

"--------Embeddings models import -----------"
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)

"---------Others modules ---------------"
from langchain_community.document_loaders import PyPDFLoader 
from langchain_chroma import Chroma 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.prompts import (ChatPromptTemplate, PromptTemplate)
from langchain import hub
from dotenv import load_dotenv
from typing_extensions import TypedDict 
from typing import List 
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END 
from pprint import pprint
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
import os
import agentops

# Load environment variables and initialize AgentOps
load_dotenv()
AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")
if not AGENTOPS_API_KEY:
    raise ValueError("AGENTOPS_API_KEY not found in environment variables")
agentops.init(AGENTOPS_API_KEY)

@agentops.record_function('initialize_tools_and_models')
def initialize_tools_and_models():
    web_search_tool = TavilySearchResults(k=3, api_key=os.getenv("TAVILY_API_KEY"))
    doc_path = r"D:\devpy\playground\advanced-rag\data\BABOK_Guide_v3_Member_2015.pdf"
    doc_content_load = PyPDFLoader(doc_path).load()
    text_splitter_function = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    doc_split_content = text_splitter_function.split_documents(doc_content_load)
    embeddings_function = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorestore = Chroma.from_documents(
        embedding=embeddings_function, 
        collection_name="rag-chroma",
        persist_directory='chroma/db',
        documents=doc_split_content
    )
    retriever = vectorestore.as_retriever()
    llm = ChatOpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    return web_search_tool, retriever, llm

web_search_tool, retriever, llm = initialize_tools_and_models()

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

@agentops.record_function('setup_graders')
def setup_graders():
    structure_llm_grader_gradeDocument = llm.with_structured_output(GradeDocuments)
    system = """You are a grader assessing relevance of a retrieved document to a user question. 
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. 
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        ("human", "Retrieved document:\n\n User question:{question}")
    ])
    retrieval_grader = grade_prompt | structure_llm_grader_gradeDocument

    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:"""
    prompt = PromptTemplate(input_variables=['context', 'question'], template=template)
    rag_chain = prompt | llm

    structured_llm_grader_hallucination = llm.with_structured_output(GradeHallucinations)
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
         Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])
    hallucination_grader = hallucination_prompt | structured_llm_grader_hallucination

    structured_llm_grader_answer = llm.with_structured_output(GradeAnswer)
    system = """You are a grader assessing whether an answer addresses / resolves a question 
         Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ])
    answer_grader = answer_prompt | structured_llm_grader_answer

    return retrieval_grader, rag_chain, hallucination_grader, answer_grader

retrieval_grader, rag_chain, hallucination_grader, answer_grader = setup_graders()

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]

@agentops.record_function('retrieve')
def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

@agentops.record_function('generate')
def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    print("-------THIS IS THE CONTENT OF THE LLM -------: ")
    print(generation)
    return {"documents": documents, "question": question, "generation": generation}

@agentops.record_function('grade_documents')
def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            print("-----THIS IS THE CONTENT OF THE RELEVANT DOCUMENT --- :")
            print(f"{d.page_content}\n")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

@agentops.record_function('web_search')
def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}

@agentops.record_function('grade_generation_v_documents_and_question')
def grade_generation_v_documents_and_question(state):
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
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

@agentops.record_function('decide_to_generate')
def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]
    if web_search == "Yes":
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

@agentops.record_function('create_workflow')
def create_workflow():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("websearch", web_search)
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
    return workflow.compile()

@agentops.record_function('main')
def main():
    app = create_workflow()
    inputs = {"question": "List the Business Process Management (BPM) frameworks mentioned in the passage."}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    pprint(value["generation"])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        agentops.log_error(str(e))
    finally:
        agentops.end_session('Success')