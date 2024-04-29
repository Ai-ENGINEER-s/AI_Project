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

from components_work_flow.edges import *
from components_work_flow.nodes import *
from components_work_flow.graphState import GraphState
from components_work_flow.constantes import *
from components_work_flow.retriever import *

from components_work_flow.graphState import GraphState 
from components_work_flow.nodes import Nodes 
from components_work_flow.edges import Edges 



class LangchainWorkFlow(BaseModel):
    """
    This function defines a workflow for processing a question and potentially retrieving information, 
    grading documents, generating text, or searching the web based on intermediate decisions.

    It utilizes pre-defined components like retriever, grader, generator, and web search 
    (likely defined elsewhere) to achieve this.


    """
    def langChainWorkflow():
        
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", Nodes.retrieve) # retrieve
        workflow.add_node("grade_documents",Nodes.grade_documents) # grade documents
        workflow.add_node("generate", Nodes.generate) # generatae
        workflow.add_node("websearch", Nodes.web_search)  # web search

   
    # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            Edges.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        workflow.add_edge("websearch", "generate")
        workflow.add_conditional_edges(
            "generate",
            Edges.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END, 
                "not useful": "websearch",
            },
            )
            # Compile
        app = workflow.compile()

        inputs = {
                "question": 
        """Problèmes (PB)
        Objectifs (OB)
        Les objectifs du projet pour le système de réservation de billets pour les Jeux Olympiques
        peuvent être définis comme suit  """}

        for output in app.stream(inputs):
            for key, value in output.items():
                pprint(f"Finished running :{key}:")
        return value["generation"]


