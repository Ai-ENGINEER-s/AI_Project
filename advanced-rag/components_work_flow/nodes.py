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

from components_work_flow.retriever import *


class Nodes:
    def retrieve(state):
        """
        Retrieve documents from vecstore 
        Args : 
        state (dict): New key added to state , documents , that contains 
        retrieved documents 
        """
        print("----RETRIEVE-----")
        question = state["question"]
        # Retrieval 
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(state):
        """
        Generate the answer using RAG on retrieve documents 
        Args : 
        state (dict) : The current graph 
        Returns : 
        state (dict) : New key added to state , generation that contains LLM generation 
        """

        print("-----GENERATE-----")
        question = state["question"]
        documents = state["documents"]

        # RAG Generation 
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
