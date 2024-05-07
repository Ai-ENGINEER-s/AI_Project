from langchain.tools.base_tool import BaseTool
from langchain.graph import StateGraph, END
from langchain.schema import Document
from typing import List
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma 
from langchain_openai import ChatOpenAI 
from langchain.prompts import ChatPromptTemplate, PromptTemplate

load_dotenv()

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved ."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no' ")

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
    documents: List[str]

class MyTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.web_search_tool = TavilySearchResults(k=3, api_key="TAVILY_API_KEY")
        self.llm = ChatOpenAI(temperature=0, api_key="sk-proj-1bELHfK7R8tA5wzNZiXrT3BlbkFJlsWmAgFKgsCM3swObNuh", model= "gpt-3.5-turbo")
        self.structure_llm_grader = self.llm.with_structured_output(GradeDocuments)
        self.grade_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a grader assessing relevance of a retrieved document to a user question. \n If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."), 
            ("human", "Retrieved document :\n\n User question :{question}")
        ])
        self.retrieval_grader = self.grade_prompt | self.structure_llm_grader
        self.text_splitter_function = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=0)
        self.embeddings_function = SentenceTransformerEmbeddings()

        # Load document
        doc_path = r"langgraph-implementations-projects/data/analyseoptimiséedusystèmeRAG.pdf"
        doc_content_load = PyPDFLoader(doc_path).load()
        self.doc_split_content = self.text_splitter_function.split_documents(doc_content_load)

        # Vectorization
        self.vectorestore = Chroma.from_documents(
            embedding=self.embeddings_function, 
            collection_name="rag-chroma",
            documents=self.doc_split_content
        )
        self.retriever = self.vectorestore.as_retriever()

        self.prompt = PromptTemplate(
            input_variables=['context', 'question'],
            template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
        )
        self.rag_chain = self.prompt | self.llm 

        self.workflow = StateGraph(GraphState)
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("grade_documents", self.grade_documents)
        self.workflow.add_node("generate", self.generate)
        self.workflow.add_node("websearch", self.web_search)
        self.workflow.set_entry_point("retrieve")
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        self.workflow.add_edge("websearch", "generate")
        self.workflow.add_edge("generate", END)

    def retrieve(self, state):
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}

    def grade_documents(self, state):
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = self.retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score.binary_score
            if grade.lower() == "yes":
                filtered_docs.append(d)
            else:
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    def web_search(self, state):
        question = state["question"]
        documents = state["documents"]
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]
        return {"documents": documents, "question": question}

    def generate(self, state):
        question = state["question"]
        documents = state["documents"]
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def decide_to_generate(self, state):
        question = state["question"]
        web_search = state["web_search"]
        if web_search == "Yes":
            return "websearch"
        else:
            return "generate"

    def execute(self, input_data):
        output = self.workflow.invoke(input_data)
        with open("output.txt", "w") as f:
            f.write(str(output))
        return output

tool = MyTool()
input_data = {"question": "Parle moi de la Généralisation pour d'autres Secteurs discuté dans le contenu du document  "}
tool.execute(input_data)
