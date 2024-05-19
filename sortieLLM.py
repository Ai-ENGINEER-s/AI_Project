
# Import des modules nécessaires
from langchain_community.document_loaders import PyPDFLoader 
from langchain_chroma import Chroma 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph , END 
from pprint import pprint
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain.pydantic_v1 import BaseModel , Field
from langchain_groq import ChatGroq
from langchain.prompts import (ChatPromptTemplate, PromptTemplate)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
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
from typing_extensions import TypedDict
from typing import List

# Chargement des variables d'environnement
dotenv_path =r"C:\Users\BARRY\Desktop\AI-WorkSpace\.env"
load_dotenv(dotenv_path)

# Initialisation de l'outil de recherche web
web_search_tool = TavilySearchResults(k=3)

# Chemin du document
doc_path = r"C:\Users\BARRY\Desktop\AI-WorkSpace\langgraph-course\Candidature_ post_stage_développeur_web.pdf"

# Chargement du contenu du document
doc_content_load = PyPDFLoader(doc_path).load()

# Fonction de scission du document
text_splitter_function= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Scission du document
doc_split_content = text_splitter_function.split_documents(doc_content_load)

# Fonction d'embedding
embeddings_function = OpenAIEmbeddings(api_key="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")

# Création du vectore store
vectorestore = Chroma.from_documents(
    embedding=embeddings_function, 
    collection_name="rag-chroma",
    persist_directory='chroma/db',
    documents=doc_split_content
)

# Retriever
retriever = vectorestore.as_retriever()

# Modèle de notation des documents
class GradeDocuments(BaseModel):
    """Score binaire pour vérification de la pertinence des documents récupérés."""
    binary_score: str = Field(description="Les documents sont pertinents pour la question, 'oui' ou 'non'")

# Création de l'instance LLM
llm = ChatOpenAI(api_key="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")
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt pour la notation des documents
system = """Vous êtes un évaluateur évaluant la pertinence d'un document récupéré par rapport à une question utilisateur. \n 
    Si le document contient des mots-clés ou un sens sémantique lié à la question, attribuez-lui une note pertinente. \n
    Donnez une note binaire 'oui' ou 'non' pour indiquer si le document est pertinent pour la question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Document récupéré: \n\n {document} \n\n Question de l'utilisateur: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

# Question et récupération des documents pertinents
question = "+212 7777 305 40"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

### Génération

# Prompt
prompt = hub.pull("rlm/rag-prompt")
print(prompt)

# Post-traitement
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chaîne
rag_chain = prompt | llm | StrOutputParser()

# Exécution
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)

# Modèle de notation des hallucinations
class GradeHallucinations(BaseModel):
    """Score binaire pour la présence d'hallucination dans la réponse générée."""
    binary_score: str = Field(description="La réponse est ancrée dans les faits, 'oui' ou 'non'")

# Instance LLM
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt pour la notation des hallucinations
system = """Vous êtes un évaluateur évaluant si une génération LLM est ancrée dans / supportée par un ensemble de faits récupérés. \n 
     Donnez une note binaire 'oui' ou 'non'. 'Oui' signifie que la réponse est ancrée dans / supportée par l'ensemble des faits."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Ensemble de faits: \n\n {documents} \n\n Génération LLM: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
print(hallucination_grader.invoke({"documents": docs, "generation": generation}))


# Modèle de notation de la réponse
class GradeAnswer(BaseModel):
    """Score binaire pour évaluer si la réponse répond à la question."""
    binary_score: str = Field(description="La réponse répond à la question, 'oui' ou 'non'")

# Instance LLM
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt pour la notation de la réponse
system = """Vous êtes un évaluateur évaluant si une réponse répond / résout une question \n 
     Donnez une note binaire 'oui' ou 'non'. 'Oui' signifie que la réponse résout la question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question de l'utilisateur: \n\n {question} \n\n Génération LLM: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
answer_grader.invoke({"question": question,"generation": generation})


# État du graphe
class GraphState(BaseModel):
    """
    Représente l'état de notre graphe.

    Attributs:
        question: question
        generation: génération LLM
        web_search: effectuer une recherche web
        documents: liste de documents 
    """
    question : str
    generation : str
    web_search : str
    documents : List[str]

# Nodes

def retrieve(state):
    """
    Récupère les documents à partir du vectorestore

    Args:
        state (dict): L'état actuel du graphe

    Returns:
        state (dict): Nouvelle clé ajoutée à l'état, documents, contenant les documents récupérés
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Récupération
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Génère une réponse en utilisant RAG sur les documents récupérés

    Args:
        state (dict): L'état actuel du graphe

    Returns:
        state (dict): Nouvelle clé ajoutée à l'état, génération, contenant la génération LLM
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # Génération RAG
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Détermine si les documents récupérés sont pertinents pour la question
    Si un document n'est pas pertinent, nous définirons un indicateur pour exécuter une recherche web
    Si un seul document pertinent est trouvé, le retourner immédiatement

    Args:
        state (dict): L'état actuel du graphe

    Returns:
        state (dict): Documents filtrés et état web_search mis à jour
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    # Notation de chaque document
    filtered_docs = []
    web_search = "No"
    relevant_count = 0
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        # Document pertinent
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            relevant_count += 1
        # Document non pertinent
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # Nous définissons un indicateur pour indiquer que nous voulons exécuter une recherche web
            web_search = "Yes"
    
    # S'il n'y a qu'un seul document pertinent, le retourner immédiatement
    if relevant_count == 1:
        print("---ONLY ONE RELEVANT DOCUMENT FOUND---")
        return {"documents": filtered_docs, "question": question, "web_search": "No"}
    
    # S'il n'y a aucun document pertinent, activer la recherche web
    if relevant_count == 0:
        print("---NO RELEVANT DOCUMENTS FOUND---")
        return {"documents": filtered_docs, "question": question, "web_search": "Yes"}
    
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def web_search(state):
    """
    Recherche web basée sur la question

    Args:
        state (dict): L'état actuel du graphe

    Returns:
        state (dict): Résultats web ajoutés aux documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Recherche web
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


### Arêtes

def decide_to_generate(state):
    """
    Détermine s'il faut générer une réponse ou effectuer une recherche web

    Args:
        state (dict): L'état actuel du graphe

    Returns:
        str: Décision binaire pour appeler le prochain noeud
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # Tous les documents ont été filtrés, vérification de la pertinence
        # Nous allons générer une nouvelle requête
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # Nous avons des documents pertinents, donc générer une réponse
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Détermine si la génération est ancrée dans le document et répond à la question

    Args:
        state (dict): L'état actuel du graphe

    Returns:
        str: Décision pour appeler le prochain noeud
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    # Vérifier les hallucinations
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Vérifier la réponse à la question
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

# Création du graphe
workflow = StateGraph(GraphState)

# Définition des noeuds
workflow.add_node("retrieve", retrieve) # Récupération
workflow.add_node("grade_documents", grade_documents) # Notation des documents
workflow.add_node("generate", generate) # Génération
workflow.add_node("websearch", web_search)  # Recherche web

# Construction du graphe
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

# Compilation
app = workflow.compile()

# Exécution du graphe
inputs = {"question": "Nous veneons lde ternoaur eaporupoaizeru ?,"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])
