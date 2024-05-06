from langchain.prompts import PromptTemplate 
from langchain_groq import ChatGroq 
from langchain_chroma import Chroma 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader 
from dotenv import load_dotenv 
from langchain_chroma import Chroma 
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)

load_dotenv()


doc_path = r"C:\devpy\playground\langgraph-course\Candidature_ post_stage_développeur_web.pdf"

doc_content= PyPDFLoader(doc_path).load()

split_function = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=100)

split_doc_content = split_function.split_documents(doc_content)

embedding_function = SentenceTransformerEmbeddings()

vectorstore= Chroma.from_documents(
    split_doc_content, 
    embedding = embedding_function,

)

llm = ChatGroq()


template_content =       """  Vous etes un assistant pour les taches de reponses aux questions . Si la question est hors portée du contexte du contenu du document demander a l'utilisateur de reformuler sa question ou suggerer lui des questions qui sont relatives aux contenu du document .
Note N°2 : Tu dois repondre en francais                     
Note N°1 :Tu es responsable de la qualité de reponse que tu fournis a l'utilisateur . En cas de mauvaise reponse donnée , tu peux etre licencié pour cela . {question} \nContext: {context} \nAnswer: """
 


prompt = PromptTemplate(
    input_variables=['context', 'question'], 
    template = template_content 
)

question = "Donne moi le resume de ce document "

chain_rag = prompt | llm 

response = chain_rag.invoke({"context": split_doc_content, "question": question})

# print(response)


