"""
pip install -U --quiet  : 

-U : stand for update 

--quiet : stand for dont display detailed progress .
When you want to keep output clean and consice 




--------- Corrective RAG implementation ------------


So the main idea of corrective RAG , we take the user question we made research and then we return relevant documents  and after that 
we return the final answer to our user if irrelevant docs so 
we fall back on web search tool that we will implement . 



# How retriever function works ?

method) def as_retriever(**kwargs: Any) -> VectorStoreRetriever
Return VectorStoreRetriever initialized from this VectorStore.

Args:
    search_type (Optional[str]): Defines the type of search that
        the Retriever should perform. Can be "similarity" (default), "mmr", or "similarity_score_threshold".
    search_kwargs (Optional[Dict]): Keyword arguments to pass to the
        search function. Can include things like:
            k: Amount of documents to return (Default: 4)
            score_threshold: Minimum relevance threshold
                for similarity_score_threshold
            fetch_k: Amount of documents to pass to MMR algorithm (Default: 20)
            lambda_mult: Diversity of results returned by MMR;

            

            Returns:
    VectorStoreRetriever: Retriever class for VectorStore.

Examples:


    # Retrieve more documents with higher diversity
    # Useful if your dataset has many similar documents
    docsearch.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 6, 'lambda_mult': 0.25}
    )

     # Fetch more documents for the MMR algorithm to consider
    # But only return the top 5
    docsearch.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 5, 'fetch_k': 50}
    )

    
    # Only retrieve documents that have a relevance score
    # Above a certain threshold
    docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.8}
    )

    # Only get the single most similar document from the dataset
    docsearch.as_retriever(search_kwargs={'k': 1})



    # Only get the single most similar document from the dataset
    docsearch.as_retriever(search_kwargs={'k': 1})

    # Use a filter to only retrieve documents from a specific paper
    docsearch.as_retriever(
        search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
    )

    
    prompt must always preceed the llm 

"""
# import


#pip install sentence-transformers
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from  langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


load_dotenv()

# load the document and split it into chunks
doc_path = r"C:\devpy\playground\langgraph-course\Candidature_ post_stage_développeur_web.pdf"
loader = PyPDFLoader(doc_path)
documents = loader.load()
llm_openai = ChatCohere(api_key="OPENAI_API_KEY")
llm_cohere = ChatCohere()
llm_groq = ChatGroq ()
# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
vectorstore = Chroma.from_documents(docs, 
                                    collection_name="Rag-chroma",
                                    embedding=embedding_function)




# retriever = vectorstore.as_retriever()

question=""" Donne moi un resumé de ce document   """

# docs_retrieved = retriever.invoke(question) 

# for doc in docs_retrieved:
#     page_text = doc.page_content


template_content = """  Vous etes un assistant pour les taches de reponses aux questions . Si la question est hors portée du contexte du contenu du document demander a l'utilisateur de reformuler sa question ou suggerer lui des questions qui sont relatives aux contenu du document .
Note N°2 : Tu dois repondre en francais                     
Note N°1 :Tu es responsable de la qualité de reponse que tu fournis a l'utilisateur . En cas de mauvaise reponse donnée , tu peux etre licencié pour cela . {question} \nContext: {context} \nAnswer: """



prompt =   PromptTemplate(

    input_variables = ['context', 'question'], 

    template = template_content
                      
  )

rag_chain = prompt |llm_groq

response = rag_chain.invoke({"context":docs, "question":question})
print(response)


