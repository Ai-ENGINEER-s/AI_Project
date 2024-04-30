"""
Vecstores in langchain support efficient storage searching of text embeddings . Langchain integrates over 50 vector stores , providing a standardized interface for ease of use . 
After embeddings texts we can store them in a vector store like chroma and perform similarity searches 
"""

from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import faiss

pdfpages =""
airtablestore=""
pdfstore = faiss.FAISS.from_documents(pdfpages, embedding=OpenAIEmbeddings())

airtablestore = faiss.FAISS.from_documents(airtablestore, embedding=OpenAIEmbeddings())