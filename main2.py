#Importation de la bibliotheque pour creer l'interface de l'application web
import streamlit as st

# Importing the PyPDF2 library to read the PDF files  and extract the text from the PDF files
from PyPDF2 import PdfReader

#Importation de la class CharacterTextSplit depuis la bibliotheque  pour diviser le text
from langchain.text_splitter import CharacterTextSplitter

# Importing the OpenAIEmbeddings class from the langchain library to create the vector store(magasin de vecteur)
#from langchain.embeddings import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings

# Importing the FAISS class from the langchain library to create the vector store
from langchain.vectorstores import FAISS

# Importing the ChatOpenAI class from the langchain library to create the language model
#from langchain.chat_models import ChatOpenAI
from langchain_cohere import ChatCohere

# Importing the ChatPromptTemplate class from the langchain library to create the prompt
from langchain_core.prompts import ChatPromptTemplate

#importation des functions

# Importing the create_stuff_documents_chain and create_retrieval_chain functions from the langchain library
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)

# Importing the create_retrieval_chain function
from langchain.chains import create_retrieval_chain

#from dotenv import load_dotenv

#load_dotenv()


def main():
    st.set_page_config(layout="wide")
    st.subheader(
        "Retrieval Augmented Generation (RAG) Pedagogical Chatbot", divider="rainbow"
    )
    with st.sidebar:
        # Title of the sidebar
        st.sidebar.title("Data Loader")
        #st.image("rag.png", width=500)
        # File uploader to upload the PDF files
        pdf_docs = st.file_uploader(
            label="Upload Your PDFs",
            accept_multiple_files=True,
        )
        # Submit button to start the process of extracting the content of the PDF files
        if st.button("Submit"):
            # Loading spinner to show the process is running
            with st.spinner("Loading..."):
                # Extract the content of the PDF
                pdf_content = ""
                # Loop through the PDF files
                for pdf in pdf_docs:
                    # Read the PDF file
                    pdf_reader = PdfReader(pdf)
                    # Loop through the pages of the PDF file
                    for page in pdf_reader.pages:
                        # Extract the text from the PDF page and add it to the pdf_content variable
                        pdf_content += page.extract_text()
                # st.write(pdf_content)
                # Get chunks of the content
                # Split the text into chunks of 1000 characters with an overlap of 200 characters
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                # Split the text into chunks of 1000 characters with an overlap of 200 characters
                chunks = text_splitter.split_text(pdf_content)
                # Display the chunks of the text
                st.write(chunks)

                # Create Vector store
          
                      # OpenAI API key
                
              
                # Create the OpenAIEmbeddings object
                
                embeddings = SentenceTransformerEmbeddings()
                
                # Create the FAISS vector store from the text chunks and the OpenAIEmbeddings object
                openai_vector_store = FAISS.from_texts(
                    texts=chunks, embedding=embeddings
                )
                # Create the language model (LLM) with the OpenAI API key
                llm = ChatCohere(cohere_api_key="sYubcP96NJwKyupvUIyxjuSLZJFRLXbRS3crkN1w")
                
                # Create the conversational retrieval chain with the language model, the memory, and the vector store
                prompt = ChatPromptTemplate.from_template(
                    """
                    Answer the following question based only on the provided context:
                    <context>
                      {context}
                    </context>
                    Question: {input}
                    """
                )
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = openai_vector_store.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                st.session_state.retrieve_chain = retrieval_chain

    # Title of the web app
    st.subheader("Chatbot zone")
    # Sidebar of the web app
    user_question = st.text_input("Ask your question :")
    if user_question:
        response = st.session_state.retrieve_chain.invoke({"input": user_question})
        st.markdown(response["answer"], unsafe_allow_html=True)


if __name__ == "__main__":
    main()