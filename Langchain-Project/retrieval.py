# retrieval plays crucial role in applications that requires user specific data , not included in the models training set . This process known as Retrieval Augmented Generation(RAG). , involves fetching external data and integrating it into the language model's generation process . Langchain provides a comprehensives suite of tools and functionalities to facilitate his process , catering to both simple and complex applications 



# And langchain achieves that through a series of components 


# 1 Documents Loaders 

"""
Document loaders anables the extraction of data from various sources . With 100 loaders available , they support a range of doccument , apps and sources (private s3 buckets, public websites , databases )

"""

# Let's check some of them here 

# 1 Text file loader 


from langchain_community.document_loaders import TextLoader 


loader = TextLoader("Langchain-Project/textFile.txt")
documents = loader.load()

#2 csv file loader 

from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='./example_data/mlb_teams_2012.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['MLB Team', 'Payroll in millions', 'Wins']
})
documents = loader.load()

# 3 PyPDF loader 

from langchain_community.document_loaders import PyPDFLoader 

laoder = PyPDFLoader('filepath of your pdf file ')

laoder = loader.load()


# Document Transformers 

"""
Document transformers in langchain are essential tools designed to manipulate documents . 

They are used for tasks such as splitting long documents into smaller chunks 
combining and filtering wich are crucial for adapting documents to model's 
context window or meeting specific application . 
The main idea of documents transformers first of all split the documents 
into smaller chunks or filters that useful sometime for adapting document to the LLM context windows . 





"""

# Let's check and see documents transformers part 

#  1 textsplitter 

from langchain.text_splitter import RecursiveCharacterTextSplitter 

state_of_the_union= "Your long text here . So let's we use documents transformers to split text ,filter them  allowing the documents to adapt llm contex windows parameters "

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap= 20, length_function= len, add_start_index=True )

texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])
print(texts[1])