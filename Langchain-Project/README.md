# Langchain Crash Course 

Chat models are a core component of Langchain . 

What is a chat Model ? 
Chat model is language model that takes as input chat message and return as output chat message . 

There are many providers of chat model like OpenAi , Cohere  , Hugging Face , etc 

# What is Langchain 
Langchain is opensource framework for building LLMs Application 

# LCEL : LangChain Expression Language 
LCEL is declarative way to easily compose chains together from simplest "prompt +LLM" chain to the most complex chains. 

# Why use LCEL 

=> First-class streaming support , this means you get  you get the best possible time-to-first-token

=> Async support Any chain built with LCEL can be called both with the synchronous API as well as with the asynchronous API.
=> Optimized parallel execution , trigger to async or sync mode to the right time . 
and so on .

**LCEL is the foundation of many of LangChain's components, and is a declarative way to compose chains. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains.
Overview: LCEL and its benefits **

# High architectural overview of Langchain 

1. Components : Components are LLMs Wrappers , Prompt Template , indexes informations 

2. Chains  : Assemble components to solve a specific task . Chains refer to sequence of calls-wheter to an LLM , a tool , or data preprocessing step

3. Agents : Agents allows LLMs to interact with its environment or external resources like APIs . The core idea of Agents is to use an LLM to choose a sequence of action to take . In agents 
an LLM is reasoning engine that is used to determine wich action to take and in wich order 

4. Tools are functions that an agent calls .There are two important considerations here : 
    1. giving the agent the right tools 
    2. Describing the tools in a way that
       is most helpful to the agent .

**Prompt Template** are input for LLMs , it's allows you to avoid hard coded text .

**Indexes**: retrieve or extract  relatives informations for the LLMs .
**Chains**  : allows to combine multiple components together to build an entire application  

**Agents** : Agents allows LLMs to interact with its environment or external resources like APIs 

LLMs Parameters undersdanting 

**Temperature**  : temperature in the LLM means how creative you want your LLM to be . Temperature varies from 0 to 1 
so 0 means that your LLM gonna be no straighforward not creative , you can make it creative by adjascing the temperature rate 

Learning Resources : 




**chunk_overlap** :  that's mean when document get splitted in chunks by example let's suppose that our chunk_overlap = 100 , this means that the firt document will have his last hundred words included int the document two first hundred words 

Text File Loader: Load a simple .txt file into a document.

from langchain.document_loaders import TextLoader

loader = TextLoader("./sample.txt")
document = loader.load()
CSV Loader: Load a CSV file into a document.

from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='./example_data/sample.csv')
documents = loader.load()
We can choose to customize the parsing by specifying field names -

loader = CSVLoader(file_path='./example_data/mlb_teams_2012.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['MLB Team', 'Payroll in millions', 'Wins']
})
documents = loader.load()
PDF Loaders: PDF Loaders in LangChain offer various methods for parsing and extracting content from PDF files. Each loader caters to different requirements and uses different underlying libraries. Below are detailed examples for each loader.

PyPDFLoader is used for basic PDF parsing.

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
pages = loader.load_and_split()
MathPixLoader is ideal for extracting mathematical content and diagrams.

from langchain.document_loaders import MathpixPDFLoader

loader = MathpixPDFLoader("example_data/math-content.pdf")
data = loader.load()
PyMuPDFLoader is fast and includes detailed metadata extraction.

from langchain.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("example_data/layout-parser-paper.pdf")
data = loader.load()

# Optionally pass additional arguments for PyMuPDF's get_text() call
data = loader.load(option="text")
PDFMiner Loader is used for more granular control over text extraction.

from langchain.document_loaders import PDFMinerLoader

loader = PDFMinerLoader("example_data/layout-parser-paper.pdf")
data = loader.load()
AmazonTextractPDFParser utilizes AWS Textract for OCR and other advanced PDF parsing features.

from langchain.document_loaders import AmazonTextractPDFLoader

# Requires AWS account and configuration
loader = AmazonTextractPDFLoader("example_data/complex-layout.pdf")
documents = loader.load()
PDFMinerPDFasHTMLLoader generates HTML from PDF for semantic parsing.

from langchain.document_loaders import PDFMinerPDFasHTMLLoader

loader = PDFMinerPDFasHTMLLoader("example_data/layout-parser-paper.pdf")
data = loader.load()
PDFPlumberLoader provides detailed metadata and supports one document per page.

from langchain.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("example_data/layout-parser-paper.pdf")
data = loader.load()