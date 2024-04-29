from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel 
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_cohere import ChatCohere 
from langchain_openai import OpenAI 
from langchain_cohere import CohereRagRetriever
doten_dir = r"C:\Users\BARRY\Desktop\AI-WorkSpace\Langchain-Basics\.env"
load_dotenv(doten_dir)
llm = ChatCohere()

"""
output parsers play a crucial role in langchain , enabling users to structure the responses generated by language models . 
n this section, we will explore the concept of output parsers and provide code examples using Langchain's PydanticOutputParser, SimpleJsonOutputParser, CommaSeparatedListOutputParser, DatetimeOutputParser, and XMLOutputParser.


So the global idea of outputparsers is to allow users structure the responses generated by language modes 


Langchain provides the PydanticOutputParser for parsing responses into Pydantic data structures


"""

from typing import List 
from langchain_openai import OpenAI 
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel , Field , validator 

from langchain_cohere import ChatCohere


model = ChatCohere()

# Let's define our desired data structured using Pydantic 

class  Joke(BaseModel):
    setup:str = Field(description="question to set up a Joke ")
    punchline:str=Field(description="answer to resolve the joke")
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question !")
        return field 
 
parser = PydanticOutputParser(pydantic_object=Joke)

# create a prompt wit format instructions 

prompt = PromptTemplate(
    template="Answer the user query .\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)

# Define a query to prompt the language model 

query = "Tell me a joke "

chain = prompt | model 
output = chain.invoke({"query":query})
# Parse the output using the parser 

parsed_result = parser.invoke(output)
print(parsed_result)
