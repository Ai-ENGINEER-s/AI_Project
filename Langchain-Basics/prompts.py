"""

Prompts are essential in guiding language models to generate relevant and
 coherent outputs. They can range from simple
 instructions to complex few-shot examples. In LangChain, 
handling prompts can be a very streamlined process,
 thanks to several dedicated classes and functions.

LangChain's PromptTemplate class is a versatile tool for creating string prompts. 
It uses Python's str.format syntax, allowing for dynamic prompt generation.
 You can define a template with placeholders and fill them with specific values as needed.



 Custom prompt templates are sometimes essential for tasks requiring unique formatting or specific instructions. Creating a custom prompt template involves defining input variables and a custom formatting method. This flexibility allows LangChain to cater to a wide array of application-specific requirementsCustom prompt templates are sometimes essential for tasks requiring unique formatting or specific instructions. Creating a custom prompt template involves defining input variables and a custom formatting method. This flexibility allows LangChain to cater to a wide array of application-specific requirements

 
"""

from dotenv import load_dotenv 
from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
dotenv_dir = r"Langchain-Basics/.env"
load_dotenv(dotenv_dir)

llm = ChatCohere()


# example_promt = prompt = PromptTemplate(
#     template="Answer the user query.\n{format_instructions}\n{query}\n",
#     input_variables=["query"],
# )

# joke_chain = llm |example_promt
# resp = joke_chain.invoke({"question"})


from langchain_core.prompts import PromptTemplate 


prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)
print_prompt = prompt_template.format(adjective="Funny",content="chickens")
print(print_prompt)

# a list of message prompt 
chain = llm | prompt_template

# res =llm.invoke("Tell me about Ibrahim Traore")
# print(res)
from langchain.schema.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage(content="You are Micheal Jordan."),
    HumanMessage(content="Which shoe manufacturer are you associated with?"),
]
response = llm.invoke(messages)
print(response.content)

from langchain_core.prompts import PromptTemplate 



prompt_template = PromptTemplate(
    template="""You are medical assistant and you are responsible for giving dignostic report to sick people .
      So they give you the disease {name} and you telle them what is the medecines for that """, 

    input_variables=["{name}"],
     
    input_types={"Assistant":"patient"}
)
chain = llm | prompt_template 
res = chain.invoke("my disease is cancer")
print(res)

from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)
formatted_messages = chat_template.format_messages(name="Bob", user_input="What is your name?")
for message in formatted_messages:
    print(message)


