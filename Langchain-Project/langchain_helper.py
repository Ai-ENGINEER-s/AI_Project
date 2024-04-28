
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from langchain_groq import ChatGroq
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

dotenv_dir =r"C:\Users\BARRY\Desktop\AI-WorkSpace\Langchain-Project\.env"
load_dotenv(dotenv_dir)



def generate_pet_name(animal_type:input, pet_color : input):
    chat = ChatGroq(temperature=0,model_name ="llama3-8b-8192")
    system ="You are assistant known for helping people find their animals names . You must finde fives beautiful names of the given pet name . Most importants parameters "
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain_name = prompt | chat
    return  chain_name.invoke({"text": animal_type, "text":pet_color}).content 


def langchainAgent():
    llm = ChatGroq(temperature=0.8,name="llama3-8b-8192")
    tools = load_tools(["wikipedia","llm-math"],llm)

    agent = initialize_agent(
        tools, llm,agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose =True
    )
   
    result = agent.invoke("what is the average age of a dog ? . Multiply the age by 7 ")
    print(result)

def myPersonalAgent():
    askYourQuestion =input("ask your question")
    llm = ChatGroq(temperature= 0.5,name="llama3-8b-8192")

    tools =load_tools(tool_names=['wikipedia'],llm=llm)
    customAgent = initialize_agent(tools=tools,llm=llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,verbose =True)
    result = customAgent.run(askYourQuestion)
    print(result)

if __name__=="__main__":
    #print(generate_pet_name("Horse", "black"))
    myPersonalAgent()

