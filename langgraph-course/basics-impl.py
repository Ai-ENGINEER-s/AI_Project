from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage 
from langgraph.graph  import END, MessageGraph 
from dotenv import load_dotenv 
from langchain_core.tools import tool 
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import StateGraph 
"""
Exemple d'execution d'un graph conversationnel 
"""



load_dotenv()


model = ChatOpenAI(temperature=0)

# Initialisation du modèle de graph  : chat model 


"""
So what did we do here? Let's break it down step by step:

First, we initialize our model and a MessageGraph.
Next, we add a single node to the graph, called "oracle", which simply calls the model with the given input.
We add an edge from this "oracle" node to the special string END. This means that execution will end after current node.
We set "oracle" as the entrypoint to the graph.
We compile the graph, ensuring that no more modifications to it can be made.
Then, when we execute the graph:

LangGraph adds the input message to the internal state, then passes the state to the entrypoint node, "oracle".
The "oracle" node executes, invoking the chat model.
The chat model returns an AIMessage. LangGraph adds this to the state.
Execution progresses to the special END value and outputs the final state.
And as a result, we get a list of two chat messages as output.


 In langgraph, a node can be either a function or a runnable.

Langgraph and AI Agents 

We need to define the agents and the function to invoke tools . 


 ------> The agent is responsible for deciding what actions to take . 

------>  Function to invoke tools : if the agent decide to take an action , this node will then execute that action  . 

-------> We also need to define some edges and some of these edges can be conditional . The reason they are conditional is that based on the output of a node, one of several paths may be taken.
 The path that is taken is not known until that node is run (the LLM decides).


"""

