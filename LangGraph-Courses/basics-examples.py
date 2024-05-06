"""
let's implemnent basic example using langgraph 

"""

"---------- Import essentials package: ---------" 

from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage 
from langgraph.graph import END  , MessageGraph
from langchain_core.tools import tool 
from langgraph.prebuilt import ToolNode 

model = ChatOpenAI()
graph = MessageGraph()

graph.add_node("oracle",model)
graph.add_edge("oracle", END )
graph.set_entry_point("oracle")

runnable  = graph.compile()

def oracle_message(messages:list):
    "---------oracle messages"
    return model.invoke(messages)


@tool 
def multiply(firstNumber:int, secondeNumber:int):
    """-----Multiplies two numbers together ."""
    return firstNumber * secondeNumber 



model_with_tools = model.bind_tools([multiply])

graph = MessageGraph()
graph.add_node("oracle", model_with_tools)
tool_node = ToolNode([multiply])

graph.add_node("multiply", tool_node)
graph.add_edeg("multiply", END)
graph.set_entry_point("oracle")
