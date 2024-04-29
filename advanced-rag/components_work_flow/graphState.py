
from typing_extensions import TypedDict
from typing import List

class GraphState(TypedDict):
    """
     Represents the state of our graph.GraphState

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents

    """
    question:str 
    generation:str 
    web_search:str 
    documents:List[str]
