
Self RAG

Self-RAG is a strategy for RAG that incorperates self-reflection / self-grading on retrieved documents and generations.

We will borrow some ideas from the paper, shown in green (below):

Retrieve documents
Evaluate them for relevance to the user question
Evaluate LLM generations for faithfulness to the documents (e.g., ensure no hallucinations)
Evaluate LLM generations for usefulness to the question (e.g., does it answer the question)
We will also build on some ideas from the Corrective RAG paper, shown in blue (below):

Use web search if any documents are not relevant or if the answer is not useful.
We implement these ideas from scratch using Mistral and LangGraph:

We use a graph to represent the control flow
The graph state includes information (question, documents, etc) that we want to pass between nodes
Each graph node modifies the state
Each graph edge decides which node to visit next


#### Each node in our graph is a function that:

(1) Take state as an input

(2) Modifies state (e.g., using an LLM)

(3) Outputs the modified state

Each edge decides which node to visit next.



"""
Each node in our graph is a function that:

(1) Take state as an input

(2) Modifies state (e.g., using an LLM)

(3) Outputs the modified state

Each edge decides which node to visit next.

"""