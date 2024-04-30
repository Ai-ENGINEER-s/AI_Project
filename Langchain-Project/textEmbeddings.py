"""
Text embedding models in langchain provide a standardized interface for various embedding model providers like OpenAi , COHERE , and Hugging Face .
"""

from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings

embeddings = CohereEmbeddings()

embeddings = embeddings.embed_documents(

           ["Hi there ","Oh hello!", "what's your name ?", "My friends call me World","Hello word !"]

)

print("Number of doucments embedded :",len(embeddings))
print("Dimension of each embedding :", len(embeddings[0]))


query = "Il nous avait dit que tu etait la pour l'aider a faire rentrer les animaux dans l'enclos . Nous sommes tous la pour vous aider a vous en sortir de cette situation ."

embed2 = OpenAIEmbeddings()

embeddings58= embed2.embed_documents(query)
print(embed2)