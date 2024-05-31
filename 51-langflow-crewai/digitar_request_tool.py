from crewai_tools import BaseTool 

from crewai_tools import BaseTool

from langflow.load import run_flow_from_json
from langchain.tools import tool 
TWEAKS = {
  "ChatOutput-KaZYp": {},
  "ChatInput-AF9VM": {},
  "Prompt-gAKsY": {},
  "AstraDBSearch-s4NWf": {},
  "CohereEmbeddings-cgkyM": {},
  "CohereModel-DQpTq": {}
}

user_question= input("Ask any question about digitar be Services")

result = run_flow_from_json(flow="C:/devpy/playground/51-langflow-crewai/digitar_website.json",
                            input_value=user_question,
                            )

print(type(result))

print(result[0].outputs[0].results)




