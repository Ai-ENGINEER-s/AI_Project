from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
import os
from crewai import Agent, Task, Crew, Process
from langchain_cohere import ChatCohere 
from langchain_groq import ChatGroq
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFacePipeline
load_dotenv()


os.getenv('HUGGINGFACE_API_KEY')
os.getenv('SERPER_API_KEY') 
# serper.dev API key


llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool(website_url="https://digitar.be/")

# Define your agents with roles and goals
chercheur = Agent(
  role='Analyste de Recherche Senior',
  goal='Découvrir des informations détaillées et précises sur l’entreprise',
  backstory="""Vous travaillez dans un think tank technologique de premier plan.
  Votre expertise réside dans l'identification des tendances émergentes et la fourniture d'analyses complètes.
  Vous avez un talent pour disséquer des données complexes et présenter des informations exploitables.""",
  verbose=True,
  allow_delegation=False,
  llm=llm,
  tools=[scrape_tool]
)
tache1 = Task(
  description="""Effectuer une analyse approfondie de l’entreprise.
  Identifier les aspects clés tels que le rôle de l'entreprise, sa taille, ses ressources (matérielles et humaines) et sa structure organisationnelle.
  Mettre également en avant les principaux produits ou services offerts par l’entreprise.""",
  expected_output="Rapport d'analyse détaillé sous forme de points",
  agent=chercheur, 
  llm=llm
)

redacteur = Agent(
  role='Stratège de Contenu Technologique',
  goal='Rédiger une présentation captivante et engageante de l’entreprise',
  backstory="""Vous êtes un stratège de contenu renommé, connu pour vos articles perspicaces et engageants.
  Vous transformez des concepts complexes en récits captivants.""",
  verbose=True,
  allow_delegation=True
)
tache2 = Task(
  description="""En utilisant l'analyse détaillée fournie, développer une présentation succincte et engageante de l’entreprise (2 à 3 pages max).
  Décrire le rôle, la taille, les ressources, l'organisation et les principaux produits ou services de l’entreprise avec vos propres mots.
  Évitez de simplement copier le contenu du site web ou des supports marketing de l’entreprise.""",
  expected_output="Présentation complète de l'entreprise d'au moins 4 paragraphes",
  agent=redacteur
)


# Instantiate your crew with a sequential process
crew = Crew(
  agents=[chercheur,redacteur],
  tasks=[tache1, tache2],
  verbose=2, # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)