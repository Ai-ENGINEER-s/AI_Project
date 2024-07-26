import streamlit as st
from crewai import Crew, Task, Agent
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Chargement de la clé API OpenAI
api_key = os.getenv('OPENAI_API_KEY')
print(api_key)

# Vérifier si l'API key est présente
if not api_key:
    st.error("La clé API OpenAI n'a pas été trouvée. Veuillez vérifier le fichier .env.")
    st.stop()

# Initialiser le modèle de langage
try:
    llm = ChatOpenAI(api_key=api_key)
except Exception as e:
    st.error(f"Erreur lors de l'initialisation de l'API OpenAI : {e}")
    st.stop()

# Définition des agents pour DataDecider
def create_agent(role, goal, backstory):
    try:
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            allow_delegation=False,
            verbose=True,
            llm=llm
        )
    except Exception as e:
        st.error(f"Erreur lors de la création de l'agent {role} : {e}")
        st.stop()

agent_question = create_agent(
    role="Spécialiste en formulation de questions",
    goal="Formuler des questions pertinentes pour interroger la base de données.",
    backstory="Vous êtes compétent pour comprendre les questions des utilisateurs et formuler des requêtes appropriées à poser à la base de données."
)

agent_reponse = create_agent(
    role="Spécialiste en analyse de réponses",
    goal="Analyser les réponses de la base de données et extraire les informations pertinentes.",
    backstory="Vous avez l'expertise nécessaire pour analyser les réponses de la base de données et extraire les informations pertinentes pour répondre à la question de l'utilisateur."
)

agent_rapport = create_agent(
    role="Spécialiste en génération de rapports",
    goal="Générer un rapport détaillé basé sur les informations extraites de la base de données.",
    backstory="Vous êtes capable de compiler les informations extraites de la base de données dans un rapport structuré et facile à comprendre."
)

# Définition des tâches pour les agents
def create_task(description, expected_output, agent):
    try:
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
        )
    except Exception as e:
        st.error(f"Erreur lors de la création de la tâche pour l'agent {agent.role} : {e}")
        st.stop()

tache_formuler_question = create_task(
    description=(
        "1. Recevoir la question de l'utilisateur.\n"
        "2. Formuler des questions appropriées pour interroger la base de données.\n"
        "3. Poser les questions à la base de données et recevoir les réponses."
    ),
    expected_output="Réponses de la base de données aux questions formulées.",
    agent=agent_question,
)

tache_analyser_reponses = create_task(
    description=(
        "1. Analyser les réponses de la base de données pour extraire les informations pertinentes.\n"
        "2. Identifier les tendances et les insights à partir des réponses."
    ),
    expected_output="Informations pertinentes extraites des réponses de la base de données.",
    agent=agent_reponse,
)

tache_generer_rapport = create_task(
    description=(
        "1. Compiler les informations extraites dans un rapport structuré.\n"
        "2. Inclure des visualisations telles que des graphiques et des tableaux pour illustrer les informations."
    ),
    expected_output="Rapport complet avec visualisations en format PDF.",
    agent=agent_rapport,
)

# Créer le Crew
try:
    crew = Crew(
        agents=[agent_question, agent_reponse, agent_rapport],
        tasks=[tache_formuler_question, tache_analyser_reponses, tache_generer_rapport],
        verbose=2
    )
except Exception as e:
    st.error(f"Erreur lors de la création du Crew : {e}")
    st.stop()

# Création de l'interface utilisateur avec Streamlit
st.title("DataDecider - DIGITAR")
question_utilisateur = st.text_input("Posez votre question :")

if st.button("Obtenir le rapport"):
    if not question_utilisateur:
        st.error("Veuillez entrer une question.")
    else:
        try:
            crew_response = crew.kickoff(inputs={"question": question_utilisateur})
            st.markdown(crew_response)
        except Exception as e:
            st.error(f"Erreur lors de l'exécution du Crew : {e}")
