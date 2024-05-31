

# import os
# from crewai import Agent, Task, Crew
# from dotenv import load_dotenv
# from langchain_cohere import ChatCohere
# from digitarTool import DigitarQATool

# from digitarTool import DigitarQuestionTool

# print("------INTEGRATION OF LANGFLOW AND CREWAI ---------")


# load_dotenv()

# # Initialisation des outils
# search_tool_digitar = DigitarQuestionTool()

# # Initialisation du modèle LLM
# llm = ChatCohere()
# question=""
# # Définition de l'agent chercheur
# researcher = Agent(
#   role='Senior Research Analyst',
#   goal='Uncover cutting-edge information about the Digitar website',
#   backstory="""
#   You are known for your role in asking questions about the content of the website using the tool.
#   You are limited to two questions. If you don't get an answer, just say you have a problem with the tool.
#   You have a knack for dissecting complex data and presenting actionable insights.
  
#   After finishing your research, you will give a meaningful summary about your research on 
#   the content of the website.
#   You can ask any question using the tool to get meaningful results before making the report.
#   Note: If you don't get an answer from your first question, don't continue, just cancel the research.
#   The tool take a question as parameter don't forget to fill your question 

#   """,
#   verbose=True,
#   allow_delegation=False,
#   tools=[search_tool_digitar], 
#   llm=llm
# )

# # Création des tâches pour les agents
# task1 = Task(
#   description="""Conduct a comprehensive analysis of the Digitar website""",
#   expected_output="Full analysis report in bullet points of the Digitar website",
#   agent=researcher
# )

# # Instanciation de l'équipe avec un processus séquentiel
# crew = Crew(
#   agents=[researcher],
#   tasks=[task1],
#   verbose=2, # Niveau de journalisation
# )

# # Démarrage du travail de l'équipe
# result = crew.kickoff()

# print("######################")
# print(result)



import streamlit as st
from langflow.load import run_flow_from_json

def main():
    # Titre et description
    st.title("Bienvenue sur Digitar Agence Chatbot")
    st.markdown("Posez-nous des questions sur les services de Digitar et nous vous fournirons des réponses utiles!")

    # Entrée utilisateur
    user_question = st.text_input("Posez une question")

    if st.button("Poser la question"):
        # Exécution du chatbot
        result = run_flow_from_json(flow="C:/devpy/playground/51-langflow-crewai/digitar_website.json",
                                    input_value=user_question)
        
        # Affichage des résultats
        st.success("Voici ce que nous avons trouvé pour vous :")
        st.write(result[0].outputs[0].results)

if __name__ == "__main__":
    main()
