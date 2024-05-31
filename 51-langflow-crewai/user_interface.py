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
