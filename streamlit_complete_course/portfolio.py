



import streamlit as st
from streamlit_lottie import st_lottie
import requests
from PIL import Image

# Configuration de la page
st.set_page_config(page_title="Portfolio de BARRY SANOUSSA", page_icon=":computer:", layout="wide")


# Chargement du CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# En-tête avec style HTML et icônes FontAwesome
st.markdown("""
    <div class="main-header">
        <h1>Portfolio de BARRY SANOUSSA</h1>
        <p>Ingénieur Informaticien à l'Université de Mundiapolis</p>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

# Section à propos de moi
st.markdown('<div class="section-header">À propos de moi</div>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])

with col1:
    image = Image.open("C:/Users/BARRY/Desktop/AI-WorkSpace/streamlit_complete_course/11zon_cropped.png")  # Remplacez par le chemin de votre image
    st.image(image, caption="BARRY SANOUSSA", use_column_width=True, width=20, clamp=True)

with col2:
    st.write("""
    Bonjour ! Je suis BARRY SANOUSSA, ingénieur informaticien à l'Université de Mundiapolis. 
    Avec une expertise étendue en intelligence artificielle, apprentissage automatique,
     traitement du langage naturel et vision par ordinateur, 
    je me spécialise également dans la création d'applications RAG et
     le développement avec des agents comme CrewAI et Phidata. 
    Passionné par l'innovation technologique, je m'efforce de développer des solutions 
    qui apportent une valeur ajoutée et répondent à des défis complexes.
    """)


st.markdown("---")

# Section compétences
st.markdown('<div class="section section-header">Compétences</div>', unsafe_allow_html=True)
st.write("""
- **Langages de programmation** : Python, R, Java
- **Bibliothèques et frameworks** : TensorFlow, Keras, PyTorch, Scikit-learn
- **Outils de traitement du langage naturel** : NLTK, SpaCy, GPT-3
- **Outils de vision par ordinateur** : OpenCV, YOLO, CNN
- **Bases de données** : SQL, NoSQL (MongoDB)
- **Outils de déploiement** : Docker, Kubernetes, Streamlit, Flask
- **Création d'applications RAG** : Expertise dans le développement de solutions RAG
- **Agents** : Développement avec CrewAI, Phidata
""")


st.markdown("---")

# Section projets
st.markdown('<div style="max-width: 800px; margin: 0 auto;">', unsafe_allow_html=True)
st.markdown('<div class="section section-header">Projets</div>', unsafe_allow_html=True)
project_data = [
    {
        "title": "1. Détection d'objets en temps réel",
        "description": """Développement d'un modèle de détection d'objets 
        en temps réel utilisant YOLOv3 et OpenCV.
        Le modèle est capable de détecter plusieurs objets avec une grande précision.""",
        "technologies": "Python, OpenCV, YOLOv3",
        "github": "https://github.com/votrecompte/projet1",
        "icon": "🔍"
    },
    {
        "title": "2. Chatbot intelligent",
        "description": """Création d'un chatbot intelligent capable de répondre à 
                          des questions complexes en utilisant GPT-3 
                          et des techniques avancées de traitement du langage naturel.""",
        "technologies": "Python, GPT-3, SpaCy",
        "github": "https://github.com/votrecompte/projet2",
        "icon": "🤖"
    },
    {
        "title": "3. Analyse de sentiment sur les réseaux sociaux",
        "description": "Développement d'un modèle d'analyse de sentiment pour analyser les commentaires sur les réseaux sociaux, en utilisant des techniques d'apprentissage supervisé et non supervisé.",
        "technologies": "Python, NLTK, Scikit-learn",
        "github": "https://github.com/votrecompte/projet3",
        "icon": "📊"
    },
    {
        "title": "4. Application RAG de gestion des stocks",
        "description": "Création d'une application RAG pour la gestion des stocks, permettant un suivi en temps réel des produits et une optimisation des niveaux de stock.",
        "technologies": "Python, RAG, SQL",
        "github": "https://github.com/votrecompte/projet4",
        "icon": "📦"
    },
    {
        "title": "5. Agent de recommandation avec CrewAI",
        "description": "Développement d'un agent de recommandation utilisant CrewAI pour offrir des suggestions personnalisées aux utilisateurs en fonction de leurs préférences et de leur historique.",
        "technologies": "Python, CrewAI",
        "github": "https://github.com/votrecompte/projet5",
        "icon": "🤖"
    }
]

for project in project_data:
    st.markdown(f"<h3>{project['title']} {project['icon']}</h3>", unsafe_allow_html=True)
    st.write(project["description"])
    st.write(f"**Technologies** : {project['technologies']}")
    st.write(f"[Lien vers le projet]({project['github']})")
    st.markdown("---")
# Section Contact
st.markdown('<div class="section section-header">Contact</div>', unsafe_allow_html=True)

contact_info = {
    "LinkedIn": {
        "icon": "fab fa-linkedin",
        "link": "https://www.linkedin.com/in/votrenom"
    },
    "GitHub": {
        "icon": "fab fa-github",
        "link": "https://github.com/Ai-ENGINEER-s"
    },
    "Email": {
        "icon": "fas fa-envelope",
        "link": "mailto:barrysanoussa19@gmail.com"
    }
}

for platform, info in contact_info.items():
    st.write(f"<div class='contact-info'><i class='{info['icon']}'></i><a href='{info['link']}' target='_blank'>{platform}</a></div>", unsafe_allow_html=True)

# Call to action button
st.write("Vous avez un projet en tête ? Discutons-en !")
st.button("Contactez-moi")


st.markdown("<h2 style='text-align: center; margin-top: 50px;'>Contact</h2>", unsafe_allow_html=True)
st.write("Vous pouvez me contacter via les liens suivants :")

