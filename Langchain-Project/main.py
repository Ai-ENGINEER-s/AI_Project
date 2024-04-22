import langchain_helper as lch 
import streamlit as st 
from youtubeAssistant import youtube_assistant as yt


# st.title("Pets name generator")
# max_char= 17
# user_animal_type = st.sidebar.selectbox("what is your pet ?", ("Cat", "Dog","Cow","Horse"))


# if user_animal_type == "Cat":
#      pet_color = st.sidebar.text_input(label="what is your Cat color ? ", max_chars=max_char)


# if user_animal_type == "Dog":
#     pet_color = st.sidebar.text_input(label="what is your Dog color ? ", max_chars=max_char)


# if user_animal_type == "Horse":
#    pet_color = st.sidebar.text_input(label="what is your Horse color ? ", max_chars=max_char)


# if user_animal_type== "Cow":
#     pet_color = st.sidebar.text_input(label="what is your Cow color ? ", max_chars=max_char)


# if pet_color:
#     response=lch.generate_pet_name(user_animal_type,pet_color)
#     st.text(response)
# st.sidebar.button(label="Launch",type='primary',use_container_width=True)


import streamlit as st
import langchain_helper as lch
import textwrap

from dotenv import load_dotenv 


dotenv_dir = r"C:\Users\BARRY\Desktop\AI-WorkSpace\Langchain-Project\.env"
load_dotenv(dotenv_dir)

st.title("YouTube Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_input(
            label="What is the YouTube video URL?",
            max_chars=50
            )
        query = st.sidebar.text_input(
            label="Ask me about the video?",
            max_chars=50,
            key="query"
            )
       
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        "[View the source code](https://github.com/rishabkumar7/pets-name-langchain/tree/main)"
        submit_button = st.form_submit_button(label='Submit')

if query and youtube_url:
        db = yt.create_db_from_youtube_video_url(youtube_url)
        response, docs = yt.get_response_from_query(db, query)
        st.subheader("Answer:")
        st.text(textwrap.fill(response, width=85))