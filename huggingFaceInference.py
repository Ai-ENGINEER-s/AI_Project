
from langchain_community.llms import huggingface_text_gen_inference
from langchain_experimental.chat_models   import Llama2Chat
from langchain_fireworks import ChatFireworks
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_fireworks import ChatFireworks



LLAM2="LL-f38Az93OpTpw2OhAmfiJvOnpbMs2uj9jBZ1cHywLuPilfkgEoqI3caPiJtg5p1VU"




chat = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct",api_key="NFoloKMdUvtilwtKOAOTst8R82gECwGMHpViN5aH3YL6GHQ9")


# ChatFireworks Wrapper
system_message = SystemMessage(content="You are to chat with the user.")
human_message = HumanMessage(content="Who are you?")

response = chat.invoke([system_message, human_message])

print(response.content)
