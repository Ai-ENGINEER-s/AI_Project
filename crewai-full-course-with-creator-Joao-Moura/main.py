import warnings 
from dotenv import load_dotenv 
import os
from langchain_groq import ChatGroq 
from langchain_cohere import ChatCohere
from crewai import Agent , Task , Crew
warnings.filterwarnings('ignore')
from IPython.display import Markdown

load_dotenv()
"https://learn.deeplearning.ai/accomplishments/2fd0815e-5af5-4932-ba60-f2f4b1e005fb?usp=sharing
"
api_key = os.getenv('COHERE_API_KEY')

llm = ChatCohere(api_key=api_key)

# so in this project we gonna need three agents for article writing system 

print("----CREATING AGENTS -----------")

planner = Agent(
    role="content planner", 
    goal ="Plan engaging and factually accurate content on the {topic}", 
    backstory ="You're working on planning a blog article "
                "about the topic : {topic} ."
                "You collect information that helps the "
                "audience learn something "
                "and make informed decisions ."
                "Your work is the basis for "
                "the content Writer to write an article on this topic" , 
                allow_delegation = False, 
                verbose = True , 
                llm=llm


)



writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate "
         "opinion piece about the topic: {topic}",
    backstory="You're working on a writing "
              "a new opinion piece about the topic: {topic}. "
              "You base your writing on the work of "
              "the Content Planner, who provides an outline "
              "and relevant context about the topic. "
              "You follow the main objectives and "
              "direction of the outline, "
              "as provide by the Content Planner. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provide by the Content Planner. "
              "You acknowledge in your opinion piece "
              "when your statements are opinions "
              "as opposed to objective statements.",
    allow_delegation=False,
    verbose=True, 
    llm=llm
)


editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with "
         "the writing style of the organization. ",
    backstory="You are an editor who receives a blog post "
              "from the Content Writer. "
              "Your goal is to review the blog post "
              "to ensure that it follows journalistic best practices,"
              "provides balanced viewpoints "
              "when providing opinions or assertions, "
              "and also avoids major controversial topics "
              "or opinions when possible.",
    allow_delegation=False,
    verbose=True, 
    llm=llm
)



print("----CREATING TASKS -----------")

plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, "
            "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
            "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
            "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document "
        "with an outline, audience analysis, "
        "SEO keywords, and resources.",
    agent=planner,
)


write = Task(
    description=(
        "1. Use the content plan to craft a compelling "
            "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
		"3. Sections/Subtitles are properly named "
            "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
            "engaging introduction, insightful body, "
            "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
            "alignment with the brand's voice.\n"
    ),
    expected_output="A well-written blog post "
        "in markdown format, ready for publication, "
        "each section should have 2 or 3 paragraphs.",
    agent=writer,
)


edit = Task(
    description=("Proofread the given blog post for "
                 "grammatical errors and "
                 "alignment with the brand's voice."),
    expected_output="A well-written blog post in markdown format, "
                    "ready for publication, "
                    "each section should have 2 or 3 paragraphs.",
    agent=editor
)


# Creating the Crew¶
# Create your crew of Agents
# Pass the tasks to be performed by those agents.
# Note: For this simple example, the tasks will be performed sequentially (i.e they are dependent on each other), so the order of the task in the list matters.
# verbose=2 allows you to see all the logs of the execution.


crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=2
)


crew_response = crew.kickoff(inputs={"topic":"Artificial Intelligence"})

# print("-------")

# print(crew_response)


Markdown(crew_response)