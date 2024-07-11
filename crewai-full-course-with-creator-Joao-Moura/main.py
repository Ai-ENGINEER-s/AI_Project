import warnings 
from dotenv import load_dotenv 
import os
from langchain_groq import ChatGroq 
from langchain_cohere import ChatCohere
from crewai import Agent, Task, Crew
import agentops
warnings.filterwarnings('ignore')
from IPython.display import Markdown

# Load environment variables
load_dotenv()

# Initialize AgentOps
AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")
if not AGENTOPS_API_KEY:
    raise ValueError("AGENTOPS_API_KEY not found in environment variables")

agentops.init(AGENTOPS_API_KEY)

api_key = os.getenv('COHERE_API_KEY')
llm = ChatCohere(api_key=api_key)

@agentops.record_function('create_agents')
def create_agents():
    print("----CREATING AGENTS -----------")
    
    @agentops.track_agent(name='content_planner')
    class ContentPlanner(Agent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    planner = ContentPlanner(
        role="content planner", 
        goal="Plan engaging and factually accurate content on the {topic}", 
        backstory="You're working on planning a blog article "
                  "about the topic : {topic} ."
                  "You collect information that helps the "
                  "audience learn something "
                  "and make informed decisions ."
                  "Your work is the basis for "
                  "the content Writer to write an article on this topic", 
        allow_delegation=False, 
        verbose=True, 
        llm=llm
    )

    @agentops.track_agent(name='content_writer')
    class ContentWriter(Agent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    writer = ContentWriter(
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

    @agentops.track_agent(name='editor')
    class Editor(Agent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    editor = Editor(
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
    
    return planner, writer, editor

@agentops.record_function('create_tasks')
def create_tasks(planner, writer, editor):
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
    
    return plan, write, edit

@agentops.record_function('create_crew')
def create_crew(agents, tasks):
    return Crew(
        agents=agents,
        tasks=tasks,
        verbose=2
    )

@agentops.record_function('main')
def main():
    try:
        planner, writer, editor = create_agents()
        plan, write, edit = create_tasks(planner, writer, editor)
        crew = create_crew([planner, writer, editor], [plan, write, edit])
        
        crew_response = crew.kickoff(inputs={"topic": "Artificial Intelligence"})
        
        return Markdown(crew_response)
    except Exception as e:
        agentops.log_error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        result = main()
        print(result)
    except Exception as e:
        agentops.log_error(str(e))
    finally:
        agentops.end_session('Success')