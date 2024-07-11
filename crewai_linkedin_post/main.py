from sample import Crew
from dotenv import load_dotenv
import agentops
import os

from agents import linkedin_scraper_agent, web_researcher_agent, doppelganger_agent
from tasks import scrape_linkedin_task, web_research_task, create_linkedin_post_task

# Load environment variables
load_dotenv()

# Initialize AgentOps
AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")
if not AGENTOPS_API_KEY:
    raise ValueError("AGENTOPS_API_KEY not found in environment variables")

agentops.init(AGENTOPS_API_KEY)

# Wrap agents with AgentOps tracking
@agentops.track_agent(name='linkedin_scraper')
class LinkedInScraperAgent(type(linkedin_scraper_agent)):
    pass

@agentops.track_agent(name='web_researcher')
class WebResearcherAgent(type(web_researcher_agent)):
    pass

@agentops.track_agent(name='doppelganger')
class DoppelgangerAgent(type(doppelganger_agent)):
    pass

tracked_linkedin_scraper = LinkedInScraperAgent(**linkedin_scraper_agent.__dict__)
tracked_web_researcher = WebResearcherAgent(**web_researcher_agent.__dict__)
tracked_doppelganger = DoppelgangerAgent(**doppelganger_agent.__dict__)

@agentops.record_function('create_crew')
def create_crew():
    return Crew(
        agents=[
            tracked_linkedin_scraper,
            tracked_web_researcher,
            tracked_doppelganger
        ],
        tasks=[
            scrape_linkedin_task,
            web_research_task,
            create_linkedin_post_task
        ]
    )

@agentops.record_function('run_crew')
def run_crew(crew):
    return crew.kickoff()

@agentops.record_function('main')
def main():
    try:
        crew = create_crew()
        result = run_crew(crew)
        print("Here is the result: ")
        print(result)
        return result
    except Exception as e:
        agentops.log_error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        agentops.log_error(str(e))
    finally:
        agentops.end_session('Success')