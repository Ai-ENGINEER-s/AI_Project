from config.settings import openai_llm
from modules.data_models import GradeAnswer , GradeHallucinations, GradeDocuments


structured_llm_grader_answer = openai_llm.with_structured_output(GradeAnswer)
structured_llm_grader_hallucination = openai_llm.with_structured_output(GradeHallucinations)
structured_llm_grader_document  = openai_llm.with_structured_output(GradeDocuments)