import os
import json
from typing import Type, Any
from pydantic.v1 import BaseModel, Field
from crewai_tools.tools.base_tool import BaseTool
from langflow.load import run_flow_from_json

class DigitarQuestionToolSchema(BaseModel):
    """Input for DigitarQuestionTool."""
    question: str = Field(..., description="Question you want to ask about Digitar be Services")

class DigitarQuestionTool(BaseTool):
    name: str = "Digitar Question Tool"
    description: str = "A tool that answers questions about Digitar be Services using a predefined flow."
    args_schema: Type[BaseModel] = DigitarQuestionToolSchema
    flow_path: str = "C:/devpy/playground/51-langflow-crewai/digitar_website.json"

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        question = kwargs.get('question')
        if not question:
            raise ValueError("A question must be provided")

        try:
            print(f"Running flow with question: {question}")
            result = run_flow_from_json(flow=self.flow_path, input_value=question)
            print(f"Flow result: {result}")
            if result and len(result) > 0 and hasattr(result[0], 'outputs') and len(result[0].outputs) > 0 and hasattr(result[0].outputs[0], 'results'):
                return result[0].outputs[0].results
            else:
                return "No results found"
        except Exception as e:
            return f"An error occurred: {str(e)}"

# Example usage
# if __name__ == "__main__":
#     user_question = input("Ask any question about Digitar be Services: ")
#     tool = DigitarQuestionTool()
#     response = tool._run(question=user_question)
#     print(response)
