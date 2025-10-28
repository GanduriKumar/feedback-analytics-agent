from src.utilities.custom_llm import CustomLLMModel
from pydantic import BaseModel

class Themes(BaseModel):
    sentiment:str
    theme:str
    classification: str
    issue_description: str

class ThemeClassifier:
    def __init__(self):
        self.chat = CustomLLMModel().getchatinstance()
        self.EXTRACT_PROMPT = ("Extract the following information from the given review text." 
                               "Provide the output in JSON format exactly matching this schema with fields: sentiment, "
                               "theme, classification, and issue_description."
                               "Review: f'{review}'"
                               "Output example:"
                               ' {'
                               '  "sentiment": "positive",'
                               '  "theme": "customer service",'
                               '  "classification": "complaint",'
                               '  "issue_description": "Battery life."'
                               '}'
                               "Now provide the JSON output for the above review."
                               )

    def extract_themes(self,review:str):
        response = self.chat.invoke(
            input=f"{self.EXTRACT_PROMPT}:{review}",
            format = Themes.model_json_schema()
        )
        result = Themes.model_validate_json(response.content)
        theme = {
            "sentiment":result.sentiment,
            "theme":result.theme,
            "classification":result.classification,
            "issue_description":result.issue_description,

        }
        # print(theme)
        return theme
