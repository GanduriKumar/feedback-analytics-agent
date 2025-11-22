from src.tools.custom_llm import CustomLLMModel
from pydantic import BaseModel
import pandas as pd, csv
import re
from functools import lru_cache

class Themes(BaseModel):
    """
    Represents a single theme classification produced for a piece of feedback or issue.

    Attributes
    ----------
    sentiment : str
        Sentiment label for the text, e.g. "positive", "neutral", or "negative".
    theme : str
        High-level theme or category assigned to the feedback, e.g. "usability", "performance".
    classification : str
        More specific label or classifier output for the theme, e.g. "bug", "feature_request".
    issue_description : str
        The original or normalized text describing the issue or feedback.

    Notes
    -----
    - This class is a Pydantic BaseModel and therefore provides data validation and easy
      serialization via .dict() and .json().
    - Consider adding field validators if stricter constraints are required (e.g., non-empty
      strings, allowed sentiment values).
    - Instances are intended to be lightweight carriers of classification results and can be
      used directly in APIs, logs, or persisted as JSON.

    Example
    -------
    >>> Themes(
            product ="Pixel9"
    ...     sentiment="negative",
    ...     theme="performance",
    ...     classification="slow_response",
    ...     issue_description="The app takes too long to load the dashboard."
    ... )
    """
    product :str
    sentiment:str
    theme:str
    classification: str
    issue_description: str

class ThemeClassifier:
    """
    ThemeClassifier
    A helper class that wraps a CustomLLMModel chat instance to extract structured
    theme information from free-text reviews. The class prepares a prompt instructing
    the underlying LLM to produce JSON that conforms to a predefined Themes schema,
    sends the prompt and review to the chat model, and validates/parses the model's
    response into a Python dictionary.
    Attributes
    ----------
    chat
        Instance returned by CustomLLMModel().getchatinstance() used to invoke the LLM.
    EXTRACT_PROMPT : str
        Prompt template used to instruct the model to extract sentiment, theme,
        classification, and issue_description and return the result as JSON.
    Methods
    -------
    extract_themes(review: str) -> dict
        Send the review text to the chat model using the EXTRACT_PROMPT and parse the
        model response into a dictionary with the following keys:
          - product (str): e,g: "Pixel8", "iPhone10", etc.
          - sentiment (str): e.g., "positive", "negative", "neutral".
          - theme (str): high-level category such as "customer service", "battery", etc.
          - classification (str): intent or message type such as "complaint", "praise".
          - issue_description (str): short, human-readable description of the issue.
        Parameters
        ----------
        review : str
            The review text to analyze.
        Returns
        -------
        dict
            A dictionary containing the parsed fields listed above.
        Raises
        ------
        Any exceptions raised by the chat.invoke call or by Themes.model_validate_json
        (for example, network errors, invalid model output, or schema validation errors)
        are propagated to the caller.
    Notes
    -----
    - The method relies on Themes.model_json_schema() when invoking the chat and
      Themes.model_validate_json(...) to validate and convert the model response.
      Ensure the Themes model and CustomLLMModel are correctly implemented and
      available in the runtime environment.
    - The prompt expects the model to return exactly the JSON schema described;
      callers should handle unexpected model outputs or consider additional retry/
      sanitization logic for production use.
    """
    def __init__(self):
        self.chat = CustomLLMModel().getchatinstance()
        # Cache the prompt to avoid string concatenation overhead
        self._prompt_template = (
            "Extract the following information from the given review text. "
            "Provide the output in JSON format exactly matching this schema with fields: "
            "sentiment, theme, classification, and issue_description.\n\n"
            "Review: {review}\n\n"
            "Output example:\n"
            '{{\n'
            '  "product": "Pixel 9",\n'
            '  "sentiment": "positive",\n'
            '  "theme": "customer service",\n'
            '  "classification": "complaint",\n'
            '  "issue_description": "Battery life."\n'
            '}}\n\n'
            "Now provide the JSON output for the above review."
        )
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _sanitize_review(review: str) -> str:
        """Sanitize review with caching for repeated patterns"""
        # Remove potential prompt injection patterns
        review = re.sub(r'[^\w\s\.\,\!\?\-\']', '', review)
        # Limit length to prevent token overflow
        max_length = 5000
        if len(review) > max_length:
            review = review[:max_length]
        return review.strip()
    
    def extract_themes(self, review: str) -> dict:
        """Extract themes with input validation and caching"""
        if not review or not isinstance(review, str):
            return {
                "product": None,
                "sentiment": None,
                "theme": None,
                "classification": None,
                "issue_description": None,
            }
        
        # Use cached sanitization
        sanitized_review = self._sanitize_review(review)
        
        # Use formatted template instead of f-string concatenation
        prompt = self._prompt_template.format(review=sanitized_review)
        
        try:
            response = self.chat.invoke(
                input=prompt,
                format=Themes.model_json_schema()
            )
            result = Themes.model_validate_json(response.content)
            
            return {
                "product": result.product,
                "sentiment": result.sentiment,
                "theme": result.theme,
                "classification": result.classification,
                "issue_description": result.issue_description,
            }
        except Exception as e:
            print(f"Error extracting themes: {e}")
            return {
                "product": None,
                "sentiment": None,
                "theme": None,
                "classification": None,
                "issue_description": None,
            }
