from src.tools.custom_llm import CustomLLMModel
from pydantic import BaseModel
import pandas as pd, csv

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
        self.EXTRACT_PROMPT = ("Extract the following information from the given review text." 
                               "Provide the output in JSON format exactly matching this schema with fields: sentiment, "
                               "theme, classification, and issue_description."
                               "Review: f'{review}'"
                               "Output example:"
                               ' {'
                               '  "product": "Pixel 9",'
                               '  "sentiment": "positive",'
                               '  "theme": "customer service",'
                               '  "classification": "complaint",'
                               '  "issue_description": "Battery life."'
                               '}'
                               "Now provide the JSON output for the above review."
                               )

    def extract_themes(self,review:str):
        """
        Tool: extract_themes
        Short description:
            Call the configured chat model to extract themes, sentiment and issue classification
            from a single review string, validate the model output against the Themes schema,
            and return a normalized dictionary suitable for downstream processing.
        Usage (agent-friendly):
            - name: extract_themes
              description: Extract theme, sentiment, classification and an issue description from a review.
              input: {"review": "<string>"}
              output: {
                  "product": "<string|null>",
                  "sentiment": "<string|null>",
                  "theme": "<string|null>",
                  "classification": "<string|null>",
                  "issue_description": "<string|null>"
        Args:
            review (str): The text of the review to analyze. This string is appended to self.EXTRACT_PROMPT
                          and sent to self.chat.invoke for analysis.
        Requirements / assumptions:
            - self.EXTRACT_PROMPT (str) must be defined and contain the prompt template to guide the model.
            - self.chat.invoke(input: str, format: Any) must be available. It should return an object with a
              .content attribute containing the model response (expected to be JSON conforming to Themes schema).
            - Themes.model_json_schema() returns the expected JSON schema to pass as the `format` parameter.
            - Themes.model_validate_json(json_str: str) validates/parses the JSON and returns an object with attributes:
                - product
                - sentiment
                - theme
                - classification
                - issue_description
        Returns:
            dict: A dictionary with the following keys:
                - "product": product name extracted from review text (str or None)
                - "sentiment": model-derived sentiment (str or None)
                - "theme": identified theme (str or None)
                - "classification": issue classification label (str or None)
                - "issue_description": extracted or synthesized issue description (str or None)
        Errors / exceptions:
            - Propagates exceptions raised by self.chat.invoke (e.g., network errors, API errors).
            - Propagates validation/parsing errors from Themes.model_validate_json (e.g., JSONDecodeError,
              ValueError) if the model output does not match the expected schema.
            - Caller should handle or surface these exceptions as appropriate.
        Side effects:
            - Makes an external call via self.chat.invoke (may be network-bound, rate-limited, or billable).
            - Embeds the review into the prompt; sensitive data in `review` will be transmitted to the model.
        Example:
            >>> tool_output = self.extract_themes("The app crashes when I try to upload a file.")
            >>> # tool_output -> {
            >>> #   "product": "Pixel 8",
            >>> #   "sentiment": "negative",
            >>> #   "theme": "stability",
            >>> #   "classification": "bug",
            >>> #   "issue_description": "App crash "
            >>> # }
        Notes for agents:
            - This docstring is designed to be machine-readable. Use the Usage block to discover input/output
              shapes. Validate the returned dict keys before using them downstream.
        """
        response = self.chat.invoke(
            input=f"{self.EXTRACT_PROMPT}:{review}",
            format = Themes.model_json_schema()
        )
        result = Themes.model_validate_json(response.content)
        theme = {
            "product": result.product,
            "sentiment":result.sentiment,
            "theme":result.theme,
            "classification":result.classification,
            "issue_description":result.issue_description,

        }
        
        # print(theme)
        return theme
