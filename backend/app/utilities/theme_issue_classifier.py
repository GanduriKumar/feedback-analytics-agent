from app.tools.custom_llm import CustomLLMModel
from app.models.schemas import LLMConfig
from pydantic import BaseModel
import pandas as pd, csv
import re
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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
    def __init__(self, llm_config: LLMConfig | None = None):
        self._model = CustomLLMModel(llm_config)
        self.chat = self._model.getchatinstance()
        self._sentiment = SentimentIntensityAnalyzer()
        # Cache the prompt to avoid string concatenation overhead
        self._prompt_template = (
            "You are a JSON-only response generator. "
            "Extract the following information from the given review text. "
            "Return ONLY a JSON object with these fields: product, sentiment, theme, classification, issue_description. "
            "Allowed sentiment values: positive, negative, neutral. "
            "Do not include any extra keys or any commentary.\n\n"
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

    @staticmethod
    def _safe_extract_json(content: str) -> dict | None:
        if not content:
            return None

        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try to extract the first JSON object from the text
        start = content.find("{")
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(content)):
            ch = content[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = content[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        return None
        return None

    @staticmethod
    def _shorten_theme(value: str | None) -> str:
        raw = (value or '').strip().lower()
        if not raw or raw in {"unknown", "unclassified", "general"}:
            return "General"

        mapping = {
            "connectivity": "Connectivity",
            "network": "Connectivity",
            "bluetooth": "Connectivity",
            "wifi": "Connectivity",
            "battery": "Battery",
            "charging": "Battery",
            "power": "Battery",
            "camera": "Camera",
            "display": "Display",
            "screen": "Display",
            "performance": "Performance",
            "stability": "Stability",
            "crash": "Stability",
            "freeze": "Stability",
            "audio": "Audio",
            "speaker": "Audio",
            "mic": "Audio",
            "update": "Update",
            "software update": "Update",
            "pricing": "Pricing",
            "price": "Pricing",
            "cost": "Pricing",
            "design": "Design",
            "ux": "UX",
            "ui": "UX",
            "usability": "UX",
            "customer service": "Support",
            "support": "Support",
        }

        for key, short in mapping.items():
            if key in raw:
                return short

        return raw.title()

    def _fallback_theme(self, review: str) -> dict:
        lowered = review.lower()

        # Sentiment via VADER
        score = self._sentiment.polarity_scores(review).get("compound", 0.0)
        if score >= 0.05:
            sentiment = "positive"
        elif score <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Product heuristic
        if "pixel" in lowered:
            product = "Pixel"
        elif "iphone" in lowered:
            product = "iPhone"
        elif "android" in lowered:
            product = "Android"
        else:
            product = "unknown"

        theme = "general"
        classification = "feedback"
        keyword_map = [
            ("bluetooth", "connectivity", "bug"),
            ("connectivity", "connectivity", "bug"),
            ("wifi", "connectivity", "bug"),
            ("signal", "connectivity", "bug"),
            ("network", "connectivity", "bug"),
            ("battery", "battery", "complaint"),
            ("charging", "battery", "complaint"),
            ("camera", "camera", "complaint"),
            ("screen", "display", "complaint"),
            ("display", "display", "complaint"),
            ("overheat", "performance", "bug"),
            ("heat", "performance", "bug"),
            ("lag", "performance", "bug"),
            ("slow", "performance", "bug"),
            ("crash", "stability", "bug"),
            ("freeze", "stability", "bug"),
            ("audio", "audio", "bug"),
            ("speaker", "audio", "bug"),
            ("mic", "audio", "bug"),
            ("update", "software update", "bug"),
            ("price", "pricing", "pricing"),
            ("cost", "pricing", "pricing"),
            ("expensive", "pricing", "pricing"),
            ("design", "design", "feedback"),
            ("build", "design", "feedback"),
        ]

        for keyword, mapped_theme, mapped_class in keyword_map:
            if keyword in lowered:
                theme = mapped_theme
                classification = mapped_class
                break

        issue_description = review.strip()[:200]

        return {
            "product": product,
            "sentiment": sentiment,
            "theme": self._shorten_theme(theme),
            "classification": classification,
            "issue_description": issue_description,
        }
    
    def extract_themes(self, review: str) -> dict:
        """Extract themes with input validation and caching"""
        if not review or not isinstance(review, str):
            return self._fallback_theme("")
        
        # Use cached sanitization
        sanitized_review = self._sanitize_review(review)
        
        # Use formatted template instead of f-string concatenation
        prompt = self._prompt_template.format(review=sanitized_review)
        
        try:
            if self._model.PROVIDER == "ollama":
                response = self.chat.invoke(
                    input=prompt,
                    format=Themes.model_json_schema()
                )
            else:
                response = self.chat.invoke(input=prompt)

            parsed = self._safe_extract_json(getattr(response, "content", ""))
            if parsed is None:
                raise ValueError("Model did not return valid JSON")

            result = Themes.model_validate(parsed)

            return {
                "product": result.product or "unknown",
                "sentiment": result.sentiment or "unknown",
                "theme": self._shorten_theme(result.theme),
                "classification": result.classification or "feedback",
                "issue_description": result.issue_description or "",
            }
        except Exception as e:
            print(f"Error extracting themes: {e}")
            return self._fallback_theme(sanitized_review)
