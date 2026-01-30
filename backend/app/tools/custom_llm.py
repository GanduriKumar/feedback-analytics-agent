import os
import re
from urllib.parse import urlsplit, urlunsplit
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client
from app.models.schemas import LLMConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

class CustomLLMModel:
    """
    CustomLLMModel provides an interface for interacting with the Ollama LLM.
    
    Methods:
        getmodelinstance() -> OllamaLLM:
            Initializes and returns an instance of the OllamaLLM model.
        
        getchatinstance() -> ChatOllama:
            Initializes and returns an instance of the ChatOllama model.
        
        create_embedding() -> OllamaEmbeddings:
            Creates and returns an embedding model instance.
        
        create_vectorstore(input_text: list) -> Chroma:
            Processes input text, creates embeddings, and returns a Chroma vector store.
        
        getclientinterface() -> Client:
            Returns an instance of the Ollama Client for API interactions.
    """
    
    def __init__(self, llm_config: LLMConfig | None = None):
        """Initializes the CustomLLMModel with configuration from environment variables."""
        load_dotenv()
        self.PROVIDER = (llm_config.provider if llm_config and llm_config.provider else os.getenv("LLM_PROVIDER") or "ollama").lower()
        self.MODEL_URL = (llm_config.baseUrl if llm_config and llm_config.baseUrl else os.getenv("BASE_URL") or "http://localhost:11434")
        self.API_KEY = llm_config.apiKey if llm_config and llm_config.apiKey else (
            os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        self.MODEL_NAME = llm_config.model if llm_config and llm_config.model else (os.getenv("INFERENCE_MODEL") or "llama3.1")
        self.VISION_MODEL = os.getenv("VISION_MODEL")
        self.MODEL_TEMPERATURE = os.getenv('MODEL_TEMPERATURE')
        self.EMBED_MODEL = os.getenv("EMBEDDING_MODEL") or "nomic-embed-text"
        self.OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
        self.GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL") or "text-embedding-004"
        self.MAX_TOKENS = os.getenv('MODEL_MAX_TOKENS')
        self.TOP_K = os.getenv('MODEL_TOP_K')

    def _parse_float(self, value: str | None) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _parse_int(self, value: str | None) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def get_embedding_model_name(self) -> str | None:
        if self.PROVIDER == "openai":
            return self.OPENAI_EMBED_MODEL
        if self.PROVIDER == "gemini":
            return self.GEMINI_EMBED_MODEL
        if self.PROVIDER == "ollama":
            return self.EMBED_MODEL
        return None

    def get_embedding_collection_name(self, base_name: str) -> str:
        base = base_name.strip() if base_name else "reviews"
        model_name = self.get_embedding_model_name()
        if not model_name:
            return base
        safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_name).strip("-")
        safe_provider = re.sub(r"[^a-zA-Z0-9._-]+", "-", self.PROVIDER).strip("-")
        return f"{base}__{safe_provider}__{safe_model}"

    def _normalize_openai_base_url(self, base_url: str) -> str:
        if not base_url:
            return base_url
        try:
            parts = urlsplit(base_url)
        except ValueError:
            return base_url

        path = parts.path or ""
        if "/v1" in path:
            prefix, _, _ = path.partition("/v1")
            normalized_path = f"{prefix}/v1"
        elif any(segment in path for segment in ("/chat/completions", "/responses", "/embeddings")):
            normalized_path = "/v1"
        else:
            return base_url

        return urlunsplit((parts.scheme, parts.netloc, normalized_path, "", ""))

    def getmodelinstance(self) -> OllamaLLM:
        """Returns an instance of the OllamaLLM model with the configured parameters."""
        if self.PROVIDER != "ollama":
            raise ValueError("getmodelinstance is only supported for Ollama provider")

        return OllamaLLM(
            base_url=self.MODEL_URL,
            api_key=self.API_KEY,
            model=self.MODEL_NAME,
            temperature=self._parse_float(self.MODEL_TEMPERATURE),
            top_k=self._parse_int(self.TOP_K),
        )

    def getchatinstance(self):
        """Returns an instance of the ChatOllama model with the configured parameters."""
        temperature = self._parse_float(self.MODEL_TEMPERATURE)

        if self.PROVIDER == "ollama":
            return ChatOllama(
                base_url=self.MODEL_URL,
                api_key=self.API_KEY,
                model=self.MODEL_NAME,
                temperature=temperature,
            )

        if self.PROVIDER == "openai":
            from langchain_openai import ChatOpenAI
            kwargs = {"model": self.MODEL_NAME}
            if self.API_KEY:
                kwargs["api_key"] = self.API_KEY
            # Only set base_url if explicitly provided and not the default Ollama URL
            if self.MODEL_URL and self.MODEL_URL != "http://localhost:11434":
                kwargs["base_url"] = self._normalize_openai_base_url(self.MODEL_URL)
            if temperature is not None:
                kwargs["temperature"] = temperature
            return ChatOpenAI(**kwargs)

        if self.PROVIDER == "anthropic":
            from langchain_anthropic import ChatAnthropic
            kwargs = {"model": self.MODEL_NAME}
            if self.API_KEY:
                kwargs["api_key"] = self.API_KEY
            if temperature is not None:
                kwargs["temperature"] = temperature
            return ChatAnthropic(**kwargs)

        if self.PROVIDER == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            kwargs = {"model": self.MODEL_NAME}
            if self.API_KEY:
                kwargs["google_api_key"] = self.API_KEY
            if temperature is not None:
                kwargs["temperature"] = temperature
            return ChatGoogleGenerativeAI(**kwargs)

        raise ValueError(f"Unsupported LLM provider: {self.PROVIDER}")

    def create_embedding(self):
        """Creates and returns an instance of the OllamaEmbeddings model."""
        if self.PROVIDER == "ollama":
            return OllamaEmbeddings(
                base_url=self.MODEL_URL,
                model=self.EMBED_MODEL,
            )

        if self.PROVIDER == "openai":
            from langchain_openai import OpenAIEmbeddings
            kwargs = {"model": self.OPENAI_EMBED_MODEL}
            if self.API_KEY:
                kwargs["api_key"] = self.API_KEY
            # Only set base_url if explicitly provided and not the default Ollama URL
            if self.MODEL_URL and self.MODEL_URL != "http://localhost:11434":
                kwargs["base_url"] = self._normalize_openai_base_url(self.MODEL_URL)
            return OpenAIEmbeddings(**kwargs)

        if self.PROVIDER == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            kwargs = {"model": self.GEMINI_EMBED_MODEL}
            if self.API_KEY:
                kwargs["google_api_key"] = self.API_KEY
            return GoogleGenerativeAIEmbeddings(**kwargs)

        if self.PROVIDER == "anthropic":
            raise ValueError("Anthropic does not provide embeddings in this setup. Please select OpenAI, Gemini, or Ollama for embeddings.")

        raise ValueError(f"Unsupported LLM provider: {self.PROVIDER}")

    def create_vectorstore(self, input_text: list) -> Chroma:
        """
        Processes input text to create embeddings and returns a Chroma vector store.
        
        Args:
            input_text (list): A list of documents to process.
        
        Returns:
            Chroma: A handle to the created Chroma vector store.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=100,
        )
        doc_list = text_splitter.create_documents(input_text)
        documents = text_splitter.split_documents(doc_list)

        vector_store = Chroma.from_documents(
            collection_name="vector_collection",
            documents=documents,
            embedding=self.create_embedding(),
            persist_directory="./chroma_langchain.db"
        )
        return vector_store

    def getclientinterface(self) -> Client:
        """Returns an instance of the Ollama Client for API interactions."""
        if self.PROVIDER != "ollama":
            raise ValueError("Ollama client interface is only available for the Ollama provider")
        return Client(self.MODEL_URL)
