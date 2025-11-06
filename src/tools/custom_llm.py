import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client

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
    
    def __init__(self):
        """Initializes the CustomLLMModel with configuration from environment variables."""
        load_dotenv()
        self.MODEL_URL = os.getenv("BASE_URL")
        self.API_KEY = os.getenv("API_KEY")
        self.MODEL_NAME = os.getenv("INFERENCE_MODEL")
        self.VISION_MODEL = os.getenv("VISION_MODEL")
        self.MODEL_TEMPERATURE = os.getenv('MODEL_TEMPERATURE')
        self.EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
        self.MAX_TOKENS = os.getenv('MODEL_MAX_TOKENS')
        self.TOP_K = os.getenv('MODEL_TOP_K')

    def getmodelinstance(self) -> OllamaLLM:
        """Returns an instance of the OllamaLLM model with the configured parameters."""
        return OllamaLLM(
            base_url=self.MODEL_URL,
            api_key=self.API_KEY,
            model=self.MODEL_NAME,
            temperature=self.MODEL_TEMPERATURE,
            top_k=self.TOP_K
        )

    def getchatinstance(self) -> ChatOllama:
        """Returns an instance of the ChatOllama model with the configured parameters."""
        return ChatOllama(
            base_url=self.MODEL_URL,
            api_key=self.API_KEY,
            model=self.MODEL_NAME,
            temperature=self.MODEL_TEMPERATURE
        )

    def create_embedding(self) -> OllamaEmbeddings:
        """Creates and returns an instance of the OllamaEmbeddings model."""
        embeddings = OllamaEmbeddings(
            base_url=self.MODEL_URL,
            model=self.EMBED_MODEL,
        )
        return embeddings

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
        return Client(self.MODEL_URL)
