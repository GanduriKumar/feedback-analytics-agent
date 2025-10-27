import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client
# from custom_configs import OUTPUT_FORMAT, MODEL_TEMPERATURE

class LLMModel:
    """
    This class defines a custom model for use with Ollama
    --------------------------------------------------------
    Methods:
        getinstance() :
            args: None
            return: Return the handle the model with specific parameters
        create_embedding():
            args: None
            return: Returns the handle to embedding function
        create_vectorstore():
            args: document list as argument
            return: returns the handle to Chroma vectorDB store
        getclientinterface():
            args: None
            return: handle of the Ollama Client interface of the llm
    """
    def __init__(self):
        """constructor for the LLMModel class and populates the host, api key and the model to use"""
        load_dotenv()
        self.MODEL_URL = os.getenv("BASE_URL")
        self.API_KEY = os.getenv("API_KEY")
        self.MODEL_NAME = os.getenv("INFERENCE_MODEL")
        self.VISION_MODEL = os.getenv("VISION_MODEL")
        self.MODEL_TEMPERATURE= os.getenv('MODEL_TEMPERATURE')
        self.EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
        self.MAX_TOKENS = os.getenv('MODEL_MAX_TOKENS')
        self.TOP_K = os.getenv('MODEL_TOP_K')
    def getmodelinstance(self):
        """Return the handle to the specific custom model
        return: OllamaLLM model with requisite configuration
        """
        return OllamaLLM(
            base_url=self.MODEL_URL,
            api_key=self.API_KEY,
            model=self.MODEL_NAME,
            temperature=self.MODEL_TEMPERATURE,
            top_k= self.TOP_K
        )
    def getchatinstance(self):
        """Return the handle to the specific custom model
        return: ChatOllama model with requisite configuration
        """
        return ChatOllama(
            base_url=self.MODEL_URL,
            api_key=self.API_KEY,
            model=self.MODEL_NAME,
            # format=OUTPUT_FORMAT,
            temperature=self.MODEL_TEMPERATURE
        )
    def create_embedding(self) -> OllamaEmbeddings:
        """create embedding
        return: List of embedding vectors
        """
        embeddings = OllamaEmbeddings(
            base_url=self.MODEL_URL,
            model=self.EMBED_MODEL,
        )
        return embeddings

    def create_vectorstore(self,input_text:list):
        """
        splits the input text in chunks with overlap,
        create embeddings using OllamaEmbedding add chunks to vector store
        and return the handle to the vector store
        :param input_text: list of documents
        :returns: Chroma vector store
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=100,
            # length_function=len
            )
        # use the text splitter to create and split the documents
        doc_list = text_splitter.create_documents(input_text)
        documents = text_splitter.split_documents(doc_list)

        # create a persistent Chroma vector store for the list of documents
        vector_store = Chroma.from_documents(
            collection_name="vector_collection",
            documents=documents,
            embedding=self.create_embedding(),
            persist_directory="./chroma_langchain.db"
        )
        return vector_store # returns the vector store handle
    def getclientinterface(self)->Client:
        """
        Returns the Ollama client for a chat/generate/create interface
        :return: ollama Client object
        """
        return Client(self.MODEL_URL)