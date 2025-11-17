from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from query_vectorDB import query_vector_db
from src.tools.custom_tools import assess_clusters, summarize_clusters, extract_themes
import pandas as pd
from typing import Optional

# Define the initial state containing user query and extracted reviews
class InputState(BaseModel):
    query: str = Field(..., description="The user's product-related search query.")
    extracted_reviews: list[str] = Field(..., description="List of reviews extracted from the vector database based on the user's query.")

# State after clustering reviews into groups
class ClusterState(BaseModel):
    query: str = Field(..., description="The user's product-related search query.")
    extracted_reviews: list[str] = Field(..., description="List of reviews extracted from the vector database based on the user's query.")
    clusters: dict[int, list[Optional[str]]]= Field(..., description="List of cluster assessments based on the cleaned reviews.")
  
# State after generating summaries for each cluster
class SummaryState(BaseModel):
    query: str = Field(..., description="The user's product-related search query.")
    extracted_reviews: list[str] = Field(..., description="List of reviews extracted from the vector database based on the user's query.")
    clusters: dict[int, list[Optional[str]]]= Field(..., description="List of cluster assessments based on the cleaned reviews.")
    cluster_summaries: list[str] = Field(..., description="Summarized main points from the assessed clusters of customer reviews.")

# Final state containing extracted themes from cluster summaries
class ThemesState(BaseModel):
    query: str = Field(..., description="The user's product-related search query.")
    extracted_reviews: list[str] = Field(..., description="List of reviews extracted from the vector database based on the user's query.")
    clusters: dict[int, list[Optional[str]]]= Field(..., description="List of cluster assessments based on the cleaned reviews.")
    cluster_summaries: list[str] = Field(..., description="Summarized main points from the assessed clusters of customer reviews.")
    # themes: dict[str, str,str, str,str] = Field(..., description="Extracted key themes from the summarized clusters of customer reviews.")
    themes: list[dict[str, str]] = Field(..., description="Extracted key themes from the summarized clusters of customer reviews.")

# Node 1: Query vector database and extract relevant reviews based on user query
def query_vector_db_tool(state:InputState)->InputState:
    reviews = query_vector_db(state.query)
    return InputState(query=state.query, extracted_reviews=reviews)

# Node 2: Group extracted reviews into clusters based on similarity
def assess_clusters_tool(state:InputState)->ClusterState:
    clusters = assess_clusters(state.extracted_reviews)
    return ClusterState(query=state.query, extracted_reviews=state.extracted_reviews, clusters=clusters)
    
# Node 3: Generate concise summaries for each cluster of reviews
def summarize_clusters_tool(state:ClusterState)->SummaryState:
    summaries = summarize_clusters(state.clusters)
    return SummaryState(query=state.query, extracted_reviews=state.extracted_reviews, clusters=state.clusters, cluster_summaries=summaries)
    
# Node 4: Extract key themes and patterns from cluster summaries
def extract_themes_tool(state:SummaryState)->ThemesState:
    themes= extract_themes(state.cluster_summaries)
    return ThemesState(query=state.query, extracted_reviews=state.extracted_reviews, clusters=state.clusters, cluster_summaries=state.cluster_summaries, themes=themes)


if  __name__ == "__main__":

    # model = ChatOllama(model = "mistral:latest")
    # model_with_tools = model.bind_tools(tools)
    
    # System message defining the assistant's role
    sys_msg = f"You are a helpful assistant tasked with performing analysis of the product reviews stored in a vector database. Use the provided tool to query the database based on user queries."
    # prompt = StringPromptTemplate(f"{sys_msg}")
    # llm_chain = LLMChain(llm=model, prompt=prompt)
    
    # Initialize the state graph with the final state schema
    graph = StateGraph(ThemesState)
    
    # Add nodes representing each stage of the analysis pipeline
    graph.add_node("Review_Extractor",query_vector_db_tool)
    graph.add_node("Cluster_Assessor",assess_clusters_tool)
    graph.add_node("Cluster_Summarizer",summarize_clusters_tool)
    graph.add_node("Theme_Extractor",extract_themes_tool)

    # Define the pipeline flow: START -> Extract -> Cluster -> Summarize -> Extract Themes -> END
    graph.add_edge(START, "Review_Extractor")
    graph.add_edge("Review_Extractor", "Cluster_Assessor")
    graph.add_edge("Cluster_Assessor", "Cluster_Summarizer")
    graph.add_edge("Cluster_Summarizer", "Theme_Extractor")
    graph.add_edge("Theme_Extractor", END)
    
    # Compile the graph into an executable agent
    agent_graph = graph.compile()
    
    # Export the graph visualization as a PNG file
    with open("agent_graph.png", "wb") as f:
        f.write(agent_graph.get_graph().draw_mermaid_png())

    # Get user input query
    user_query = input("Enter your product related search query: ")
    
    # Initialize the state with empty values (will be populated by pipeline)
    initial_state = ThemesState(query=user_query, extracted_reviews=[],clusters={}, cluster_summaries=[], themes=[])
    
    # Execute the agent graph pipeline
    final_state = agent_graph.invoke(initial_state)
    items_dict = dict(final_state.items())
    
    # Convert extracted themes to DataFrame and save as JSON
    df = pd.DataFrame(items_dict['themes'])
    df.to_json("feedback_analysis_results.json", orient="records", lines=True)
   

