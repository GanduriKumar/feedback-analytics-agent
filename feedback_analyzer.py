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
    """
    AI Schema: InputState
    Purpose:
      Represents the initial state of the feedback analysis pipeline after querying
      the vector database. Contains the user's query and the raw reviews extracted.

    Fields:
      query (str): User's natural language product search query (e.g., "Pixel vs iPhone battery").
      extracted_reviews (list[str]): Raw review texts retrieved from vector DB matching the query.

    Usage Context:
      - Entry point for the pipeline after Review_Extractor node.
      - Input to Cluster_Assessor node.

    Validation:
      All fields are required (Field(...)).

    Example:
      InputState(
        query="Compare Pixel and iPhone camera quality",
        extracted_reviews=["Great camera but...", "Poor low light performance..."]
      )
    """
    query: str = Field(..., description="The user's product-related search query.")
    extracted_reviews: list[str] = Field(..., description="List of reviews extracted from the vector database based on the user's query.")

# State after clustering reviews into groups
class ClusterState(BaseModel):
    """
    AI Schema: ClusterState
    Purpose:
      Extends InputState with clustering results. Represents the state after reviews
      have been grouped into similarity-based clusters.

    Fields:
      query (str): Original user query (inherited).
      extracted_reviews (list[str]): Original reviews (inherited).
      clusters (dict[int, list[Optional[str]]]): Mapping of cluster_id to list of review texts.
                                                   Key: cluster number, Value: reviews in that cluster.

    Usage Context:
      - Output from Cluster_Assessor node (assess_clusters).
      - Input to Cluster_Summarizer node.

    Data Flow:
      InputState -> assess_clusters() -> ClusterState

    Example:
      ClusterState(
        query="Battery complaints",
        extracted_reviews=["...", "..."],
        clusters={0: ["Battery dies fast", "Poor battery"], 1: ["Good battery life"]}
      )
    """
    query: str = Field(..., description="The user's product-related search query.")
    extracted_reviews: list[str] = Field(..., description="List of reviews extracted from the vector database based on the user's query.")
    clusters: dict[int, list[Optional[str]]]= Field(..., description="List of cluster assessments based on the cleaned reviews.")
  
# State after generating summaries for each cluster
class SummaryState(BaseModel):
    """
    AI Schema: SummaryState
    Purpose:
      Extends ClusterState with human-readable summaries for each cluster.
      Represents the state after LLM-based summarization of clustered reviews.

    Fields:
      query (str): Original user query (inherited).
      extracted_reviews (list[str]): Original reviews (inherited).
      clusters (dict[int, list[Optional[str]]]): Cluster mappings (inherited).
      cluster_summaries (list[str]): Ordered list of summary texts, one per cluster.
                                      Index corresponds to cluster_id.

    Usage Context:
      - Output from Cluster_Summarizer node (summarize_clusters).
      - Input to Theme_Extractor node.

    Data Flow:
      ClusterState -> summarize_clusters() -> SummaryState

    Example:
      SummaryState(
        query="Battery issues",
        extracted_reviews=[...],
        clusters={0: [...], 1: [...]},
        cluster_summaries=["Cluster 0: Users report rapid battery drain", "Cluster 1: Positive battery feedback"]
      )
    """
    query: str = Field(..., description="The user's product-related search query.")
    extracted_reviews: list[str] = Field(..., description="List of reviews extracted from the vector database based on the user's query.")
    clusters: dict[int, list[Optional[str]]]= Field(..., description="List of cluster assessments based on the cleaned reviews.")
    cluster_summaries: list[str] = Field(..., description="Summarized main points from the assessed clusters of customer reviews.")

# Final state containing extracted themes from cluster summaries
class ThemesState(BaseModel):
    """
    AI Schema: ThemesState
    Purpose:
      Final state schema representing the complete pipeline output. Contains extracted
      themes/patterns derived from cluster summaries using LLM analysis.

    Fields:
      query (str): Original user query (inherited).
      extracted_reviews (list[str]): Original reviews (inherited).
      clusters (dict[int, list[Optional[str]]]): Cluster mappings (inherited).
      cluster_summaries (list[str]): Cluster summaries (inherited).
      themes (list[dict[str, str]]): Extracted themes as structured dictionaries.
                                      Each dict typically contains keys like 'theme', 'description', 'sentiment'.

    Usage Context:
      - Output from Theme_Extractor node (extract_themes).
      - Final state exported to feedback_analysis_results.json.

    Data Flow:
      SummaryState -> extract_themes() -> ThemesState -> JSON export

    Persistence:
      Serialized to JSONL file via pandas DataFrame.

    Example:
      ThemesState(
        query="Pixel battery issues",
        extracted_reviews=[...],
        clusters={...},
        cluster_summaries=[...],
        themes=[
          {"theme": "Battery Drain", "description": "Rapid discharge during normal use", "sentiment": "negative"},
          {"theme": "Charging Speed", "description": "Slow charging complaints", "sentiment": "negative"}
        ]
      )
    """
    query: str = Field(..., description="The user's product-related search query.")
    extracted_reviews: list[str] = Field(..., description="List of reviews extracted from the vector database based on the user's query.")
    clusters: dict[int, list[Optional[str]]]= Field(..., description="List of cluster assessments based on the cleaned reviews.")
    cluster_summaries: list[str] = Field(..., description="Summarized main points from the assessed clusters of customer reviews.")
    # themes: dict[str, str,str, str,str] = Field(..., description="Extracted key themes from the summarized clusters of customer reviews.")
    themes: list[dict[str, str]] = Field(..., description="Extracted key themes from the summarized clusters of customer reviews.")

# Node 1: Query vector database and extract relevant reviews based on user query
def query_vector_db_tool(state:InputState)->InputState:
    """
    AI Tool: query_vector_db_tool
    Purpose:
      LangGraph node that queries the vector database to retrieve reviews matching
      the user's product query. Acts as the data ingestion stage of the pipeline.

    Parameters:
      state (InputState): Contains user query; extracted_reviews is initially empty.

    Returns:
      InputState: Updated state with populated extracted_reviews field.

    Behavior:
      - Calls query_vector_db() with the user query.
      - Returns new InputState preserving query and adding retrieved reviews.

    Dependencies:
      - query_vectorDB.query_vector_db function
      - Vector database must be initialized and populated

    Side Effects:
      - Database queries (read-only)

    Example Flow:
      InputState(query="Pixel battery") -> query_vector_db_tool() ->
      InputState(query="Pixel battery", extracted_reviews=["Review 1", "Review 2"])
    """
    reviews = query_vector_db(state.query)
    return InputState(query=state.query, extracted_reviews=reviews)

# Node 2: Group extracted reviews into clusters based on similarity
def assess_clusters_tool(state:InputState)->ClusterState:
    """
    AI Tool: assess_clusters_tool
    Purpose:
      LangGraph node that performs unsupervised clustering on extracted reviews
      to group similar feedback together using sentence embeddings.

    Parameters:
      state (InputState): Contains query and extracted_reviews.

    Returns:
      ClusterState: Extended state with clusters field populated.

    Behavior:
      - Calls assess_clusters() which uses sentence transformers for embeddings.
      - Groups reviews into clusters based on semantic similarity.
      - Returns new ClusterState with all previous fields plus cluster mappings.

    Dependencies:
      - src.tools.custom_tools.assess_clusters function
      - Sentence transformer models (defined in review_clustering.py)

    Algorithm:
      Typically uses DBSCAN or K-means clustering on sentence embeddings.

    Example Flow:
      InputState(extracted_reviews=["Review 1", "Review 2"]) -> assess_clusters_tool() ->
      ClusterState(clusters={0: ["Review 1"], 1: ["Review 2"]})
    """
    clusters = assess_clusters(state.extracted_reviews)
    return ClusterState(query=state.query, extracted_reviews=state.extracted_reviews, clusters=clusters)
    
# Node 3: Generate concise summaries for each cluster of reviews
def summarize_clusters_tool(state:ClusterState)->SummaryState:
    """
    AI Tool: summarize_clusters_tool
    Purpose:
      LangGraph node that generates human-readable summaries for each cluster
      using LLM-based text summarization.

    Parameters:
      state (ClusterState): Contains query, reviews, and cluster mappings.

    Returns:
      SummaryState: Extended state with cluster_summaries field populated.

    Behavior:
      - Calls summarize_clusters() which uses CustomLLMModel (Ollama).
      - Generates one summary per cluster capturing main themes.
      - Returns new SummaryState preserving all previous fields plus summaries.

    Dependencies:
      - src.tools.custom_tools.summarize_clusters function
      - CustomLLMModel (Ollama) for LLM operations

    LLM Integration:
      Uses completion or chat mode to condense cluster reviews into summaries.

    Example Flow:
      ClusterState(clusters={0: ["Bad battery", "Dies fast"]}) -> summarize_clusters_tool() ->
      SummaryState(cluster_summaries=["Users report rapid battery drain issues"])
    """
    summaries = summarize_clusters(state.clusters)
    return SummaryState(query=state.query, extracted_reviews=state.extracted_reviews, clusters=state.clusters, cluster_summaries=summaries)
    
# Node 4: Extract key themes and patterns from cluster summaries
def extract_themes_tool(state:SummaryState)->ThemesState:
    """
    AI Tool: extract_themes_tool
    Purpose:
      LangGraph node that performs final thematic analysis on cluster summaries
      to extract structured themes/patterns using LLM.

    Parameters:
      state (SummaryState): Contains query, reviews, clusters, and summaries.

    Returns:
      ThemesState: Final state with themes field populated as structured dicts.

    Behavior:
      - Calls extract_themes() which analyzes summaries using LLM.
      - Identifies key themes, assigns sentiment, and structures output.
      - Returns complete ThemesState ready for export to JSON.

    Dependencies:
      - src.tools.custom_tools.extract_themes function
      - CustomLLMModel (Ollama) for LLM-based theme extraction

    Output Schema:
      themes as list[dict[str, str]] with keys like 'theme', 'description', 'sentiment'.

    Termination:
      This is the final node before END in the pipeline.

    Example Flow:
      SummaryState(cluster_summaries=["Battery issues", "Great camera"]) -> extract_themes_tool() ->
      ThemesState(themes=[{"theme": "Battery", "sentiment": "negative"}, {"theme": "Camera", "sentiment": "positive"}])
    """
    themes= extract_themes(state.cluster_summaries)
    return ThemesState(query=state.query, extracted_reviews=state.extracted_reviews, clusters=state.clusters, cluster_summaries=state.cluster_summaries, themes=themes)


def execute_graph_pipeline(user_query: str):
    """
    AI Tool: execute_graph_pipeline
    Purpose:
      Orchestrates the end-to-end feedback analysis pipeline using LangGraph:
      1) query_vector_db -> 2) assess_clusters -> 3) summarize_clusters -> 4) extract_themes.
      Produces a JSONL file with extracted themes and a PNG graph of the pipeline.

    Parameters:
      user_query (str): Natural language query describing a product concern/comparison
                        (e.g., "Pixel battery life vs iPhone").

    Behavior:
      - Builds a StateGraph with four nodes:
          - Review_Extractor: fetches relevant reviews from a vector database.
          - Cluster_Assessor: groups reviews into clusters.
          - Cluster_Summarizer: generates concise summaries per cluster.
          - Theme_Extractor: extracts key themes from summaries.
      - Compiles and invokes the graph with an initial ThemesState containing the user query.
      - Serializes the resulting themes to `feedback_analysis_results.json` (JSON Lines).
      - Exports a pipeline diagram to `agent_graph.png`.

    Input/Output Schema:
      Input: ThemesState(query=str, extracted_reviews=[], clusters={}, cluster_summaries=[], themes=[])
      Output file: feedback_analysis_results.json (list[dict[str, str]] as JSONL)
      Output file: agent_graph.png (graph visualization)

    Side Effects:
      - File I/O: writes PNG and JSONL files in the working directory.
      - May invoke external systems via query_vector_db and downstream tools.

    Determinism:
      Non-deterministic due to clustering/LLM components; repeated runs may vary.

    Example:
      execute_graph_pipeline("Compare camera quality issues between Pixel and iPhone")
    """
    # System message for future LLM integration (not used directly here)
    sys_msg = (
        "You are a helpful assistant tasked with performing analysis of the product "
        "reviews stored in a vector database. Use the provided tool to query the "
        "database based on user queries."
    )

    # Initialize the state graph with the final state schema
    # print(f"query string in execute_graph_pipeline: {user_query}")
    graph = StateGraph(ThemesState)

    # Register pipeline stages as nodes
    graph.add_node("Review_Extractor", query_vector_db_tool)
    graph.add_node("Cluster_Assessor", assess_clusters_tool)
    graph.add_node("Cluster_Summarizer", summarize_clusters_tool)
    graph.add_node("Theme_Extractor", extract_themes_tool)

    # Define execution flow
    graph.add_edge(START, "Review_Extractor")
    graph.add_edge("Review_Extractor", "Cluster_Assessor")
    graph.add_edge("Cluster_Assessor", "Cluster_Summarizer")
    graph.add_edge("Cluster_Summarizer", "Theme_Extractor")
    graph.add_edge("Theme_Extractor", END)

    # Compile the graph into an executable agent
    agent_graph = graph.compile()

    # Export the graph visualization for debugging/observability
    with open("agent_graph.png", "wb") as f:
        f.write(agent_graph.get_graph().draw_mermaid_png())

    # Seed initial state for invocation
    initial_state = ThemesState(
        query=user_query,
        extracted_reviews=[],
        clusters={},
        cluster_summaries=[],
        themes=[],
    )

    # Execute the pipeline and collect results
    final_state = agent_graph.invoke(initial_state)
    items_dict = dict(final_state.items())

    # Persist themes as JSON Lines for downstream analytics
    df = pd.DataFrame(items_dict["themes"])
    df.to_json("feedback_analysis_results.json", orient="records", lines=True)
    return items_dict["themes"]

if __name__ == "__main__":
    user_query = input("Enter your product related search query: ")
    execute_graph_pipeline(user_query)