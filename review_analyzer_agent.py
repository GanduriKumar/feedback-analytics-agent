from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from feedback_analyzer import execute_graph_pipeline
from langchain.tools import tool
from langchain_core.messages import HumanMessage
import pandas as pd, csv, dotenv, os, ast

# Load environment variables from .env file (INFERENCE_MODEL, Reddit credentials, etc.)
dotenv.load_dotenv()

# Initialize ChatOllama LLM with model from environment configuration
# Streaming is disabled to ensure compatibility with agent execution flow
# Temperature 0.7 provides balanced creativity vs consistency in responses
llm = ChatOllama(model=os.getenv("INFERENCE_MODEL"), temperature=0.7, disable_streaming=True)

@tool
def get_themes(query: str) -> str:
    """
    Executes the complete feedback analytics pipeline for customer review analysis.
    
    This tool orchestrates the entire data processing workflow:
    1. Reddit data collection via PRAW
    2. Sentiment analysis using VADER
    3. Review classification with LLM
    4. Theme extraction and issue identification
    5. Clustering similar feedback
    
    Args:
        query: User query specifying what to analyze (e.g., "Pixel vs iPhone", subreddit name)
    
    Returns:
        JSON string containing extracted themes, sentiments, and insights
    """
    return execute_graph_pipeline(query)

# Register the feedback analysis tool for agent use
tools = [get_themes]

# Create LangChain agent that can autonomously decide when to invoke the get_themes tool
# The agent interprets user requests and translates them into appropriate tool calls
review_analyzer_agent = create_agent(model=llm, tools=tools)

# Prompt user for their analysis request
# Examples: "Pixel camera issues", "iPhone battery life", "Pixel vs iPhone"
query_str = input("Please enter your request for analyzing customer reviews: ")

# Construct explicit agent instruction to ensure tool invocation
# The message guides the agent to use get_themes with the user's query parameter
agent_msg = (f"Analyze customer reviews for {query_str} using the get_themes tool. "
             f"Pass '{query_str}' as input to the tool.")

# Invoke agent with the constructed message
# Agent processes the request, calls get_themes tool, and returns structured results
agent_response = review_analyzer_agent.invoke({"messages": [HumanMessage(agent_msg)]})

# Extract and process tool output from agent's message chain
# Tool messages contain the actual analysis results from the pipeline
themes_list = []
for msg in agent_response["messages"]:
    if msg.type == "tool":
        # Parse the string representation of the list/dict back to Python object
        themes_list = ast.literal_eval(msg.content)

# Convert analysis results to DataFrame and export as JSON Lines format
# Each line is a valid JSON object representing one theme/insight
df = pd.DataFrame(themes_list)
df.to_json('review_analysis_output.json', orient='records', lines=True)

