from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from feedback_analyzer import execute_graph_pipeline
from langchain.tools import tool
from langchain_core.messages import HumanMessage
import pandas as pd
import csv
import dotenv
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_query(query: str) -> str:
    """
    Validate and sanitize user query input.
    
    Args:
        query: Raw user input query
        
    Returns:
        Sanitized query string
        
    Raises:
        ValueError: If query is invalid or contains malicious content
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', query.strip())
    
    # Limit query length
    max_length = 500
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        logger.warning(f"Query truncated to {max_length} characters")
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'<script',
        r'javascript:',
        r'on\w+\s*=',  # Event handlers
        r'\.\./|\.\.\\',  # Path traversal
        r';\s*\w+\s*\(',  # Command injection attempts
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            raise ValueError(f"Query contains suspicious content: {pattern}")
    
    return sanitized


def safe_parse_json(content: str) -> List[Dict[str, Any]]:
    """
    Safely parse JSON content without using ast.literal_eval.
    
    Args:
        content: JSON string to parse
        
    Returns:
        Parsed data structure
        
    Raises:
        ValueError: If content cannot be safely parsed
    """
    try:
        # Try parsing as JSON first (safer than ast.literal_eval)
        parsed = json.loads(content)
        
        # Validate the structure
        if isinstance(parsed, list):
            if all(isinstance(item, dict) for item in parsed):
                return parsed
            else:
                raise ValueError("Invalid data structure: list must contain dictionaries")
        elif isinstance(parsed, dict):
            return [parsed]
        else:
            raise ValueError("Invalid data structure: expected list or dict")
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        raise ValueError(f"Failed to parse response content: {e}")


def get_safe_output_path(filename: str = "review_analysis_output.json") -> Path:
    """
    Get validated output file path within workspace.
    
    Args:
        filename: Desired output filename
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid or outside workspace
    """
    # Sanitize filename
    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename != filename:
        raise ValueError("Invalid filename: path traversal detected")
    
    # Ensure .json extension
    if not safe_filename.endswith('.json'):
        safe_filename += '.json'
    
    output_path = Path(os.getcwd()) / safe_filename
    
    # Verify path is within workspace
    workspace_root = Path(os.getcwd()).resolve()
    try:
        output_path.resolve().relative_to(workspace_root)
    except ValueError:
        raise ValueError("Output path is outside workspace directory")
    
    return output_path


def validate_environment() -> Dict[str, str]:
    """
    Validate required environment variables are present.
    
    Returns:
        Dictionary of validated environment variables
        
    Raises:
        ValueError: If required variables are missing
    """
    dotenv.load_dotenv()
    
    required_vars = ["INFERENCE_MODEL"]
    env_config = {}
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            raise ValueError(f"Required environment variable {var} is not set")
        env_config[var] = value
    
    return env_config


@tool
def get_themes(query: str) -> str:
    """
    Executes the feedback analytics pipeline for customer review analysis.
    
    Args:
        query: User query specifying what to analyze
    
    Returns:
        JSON string containing extracted themes and insights
        
    Raises:
        ValueError: If query is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        # Validate query before processing
        sanitized_query = validate_query(query)
        logger.info(f"Processing query: {sanitized_query[:100]}")
        
        # Execute pipeline with sanitized input
        results = execute_graph_pipeline(sanitized_query)
        
        # Return as JSON string for safe transmission
        return json.dumps(results)
        
    except ValueError as e:
        logger.error(f"Query validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise RuntimeError(f"Analysis failed: {e}")


def main():
    """Main execution function with comprehensive error handling."""
    try:
        # Validate environment
        env_config = validate_environment()
        logger.info("Environment validated successfully")
        
        # Initialize LLM with validated configuration
        llm = ChatOllama(
            model=env_config["INFERENCE_MODEL"],
            temperature=0.7,
            disable_streaming=True
        )
        
        # Register tools
        tools = [get_themes]
        
        # Create agent
        review_analyzer_agent = create_agent(model=llm, tools=tools)
        logger.info("Agent initialized successfully")
        
        # Get and validate user input
        print("Please enter your request for analyzing customer reviews:")
        print("Examples: 'Pixel camera issues', 'iPhone battery life', 'Pixel vs iPhone'")
        query_str = input("> ").strip()
        
        # Validate query
        try:
            sanitized_query = validate_query(query_str)
        except ValueError as e:
            logger.error(f"Invalid query: {e}")
            print(f"Error: {e}")
            return
        
        # Construct agent message
        agent_msg = (
            f"Analyze customer reviews for {sanitized_query} using the get_themes tool. "
            f"Pass '{sanitized_query}' as input to the tool."
        )
        
        # Invoke agent
        logger.info("Invoking agent...")
        agent_response = review_analyzer_agent.invoke({
            "messages": [HumanMessage(agent_msg)]
        })
        
        # Extract and safely parse results
        themes_list = []
        for msg in agent_response["messages"]:
            if msg.type == "tool":
                try:
                    # Use safe JSON parsing instead of ast.literal_eval
                    themes_list = safe_parse_json(msg.content)
                    logger.info(f"Successfully parsed {len(themes_list)} themes")
                except ValueError as e:
                    logger.error(f"Failed to parse tool output: {e}")
                    print(f"Error: Failed to parse analysis results: {e}")
                    return
        
        if not themes_list:
            logger.warning("No themes extracted from analysis")
            print("Warning: No themes were extracted from the analysis")
            return
        
        # Get safe output path
        output_path = get_safe_output_path("review_analysis_output.json")
        
        # Write results with error handling
        try:
            df = pd.DataFrame(themes_list)
            df.to_json(str(output_path), orient='records', lines=True)
            
            # Set secure file permissions (owner read/write only)
            os.chmod(str(output_path), 0o600)
            
            logger.info(f"Results written to {output_path}")
            print(f"\nAnalysis complete. Results saved to: {output_path}")
            print(f"Total themes extracted: {len(themes_list)}")
            
        except Exception as e:
            logger.error(f"Failed to write output file: {e}")
            print(f"Error: Failed to save results: {e}")
            return
            
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Configuration Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"Unexpected Error: {e}")


if __name__ == "__main__":
    main()

