import os
import asyncio
import re
from typing import List, Optional
from datetime import datetime

# Basic imports first
import requests
import bs4
from bs4 import BeautifulSoup
import json
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from helpers.WebSearchHelper import WebSearchHelper
from helpers.WebScrapeAgent import WebScrapeAgent


# Test ReActAgent imports
print("\nTesting ReActAgent imports...")
REACT_AGENT = None
REACT_IMPORT_ERROR = None

try:
    from llama_index.core.agent import ReActAgent
    REACT_AGENT = ReActAgent
except ImportError as e:
    print(f"✗ ReActAgent import from core.agent failed: {e}")
    REACT_IMPORT_ERROR = str(e)

if REACT_AGENT is None:
    try:
        from llama_index.agent.react import ReActAgent  # type: ignore
        REACT_AGENT = ReActAgent

    except ImportError as e:
        print(f"✗ ReActAgent import from agent.react failed: {e}")
        REACT_IMPORT_ERROR = str(e)

if REACT_AGENT is None:
    print(f"✗ No ReActAgent available. Last error: {REACT_IMPORT_ERROR}")
    print("Will use WebScrapeAgent fallback")

class OllamaIntegration:
    """Main assistant class that combines Ollama with web search"""
    
    def __init__(self, 
                 model_name: str = "dolphin-llama3",
                 ollama_base_url: str = "http://localhost:11434",
                 search_api_key: Optional[str] = None):
        
        print(f"🚀 Initializing with model: {model_name}")
        
        # Initialize Ollama LLM
        self.llm = Ollama(
            model=model_name,
            base_url=ollama_base_url,
            request_timeout=120.0
        )
        
        # Set up embeddings (using local HuggingFace model)
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Configure LlamaIndex settings
        Settings.llm = self.llm
        Settings.embed_model = embed_model
        
        # Initialize web search tool
        self.web_search = WebSearchHelper(search_api_key)
        
        # Create the agent
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create an agent with web search capabilities"""
        print("🔧 Creating agent...")
        
        def search_web(query: str) -> str:
            """
            A function to perform a web search and return formatted results.
            Call this when you need to find up-to-date information, or otherwise 
            can't answer a user question. Formulate your response based on the search 
            results.

            args:
                query (str): The search query string.
            
            returns:
                str: Formatted search results including titles, snippets, and URLs.
            """
            print(f"🔍 Searching web for: {query}")
            results = self.web_search.search_brave(query, max_results=3)
            
            if not results:
                return "No search results found for this query."
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_result = f"\n--- Result {i} ---"
                formatted_result += f"\nTitle: {result['title']}"
                formatted_result += f"\nContent: {result['content'][:500]}..."
                if result['url']:
                    formatted_result += f"\nURL: {result['url']}"
                formatted_results.append(formatted_result)
            
            return "\n".join(formatted_results)
        
        def get_webpage_content(url: str) -> str:
            """
            Extracts and returns the text content from a specific webpage URL.
            Use this tool when:
            - Asked to read or analyze a specific webpage
            - Given a URL to extract information from
            - Need to get the full content of an article
            
            Args:
                url (str): The full webpage URL (must start with http:// or https://)
            
            Returns:
                str: The extracted text content from the webpage
            
            Example:
                get_webpage_content("https://example.com/article")
            """
            print(f"📄 Extracting content from: {url}")
            content = self.web_search.extract_webpage_content(url)
            return content if content else "Could not extract content from this URL."
        
        def get_current_datetime() -> str:
            """
            A function to return todays date and time.
            Call this before any other functions if you are unaware of the current date or time.
            """
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert functions to LlamaIndex tools
        search_tool = FunctionTool.from_defaults(fn=search_web)
        webpage_tool = FunctionTool.from_defaults(fn=get_webpage_content)
        datetime_tool = FunctionTool.from_defaults(fn=get_current_datetime)
        
        tools = [search_tool, webpage_tool, datetime_tool]
        
        # Add custom system prompt
        system_prompt = """You are an AI assistant with access to several tools:
        - search_web: Use this for finding current information or when you need to search the internet
        - get_webpage_content: Use this when given a specific URL or asked to analyze a webpage
        - get_current_datetime: Use this to check the current date/time
        
        Important:
        1. When a user provides a URL or asks about a specific webpage, ALWAYS use get_webpage_content
        2. For general information needs, use search_web
        3. Always check the current time with get_current_datetime before searching for current events
        
        Think carefully about which tool best fits the user's request."""

        # Create agent with custom prompt
        if REACT_AGENT and hasattr(REACT_AGENT, 'from_tools'):
            try:
                print("🧪 Attempting to create ReActAgent...")
                agent = REACT_AGENT.from_tools(
                    tools,
                    llm=self.llm,
                    verbose=True,
                    max_iterations=10,
                    system_prompt=system_prompt  # Add the custom prompt
                )
                print("✓ ReActAgent created successfully!")
                return agent
            except Exception as e:
                print(f"✗ ReActAgent creation failed: {e}")
                print("  Falling back to WebScrapeAgent")
        
        # Use WebScrapeAgent as fallback
        print("🤖 Creating WebScrapeAgent")
        return WebScrapeAgent(tools, self.llm)
    
    def chat(self, message: str) -> str:
        """Send a message to the assistant and get a response."""
        print(f"\n💬 User: {message}")
        print("🤖 Assistant is thinking...")
        
        # Add URL detection
        url_pattern = r'https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)'
        urls = re.findall(url_pattern, message)
        
        if urls:
            # If URLs are found, explicitly mention them in the prompt
            message = f"""This request contains the following URLs that you should analyze using get_webpage_content: {urls}
            Original request: {message}"""
        
        try:
            response = self.agent.chat(message)
            return str(response)
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    async def achat(self, message: str) -> str:
        """Async version of chat"""
        return self.chat(message)
