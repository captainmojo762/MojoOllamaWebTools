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
    print("✓ ReActAgent imported from llama_index.core.agent")
    print(f"  Available methods: {[method for method in dir(ReActAgent) if not method.startswith('_')]}")
    print(f"  Has from_tools: {hasattr(ReActAgent, 'from_tools')}")
except ImportError as e:
    print(f"✗ ReActAgent import from core.agent failed: {e}")
    REACT_IMPORT_ERROR = str(e)

if REACT_AGENT is None:
    try:
        from llama_index.agent.react import ReActAgent  # type: ignore
        REACT_AGENT = ReActAgent
        print("✓ ReActAgent imported from llama_index.agent.react")
        print(f"  Available methods: {[method for method in dir(ReActAgent) if not method.startswith('_')]}")
        print(f"  Has from_tools: {hasattr(ReActAgent, 'from_tools')}")
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
            """Search the web for current information about a given topic."""
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
            """Extract and return the text content from a specific webpage."""
            print(f"📄 Extracting content from: {url}")
            content = self.web_search.extract_content(url)
            return content if content else "Could not extract content from this URL."
        
        def get_current_datetime() -> str:
            """Get the current date and time."""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert functions to LlamaIndex tools
        search_tool = FunctionTool.from_defaults(fn=search_web)
        webpage_tool = FunctionTool.from_defaults(fn=get_webpage_content)
        datetime_tool = FunctionTool.from_defaults(fn=get_current_datetime)
        
        tools = [search_tool, webpage_tool, datetime_tool]
        
        # Test if we can use ReActAgent
        if REACT_AGENT and hasattr(REACT_AGENT, 'from_tools'):
            try:
                print("🧪 Attempting to create ReActAgent...")
                agent = REACT_AGENT.from_tools(
                    tools,
                    llm=self.llm,
                    verbose=True,
                    max_iterations=10
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
        
        try:
            response = self.agent.chat(message)
            return str(response)
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    async def achat(self, message: str) -> str:
        """Async version of chat"""
        return self.chat(message)
