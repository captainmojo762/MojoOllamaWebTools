import os
import asyncio
import re
from typing import List, Optional
from datetime import datetime

# Basic imports first
import requests
from bs4 import BeautifulSoup
import json

# LlamaIndex imports - let's test each one
print("Testing LlamaIndex imports...")

try:
    from llama_index.core import Settings
    print("✓ Settings imported successfully")
except ImportError as e:
    print(f"✗ Settings import failed: {e}")

try:
    from llama_index.core.tools import FunctionTool
    print("✓ FunctionTool imported successfully")
except ImportError as e:
    print(f"✗ FunctionTool import failed: {e}")

try:
    from llama_index.llms.ollama import Ollama
    print("✓ Ollama imported successfully")
except ImportError as e:
    print(f"✗ Ollama import failed: {e}")

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    print("✓ HuggingFaceEmbedding imported successfully")
except ImportError as e:
    print(f"✗ HuggingFaceEmbedding import failed: {e}")

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
        from llama_index.agent.react import ReActAgent
        REACT_AGENT = ReActAgent
        print("✓ ReActAgent imported from llama_index.agent.react")
        print(f"  Available methods: {[method for method in dir(ReActAgent) if not method.startswith('_')]}")
        print(f"  Has from_tools: {hasattr(ReActAgent, 'from_tools')}")
    except ImportError as e:
        print(f"✗ ReActAgent import from agent.react failed: {e}")
        REACT_IMPORT_ERROR = str(e)

if REACT_AGENT is None:
    print(f"✗ No ReActAgent available. Last error: {REACT_IMPORT_ERROR}")
    print("Will use SimpleToolAgent fallback")


class WebSearchTool:
    """Simple web search and content extraction tool"""
    
    def __init__(self, search_api_key: Optional[str] = None):
        self.search_api_key = search_api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search_duckduckgo(self, query: str, max_results: int = 3) -> List[dict]:
        """Search using DuckDuckGo's instant answer API"""
        try:
            # DuckDuckGo instant answer API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            
            results = []
            
            # Add abstract if available
            if data.get('Abstract'):
                results.append({
                    'title': data.get('AbstractText', 'Search Result'),
                    'content': data.get('Abstract'),
                    'url': data.get('AbstractURL', '')
                })
            
            # Add related topics
            for topic in data.get('RelatedTopics', [])[:max_results-len(results)]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'title': topic.get('Text', '').split(' - ')[0],
                        'content': topic.get('Text', ''),
                        'url': topic.get('FirstURL', '')
                    })
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def extract_content(self, url: str) -> str:
        """Extract text content from a webpage"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()
            
            # Extract text from main content areas
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.post', '.entry-content', 'p'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = "\n".join([elem.get_text().strip() for elem in elements[:3]])
                    break
            
            if not content:
                content = soup.get_text()
            
            # Clean up the text
            lines = [line.strip() for line in content.split('\n')]
            content = '\n'.join([line for line in lines if line and len(line) > 10])
            
            return content[:2000]  # Limit content length
            
        except Exception as e:
            print(f"Content extraction error for {url}: {e}")
            return ""


class SimpleToolAgent:
    """Simple agent that can use tools with keyword detection"""
    
    def __init__(self, tools: List[FunctionTool], llm):
        self.tools = tools
        self.llm = llm
        self.tool_map = {tool.metadata.name: tool for tool in tools}
    
    def chat(self, message: str) -> str:
        """Chat with automatic tool detection and calling"""
        
        # Check if the message might need web search
        search_indicators = [
            "current", "latest", "recent", "news", "weather", "stock", "price",
            "today", "now", "2024", "2025", "what's happening", "developments",
            "breaking", "update", "status"
        ]
        
        # Check if query needs current time/date
        time_indicators = ["time", "date", "when", "what time", "current time"]
        
        message_lower = message.lower()
        needs_search = any(indicator in message_lower for indicator in search_indicators)
        needs_time = any(indicator in message_lower for indicator in time_indicators)
        
        context_info = ""
        
        # Get current time if needed
        if needs_time and "get_current_datetime" in self.tool_map:
            current_time = self.tool_map["get_current_datetime"].fn()
            context_info += f"\nCurrent date and time: {current_time}\n"
        
        # Search web if needed
        if needs_search and "search_web" in self.tool_map:
            print("🔍 Detected need for web search...")
            print(f"🔍 DEBUG: Calling search_web tool with message: '{message}'")
            search_result = self.tool_map["search_web"].fn(message)
            print(f"🔍 DEBUG: Search result length: {len(search_result)} characters")
            print(f"🔍 DEBUG: Search result preview: {search_result[:200]}...")
            context_info += f"\nWeb search results:\n{search_result}\n"
        
        # Create enhanced query for LLM
        if context_info:
            enhanced_query = f"""
User question: {message}

Additional context to help answer the question:
{context_info}

Please provide a comprehensive answer using this information.
"""
        else:
            enhanced_query = message
        
        # Get response from LLM
        response = self.llm.complete(enhanced_query)
        return str(response)


class OllamaWebAssistant:
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
        self.web_search = WebSearchTool(search_api_key)
        
        # Create the agent
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create an agent with web search capabilities"""
        print("🔧 Creating agent...")
        
        def search_web(query: str) -> str:
            """Search the web for current information about a given topic."""
            print(f"🔍 Searching web for: {query}")
            results = self.web_search.search_duckduckgo(query, max_results=3)
            
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
                print("  Falling back to SimpleToolAgent")
        
        # Use SimpleToolAgent as fallback
        print("🤖 Creating SimpleToolAgent")
        return SimpleToolAgent(tools, self.llm)
    
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


def main():
    """Example usage of the Ollama Web Assistant"""
    
    print("=" * 60)
    print("🚀 OLLAMA WEB ASSISTANT DIAGNOSTIC VERSION")
    print("=" * 60)
    
    try:
        # Initialize the assistant
        print("\n🔧 Initializing Ollama Web Assistant...")
        assistant = OllamaWebAssistant(
            model_name="dolphin-llama3",  # Change to your preferred Ollama model
            ollama_base_url="http://localhost:11434"
        )
        
        print("\n✅ Assistant ready! Type 'quit' to exit.\n")
        
        # Interactive chat loop
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                response = assistant.chat(user_input)
                print(f"\nAssistant: {response}\n")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Chat error: {e}")
                
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


# Example usage in a script:
"""
# Initialize the assistant
assistant = OllamaWebAssistant(model_name="dolphin-llama3")

# Ask questions that might require web search
responses = [
    assistant.chat("What's the current weather in New York?"),
    assistant.chat("Tell me about the latest developments in AI"),
    assistant.chat("What is the capital of France?"),  # This won't need web search
    assistant.chat("What are the recent news about Tesla stock?")
]

for response in responses:
    print(f"Response: {response}\n")
"""