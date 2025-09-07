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

class WebScrapeAgent:
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