import os
import asyncio
import re
from typing import List, Optional
from datetime import datetime

# Basic imports first
import requests
from bs4 import BeautifulSoup
import json

from helpers.OllamaIntegration import OllamaIntegration


"""This script is meant to give locally run Ollama models access to web search capabilities.
Using the llama-index framework, it provides web scraping and search tools to Ollama models, 
allowing them to provide more robust responses."""
def main():

    
    print("=" * 60)
    print("🚀 OLLAMA WEB ASSISTANT DIAGNOSTIC VERSION")
    print("=" * 60)
    
    try:
        # Initialize the assistant
        print("\n🔧 Initializing Ollama Web Assistant...")
        assistant = OllamaIntegration(
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
assistant = OllamaIntegration(model_name="dolphin-llama3")

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