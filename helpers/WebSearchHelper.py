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

class WebSearchHelper:
    """Simple web search and content extraction tool"""
    
    def __init__(self, search_api_key: Optional[str] = None):
        self.search_api_key = search_api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search_brave(self, query: str, max_results: int = 3) -> List[dict]:
        """Search using DuckDuckGo's instant answer API"""
        try:
            # Construct the search URL
            url = 'https://search.brave.com/search?q=' + query

            # Fetch the URL data using requests.get(url),
            # store it in a variable, request_result.
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            request_result=requests.get( url, headers=headers )

            # Creating soup from the fetched request
            soup = BeautifulSoup(request_result.text, "html.parser")

            # soup.find_all results divs
            # all major headings of our search result,
            heading_object=soup.find_all( 'div', class_='snippet' )

            # Iterate, parse, and generate structured results.
            results = []
            
            # dumb isintance checks to satisfy pylance type checking
            for info in heading_object:
                if isinstance(info, bs4.Tag):
                    infoclass = str(info.get('class'))
                    if "standalone" not in infoclass:
                        tag = info.find('a')
                        if isinstance(tag, bs4.Tag):
                            link = tag.get('href')
                        else:
                            link =""
                        
                        tag = info.find('div', class_='title')
                        if isinstance(tag, bs4.Tag):
                            title = tag.getText()
                        else:
                            title =""
                        
                        tag = info.find('div', class_='snippet-description')
                        if isinstance(tag, bs4.Tag):
                            description = tag.getText()
                        else:
                            description =""
                        
                        tag = info.find('div', class_='item-attributes')
                        if isinstance(tag, bs4.Tag):
                            attributes = tag.getText()
                        else:
                            attributes =""
                                            
                        results.append({
                            'title': title,
                            'content': description,
                            'url': link,
                            'attributes': attributes
                        })

                        if len(results) >= max_results:
                            break


            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
        
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
        
        
    def extract_webpage_content(self, url: str) -> str:
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
