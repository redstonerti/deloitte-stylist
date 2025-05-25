import requests
import json
import os
from typing import List, Dict, Optional
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS


class AzureGPT4MiniClient:
    def __init__(self, endpoint: str, api_key: str, deployment_name: str = "gpt-4o-mini"):
        """
        Initialize Azure OpenAI client

        Args:
            endpoint: Your Azure OpenAI endpoint (e.g., "https://your-resource.openai.azure.com/")
            api_key: Your Azure OpenAI API key
            deployment_name: Name of your GPT-4 mini deployment
        """
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = "2023-12-01-preview"  # Updated API version

        # Construct the full URL
        self.url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions"

        # Set up headers
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }

    def chat_completion(self,
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       top_p: float = 1.0,
                       stream: bool = False) -> Dict:
        """
        Send a chat completion request to Azure OpenAI

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Controls randomness (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            top_p: Controls diversity via nucleus sampling
            stream: Whether to stream the response

        Returns:
            Response dictionary from Azure OpenAI
        """

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream
        }
        
        params = {
            "api-version": self.api_version
        }
        
        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                params=params,
                timeout=30
            )
            
            # Raise an exception for bad status codes
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response content: {response.text}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Response content: {response.text}")
            raise

# Configuration
API_KEY = "bc70w1vQSgfOTeeFK8PruLKvsKHgJPLZJ8kESM16IMl6HM6avt4DJQQJ99BEACHYHv6XJ3w3AAAAACOGMCJJ"
AZURE_ENDPOINT = "https://p3240-mb1zlkmw-eastus2.services.ai.azure.com"
DEPLOYMENT_NAME = "gpt-4o-mini"
SERPAPI_KEY = "639176bdb21285c3fb637611ff60248168f08ab3a17886a8a56ebc8511726d73"

# Initialize the Azure OpenAI client globally
azure_client = AzureGPT4MiniClient(
    endpoint=AZURE_ENDPOINT,
    api_key=API_KEY,
    deployment_name=DEPLOYMENT_NAME
)

def call_azure_openai(style, mood, occasion, season, gender):
    """
    Use Azure OpenAI to generate fashion search queries based on user preferences
    """
    
    prompt = f"""
    Create specific search queries for fashion items based on these preferences:
    - Style: {style}
    - Mood: {mood}
    - Occasion: {occasion}
    - Season: {season}
    - Gender: {gender}
    
    Generate search queries for these clothing categories: top, bottom, shoes, accessories.
    Return the response as a JSON object with keys: "top", "bottom", "shoes", "accessories".
    Each value should be a specific search query string optimized for fashion e-commerce sites.
    
    Example format:
    {{
        "top": "women casual cotton t-shirt summer",
        "bottom": "men denim jeans relaxed fit",
        "shoes": "women white sneakers comfortable",
        "accessories": "unisex leather watch minimalist"
    }}
    """
    
    messages = [
        {"role": "system", "content": "You are a fashion stylist expert. Generate specific, searchable fashion queries."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = azure_client.chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
        
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            # Try to parse JSON from the response
            try:
                # Extract JSON from the response (in case there's extra text)
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    # Fallback if JSON parsing fails
                    return {
                        "top": f"{gender} {style} {mood} top {season}",
                        "bottom": f"{gender} {style} {mood} bottom {season}",
                        "shoes": f"{gender} {style} {mood} shoes {season}",
                        "accessories": f"{gender} {style} {mood} accessories {season}"
                    }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "top": f"{gender} {style} {mood} top {season}",
                    "bottom": f"{gender} {style} {mood} bottom {season}",
                    "shoes": f"{gender} {style} {mood} shoes {season}",
                    "accessories": f"{gender} {style} {mood} accessories {season}"
                }
        else:
            # Fallback response
            return {
                "top": f"{gender} {style} {mood} top {season}",
                "bottom": f"{gender} {style} {mood} bottom {season}",
                "shoes": f"{gender} {style} {mood} shoes {season}",
                "accessories": f"{gender} {style} {mood} accessories {season}"
            }
            
    except Exception as e:
        print(f"Error in call_azure_openai: {e}")
        # Return fallback queries
        return {
            "top": f"{gender} {style} {mood} top {season}",
            "bottom": f"{gender} {style} {mood} bottom {season}",
            "shoes": f"{gender} {style} {mood} shoes {season}",
            "accessories": f"{gender} {style} {mood} accessories {season}"
        }

def search_fashion_item(query):
    """
    Search for fashion items using SerpAPI Google Shopping with enhanced image and URL handling
    """
    try:
        url = "https://serpapi.com/search"
        params = {
            "engine": "google_shopping",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": 10,  # Get more results to have better options
            "gl": "us",  # Country code
            "hl": "en"   # Language
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        print(f"SerpAPI Shopping response for '{query}':")
        print(f"Available keys: {list(data.keys())}")
        
        if "shopping_results" in data:
            print(f"Found {len(data['shopping_results'])} shopping results")
            for i, item in enumerate(data["shopping_results"][:5]):
                print(f"Item {i+1} keys: {list(item.keys())}")
                
                # Debug: Print the actual item structure
                print(f"Item {i+1} data: {json.dumps(item, indent=2)}")
                
                # Extract all possible image URLs
                image_url = (
                    item.get("thumbnail") or 
                    item.get("image") or 
                    item.get("product_image") or
                    ""
                )
                
                # Extract product URL - try all possible fields
                product_url = (
                    item.get("link") or 
                    item.get("product_link") or 
                    item.get("url") or
                    item.get("product_url") or
                    item.get("merchant", {}).get("link") if isinstance(item.get("merchant"), dict) else None or
                    "#"
                )
                
                # If still no URL, try to construct from product_id and source
                if product_url == "#" and item.get("product_id"):
                    source = item.get("source", "").lower()
                    if "amazon" in source:
                        product_url = f"https://amazon.com/dp/{item.get('product_id')}"
                    elif "ebay" in source:
                        product_url = f"https://ebay.com/itm/{item.get('product_id')}"
                
                print(f"Extracted URL: {product_url}")
                print(f"Extracted Image: {image_url}")
                
                # Clean and format price
                price = item.get("price", "Price not available")
                if isinstance(price, str) and not price.startswith("$") and price != "Price not available":
                    price = f"${price}"
                
                result = {
                    "title": item.get("title", "Fashion Item")[:100] + ("..." if len(item.get("title", "")) > 100 else ""),
                    "price": price,
                    "link": product_url,
                    "image": image_url,
                    "thumbnail": image_url,  # Keep both for compatibility
                    "source": item.get("source", "Unknown Store"),
                    "rating": item.get("rating", ""),
                    "reviews": item.get("reviews", ""),
                    "delivery": item.get("delivery", ""),
                    "position": item.get("position", ""),
                    "product_id": item.get("product_id", "")
                }
                print("--------------------------------------",result["link"],"--------------------------------------")
                results.append(result)
                
        else:
            print("No 'shopping_results' key found in response")
            print(f"Response keys: {list(data.keys())}")
        
        # If no shopping results, try Google Images with different approach
        if not results:
            print("No shopping results found, trying Google Images...")
            image_results = search_fashion_images_with_urls(query)
            results.extend(image_results)
        
        return results[:5] if results else [get_fallback_result(query)]
        
    except requests.exceptions.RequestException as e:
        print(f"Network error in search_fashion_item: {e}")
        return [get_fallback_result(query)]
    except Exception as e:
        print(f"Error in search_fashion_item: {e}")
        return [get_fallback_result(query)]

def search_fashion_images_with_urls(query):
    """
    Search for fashion images using SerpAPI Google Images and try to extract shopping URLs
    """
    try:
        url = "https://serpapi.com/search"
        params = {
            "engine": "google_images",
            "q": query + " buy online store",
            "api_key": SERPAPI_KEY,
            "num": 5,
            "safe": "active"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        if "images_results" in data:
            for item in data["images_results"][:3]:
                # Try to get a shopping URL from the image source
                source_url = item.get("link", "#")
                
                # Check if the source looks like a shopping site
                shopping_indicators = ["shop", "buy", "store", "amazon", "ebay", "etsy", "walmart", "target"]
                is_shopping_site = any(indicator in source_url.lower() for indicator in shopping_indicators)
                
                result = {
                    "title": item.get("title", query.title()),
                    "price": "Check site for price",
                    "link": source_url if is_shopping_site else "#",
                    "image": item.get("original", item.get("thumbnail", "")),
                    "thumbnail": item.get("thumbnail", ""),
                    "source": item.get("source", "Image Search"),
                    "rating": "",
                    "reviews": "",
                    "delivery": "",
                    "position": ""
                }
                results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Error in search_fashion_images_with_urls: {e}")
        return []

def get_fallback_result(query):
    """
    Return a fallback result when search fails
    """
    return {
        "title": f"No results found for '{query}'",
        "price": "",
        "link": "#",
        "image": "https://via.placeholder.com/300x300?text=No+Image",
        "thumbnail": "https://via.placeholder.com/150x150?text=No+Image",
        "source": "Search unavailable",
        "rating": "",
        "reviews": "",
        "delivery": "",
        "position": ""
    }

def main():
    """
    Test function for the Azure GPT-4 Mini client
    """
    # Example conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    try:
        # Make the API call
        response = azure_client.chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        
        # Extract and print the response
        if 'choices' in response and len(response['choices']) > 0:
            assistant_message = response['choices'][0]['message']['content']
            print("Assistant:", assistant_message)
        else:
            print("No response received")
            
    except Exception as e:
        print(f"Error occurred: {e}")

# Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

print("TEMPLATE FOLDER:", os.path.abspath(app.template_folder))

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Original HTML form route
    """
    outfit = {}
    if request.method == 'POST':
        style = request.form.get('style')
        mood = request.form.get('mood')
        occasion = request.form.get('occasion')
        season = request.form.get('season')
        gender = request.form.get('gender')

        # Use the new Azure OpenAI function
        queries = call_azure_openai(style, mood, occasion, season, gender)

        # Search for each fashion item
        for item, query in queries.items():
            results = search_fashion_item(query)
            outfit[item] = results[0] if results else None

    return render_template('main.html', outfit=outfit)

@app.route('/api/outfit', methods=['POST'])
def get_outfit_json():
    """
    JSON API endpoint for outfit recommendations
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        style = data.get('style')
        mood = data.get('mood')
        occasion = data.get('occasion')
        season = data.get('season')
        gender = data.get('gender')

        # Validate required fields
        if not all([style, mood, occasion, season, gender]):
            return jsonify({'error': 'All fields are required'}), 400

        print(f"Received preferences: {data}")

        # Use the existing Azure OpenAI function
        queries = call_azure_openai(style, mood, occasion, season, gender)
        print(f"Generated queries: {queries}")

        # Search for each fashion item
        outfit = {}
        for item, query in queries.items():
            print(f"Searching for {item} with query: {query}")
            results = search_fashion_item(query)
            outfit[item] = results[0] if results else None
            print(f"Found {len(results)} results for {item}")

        print(f"Final outfit: {outfit}")
        return jsonify(outfit)
        
    except Exception as e:
        print(f"Error in get_outfit_json: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    app.run(debug=True)