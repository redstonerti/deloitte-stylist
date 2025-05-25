#!/usr/bin/env python3
"""
Gemini 2.0 Flash Image Generation - Clothing Editor

This script uses Google's Gemini API to modify clothing in images based on text descriptions.
It takes an input image and a description of desired clothing changes, then generates
a new image with the modified clothing.

Requirements:
- google-generativeai library
- PIL (Pillow) for image handling
- Valid Google AI API key

Usage:
    python clothing_editor.py
"""

import google.generativeai as genai
from PIL import Image
import os
import base64
import io
from utils import *


class ClothingEditor:
    def __init__(self, api_key=None, model_name=None):
        """
        Initialize the ClothingEditor with Google AI API key.
        
        Args:
            api_key (str): Google AI API key. If None, will try to get from JSON file.
            model_name (str): Model name to use. If None, will try to find a suitable model.
        """
        if api_key is None:
            try:
                google_keys = read_json("google keys.json")
                api_key = google_keys['API KEY']
            except Exception as e:
                print(f"Error reading API key from JSON: {e}")
                api_key = os.getenv('GOOGLE_AI_API_KEY')
        
        if not api_key:
            raise ValueError("API key is required. Set GOOGLE_AI_API_KEY environment variable or add it to 'google keys.json'.")
        
        genai.configure(api_key=api_key)
        
        # List available models and find a suitable one
        if model_name is None:
            model_name = self._find_suitable_model()
        
        print(f"Using model: {model_name}")
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
    
    def _find_suitable_model(self):
        """
        Find a suitable model for image generation from available models.
        
        Returns:
            str: Model name that supports image generation
        """
        try:
            models = genai.list_models()
            
            # Priority order of models to try
            preferred_models = [
                'gemini-2.0-flash-preview-image-generation',
                'gemini-2.0-flash-exp-image-generation',
                'gemini-2.0-flash',
                'gemini-1.5-pro',
                'gemini-1.5-flash'
            ]
            
            available_models = []
            print("Available models:")
            for model in models:
                model_name = model.name.replace('models/', '')
                available_models.append(model_name)
                print(f"  - {model_name}")
                
                # Check if this model supports generateContent
                if hasattr(model, 'supported_generation_methods'):
                    if 'generateContent' in model.supported_generation_methods:
                        print(f"    Supports generateContent: Yes")
                    else:
                        print(f"    Supports generateContent: No")
            
            # Try to find a preferred model that's available
            for preferred in preferred_models:
                if preferred in available_models:
                    print(f"Found preferred model: {preferred}")
                    return preferred
            
            # If no preferred model found, use the first available one that supports generateContent
            for model in models:
                model_name = model.name.replace('models/', '')
                if hasattr(model, 'supported_generation_methods'):
                    if 'generateContent' in model.supported_generation_methods:
                        print(f"Using first available model with generateContent: {model_name}")
                        return model_name
            
            # Fallback to gemini-1.5-flash (most likely to work)
            print("Using fallback model: gemini-1.5-flash")
            return 'gemini-1.5-flash'
            
        except Exception as e:
            print(f"Error listing models: {e}")
            print("Falling back to gemini-1.5-flash")
            return 'gemini-1.5-flash'
    
    def load_image(self, image_path):
        """
        Load and prepare image for API call.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            PIL.Image: Loaded image object
        """
        try:
            image = Image.open(image_path)
            # Convert to RGB if necessary (removes alpha channel, handles different formats)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")
    
    def create_clothing_prompt(self, original_clothing, new_clothing, additional_instructions=""):
        """
        Create a detailed prompt for clothing modification.
        
        Args:
            original_clothing (str): Description of current clothing
            new_clothing (str): Description of desired clothing
            additional_instructions (str): Any additional instructions
            
        Returns:
            str: Formatted prompt for the API
        """
        prompt = f"""Please modify this image by changing the person's clothing. 

Current clothing: {original_clothing}
Change to: {new_clothing}

Instructions:
- Keep the person's pose, face, and body position exactly the same
- Only change the clothing as specified
- Maintain realistic lighting and shadows on the new clothing
- Ensure the new clothing fits naturally on the person's body
- Keep the background and all other elements unchanged
- Make the clothing change look natural and realistic

{additional_instructions}

Generate the modified image with these clothing changes."""
        
        return prompt
    
    def edit_clothing(self, image_path, original_clothing, new_clothing, additional_instructions=""):
        """
        Edit clothing in the image based on the provided descriptions.
        
        Args:
            image_path (str): Path to the input image
            original_clothing (str): Description of current clothing (e.g., "yellow shirt")
            new_clothing (str): Description of desired clothing (e.g., "white shirt")
            additional_instructions (str): Any additional instructions
            
        Returns:
            PIL.Image: Generated image with modified clothing, or None if no image generated
        """
        try:
            # Load the input image
            input_image = self.load_image(image_path)
            
            # Create the prompt - adjust based on model capabilities
            if 'image-generation' in self.model_name or 'imagen' in self.model_name.lower():
                # For image generation models, use image editing prompt
                prompt = self.create_clothing_prompt(original_clothing, new_clothing, additional_instructions)
            else:
                # For regular models, ask for a description that could be used for image generation
                prompt = f"""Looking at this image, I want to modify the person's clothing. 
                
Current clothing: {original_clothing}
Desired clothing: {new_clothing}

Please provide a detailed description of how this image should be modified to change the clothing as specified. Focus on:
- Exact clothing replacement needed
- How to maintain the person's pose and appearance
- Lighting and shadow adjustments needed
- Color and texture details for the new clothing

{additional_instructions}"""
            
            print("Sending request to Gemini API...")
            print(f"Using model: {self.model_name}")
            print(f"Prompt: {prompt[:100]}...")
            
            # Generate the response
            response = self.model.generate_content([prompt, input_image])
            
            # Check if the response contains an image (for image generation models)
            if hasattr(response, 'parts') and response.parts:
                for part in response.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        # Decode the base64 image data
                        image_data = base64.b64decode(part.inline_data.data)
                        generated_image = Image.open(io.BytesIO(image_data))
                        return generated_image
            
            # If no image in parts, check if there's a direct image response
            if hasattr(response, 'image'):
                return response.image
            
            # If no image was generated, print the text response
            if hasattr(response, 'text'):
                print("No image generated. Model response:")
                print(response.text)
                print("\nNote: The current model may not support image generation.")
                print("The response above describes how the image should be modified.")
                return None
            
            raise Exception("No image was generated and no text response received")
            
        except Exception as e:
            raise Exception(f"Error generating image: {str(e)}")
    
    def save_image(self, image, output_path):
        """
        Save the generated image to a file.
        
        Args:
            image (PIL.Image): Image to save
            output_path (str): Path where to save the image
        """
        try:
            image.save(output_path, 'JPEG', quality=95)
            print(f"Image saved to: {output_path}")
        except Exception as e:
            raise Exception(f"Error saving image: {str(e)}")


def main():
    """
    Main function demonstrating how to use the ClothingEditor.
    """
    # Initialize the editor (make sure to set your API key)
    try:
        editor = ClothingEditor()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please add your Google AI API key to 'google keys.json' or set GOOGLE_AI_API_KEY environment variable")
        return
    
    # Example usage
    try:
        # Configure these paths and descriptions for your use case
        input_image_path = "yellow_shirt.jpg"  # Path to your input image
        output_image_path = "output_image.jpg"  # Where to save the result
        
        # Describe the current and desired clothing
        current_clothing = "yellow shirt"
        new_clothing = "white button-up shirt"
        
        # Optional additional instructions
        additional_instructions = "Make sure the shirt looks professional and well-fitted"
        
        print("Starting clothing modification...")
        print(f"Input image: {input_image_path}")
        print(f"Changing '{current_clothing}' to '{new_clothing}'")
        
        # Generate the modified image
        result_image = editor.edit_clothing(
            image_path=input_image_path,
            original_clothing=current_clothing,
            new_clothing=new_clothing,
            additional_instructions=additional_instructions
        )
        
        # Save the result if an image was generated
        if result_image is not None:
            editor.save_image(result_image, output_image_path)
            print("Clothing modification completed successfully!")
        else:
            print("No image was generated. Check the model response above.")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Example of how to use the script directly
    print("Gemini Clothing Editor")
    print("=====================")
    
    # You can also use it interactively:
    # Uncomment the following lines for interactive mode
    
    """
    input_path = input("Enter path to input image: ")
    current_clothes = input("Describe current clothing: ")
    new_clothes = input("Describe desired clothing: ")
    output_path = input("Enter output path (or press Enter for 'modified_image.jpg'): ") or "modified_image.jpg"
    
    try:
        editor = ClothingEditor()
        result = editor.edit_clothing(input_path, current_clothes, new_clothes)
        if result is not None:
            editor.save_image(result, output_path)
            print(f"Success! Modified image saved to {output_path}")
        else:
            print("No image was generated.")
    except Exception as e:
        print(f"Error: {e}")
    """
    
    # Run the main example
    main()