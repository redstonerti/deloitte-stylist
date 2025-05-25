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
    def __init__(self, api_key=None):
        """
        Initialize the ClothingEditor with Google AI API key.
        
        Args:
            api_key (str): Google AI API key. If None, will try to get from environment variable.
        """
        google_keys = read_json("google keys.json")
        api_key = google_keys['API KEY']
        
        if not api_key:
            raise ValueError("API key is required. Set GOOGLE_AI_API_KEY environment variable or pass it directly.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-preview-image-generation')
    
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
            PIL.Image: Generated image with modified clothing
        """
        try:
            # Load the input image
            input_image = self.load_image(image_path)
            
            # Create the prompt
            prompt = self.create_clothing_prompt(original_clothing, new_clothing, additional_instructions)
            
            print("Sending request to Gemini API...")
            print(f"Prompt: {prompt[:100]}...")
            
            # Generate the modified image
            response = self.model.generate_content([prompt, input_image])
            
            # Check if the response contains an image
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
            
            raise Exception("No image was generated in the response")
            
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
        print("Please set your Google AI API key as an environment variable:")
        print("export GOOGLE_AI_API_KEY='your-api-key-here'")
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
        
        # Save the result
        editor.save_image(result_image, output_image_path)
        
        print("Clothing modification completed successfully!")
        
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
        editor.save_image(result, output_path)
        print(f"Success! Modified image saved to {output_path}")
    except Exception as e:
        print(f"Error: {e}")
    """
    
    # Run the main example
    main()