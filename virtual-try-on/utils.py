import logging
import json
import os
import base64
from pathlib import Path

import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_foundation_model(client, model_identifier):
    try:
        return client.get_foundation_model(
            modelIdentifier=model_identifier
        )["modelDetails"]
    except ClientError:
        logger.error(
            f"Couldn't get foundation models details for {model_identifier}"
        )
        raise

def list_foundation_models(client):
    try:
        response = client.list_foundation_models()
        models = response["modelSummaries"]
        logger.info("Got %s foundation models.", len(models))
        return models
    except ClientError:
        logger.error("Couldn't list foundation models.")
        raise

def list_model_ids(bedrock_client):
    fm_models = list_foundation_models(bedrock_client)
    for model in fm_models:
        print(f"Model: {model['modelName']}, id: {model['modelId']}")

def read_jpg_to_base64_str(path_string):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jpg_path = os.path.join(script_dir, path_string)
    with open(jpg_path, "rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
        encoded_str = encoded_bytes.decode('utf-8')
    return encoded_str

def write_base_64_image(base64_str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir,"output_image.jpeg")
    image_data = base64.b64decode(base64_str)
    with open(image_path, "wb") as f:
        f.write(image_data)

def read_json(path_string):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, path_string)
    with open(json_path, "r", encoding="utf-8") as f:
        json_file = json.load(f)
    return json_file

def write_json_to_file(json_object,file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir,file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(json_object, f, indent=4)

def list_inference_profiles(bedrock_client):
    try:
        # Paginate through inference profiles if there are many
        paginator = bedrock_client.get_paginator('list_inference_profiles')
        for page in paginator.paginate():
            for profile in page.get('inferenceProfileSummaries', []):
                print(f"Inference Profile ID: {profile['inferenceProfileId']}")
                for model in profile.get('models', []):
                    print(f"  Model ARN: {model.get('modelArn')}")
                print('-' * 40)
    except Exception as e:
        print(f"Error listing inference profiles: {e}")