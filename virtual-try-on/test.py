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
    image_data = base64.b64decode(base64_str)
    with open("output_image.jpeg", "wb") as f:
        f.write(image_data)

def read_json(path_string):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, path_string)
    with open(json_path, "r", encoding="utf-8") as f:
        json_file = json.load(f)
    return json_file

def main():
    keys_file = read_json("keys.json")
    region = keys_file.get("AWS_DEFAULT_REGION")
    access_key = keys_file.get("AWS_ACCESS_KEY_ID")
    secret_key = keys_file.get("AWS_SECRET_ACCESS_KEY")
    session_token = keys_file.get("AWS_SESSION_TOKEN")

    bedrock_client = boto3.client(service_name="bedrock", region_name=region,aws_access_key_id=access_key,
    aws_secret_access_key=secret_key, aws_session_token=session_token)
    bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region,aws_access_key_id=access_key,
    aws_secret_access_key=secret_key, aws_session_token=session_token)


    if(True):
        #list_model_ids(bedrock_client)
        #prompt_ai(bedrock_runtime,'amazon.nova-pro-v1:0',"nova pro prompt.json")
        prompt_ai(bedrock_runtime,'amazon.titan-image-generator-v2:0',"prompt titan.json")
        #prompt_ai(bedrock_runtime,'stability.stable-diffusion-xl-inpainting-v1:0','prompt stable diffusion 1.0.json')
    else:
        #print(get_foundation_model(bedrock_client,model_id))
        #list_model_ids(bedrock_client)

        response = bedrock_client.list_foundation_models()

        matching_models = []

        for model in response['modelSummaries']:
            inputs = model.get('inputModalities', [])
            outputs = model.get('outputModalities', [])
            inferences = model.get('inferenceTypesSupported', [])
            if (
                'IMAGE' in inputs and
                'TEXT' in outputs and
                'ON_DEMAND' in inferences
            ):
                print({
                    "modelId": model['modelId'],
                    "provider": model['providerName']
                })
def prompt_ai(bedrock_runtime,model_id,prompt_path):
    prompt = read_json(prompt_path)  # this loads the JSON content into a Python dict
    prompt['inPaintingParams']['image'] = read_jpg_to_base64_str('user.jpg')
    body_bytes = json.dumps(prompt).encode("utf-8")

    response = bedrock_runtime.invoke_model(
        body=body_bytes,
        contentType='application/json',
        accept='application/json',
        modelId=model_id,
        performanceConfigLatency='standard'
    )

    output = json.loads(response['body'].read())
    write_json_to_file(output,"response.json")

    base64_str = output['images'][0]
    write_base_64_image(base64_str)

    print("Response written to response.json")
def write_json_to_file(json_object,file_path):
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

if __name__ == "__main__":
    main()

