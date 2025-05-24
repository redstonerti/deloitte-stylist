import logging
import json
import os
import base64

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
        #print(json.dumps(model, indent=2))
def main():
    # env
    load_dotenv(dotenv_path='keys.env')
    region = os.getenv("AWS_DEFAULT_REGION")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")
    bedrock_client = boto3.client(service_name="bedrock", region_name=region,aws_access_key_id=access_key,
aws_secret_access_key=secret_key, aws_session_token=session_token)
    bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region,aws_access_key_id=access_key,
aws_secret_access_key=secret_key, aws_session_token=session_token)
    #print(jpg_to_base64_str('india.png'))
    if(False):
        #prompt_ai(bedrock_runtime,'amazon.titan-image-generator-v2:0',"titan prompt.json")
        prompt_ai(bedrock_runtime,'stability.stable-image-ultra-v1:1','prompt stable diffusion.json')
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

        print(matching_models)

def prompt_ai(bedrock_runtime,model_id,prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        payload = json.load(f)  # this loads the JSON content into a Python dict
        payload['image'] = jpg_to_base64_str('india.png')
        #print(payload)
    body_bytes = json.dumps(payload).encode("utf-8")
    response = bedrock_runtime.invoke_model(
        body=body_bytes,
        contentType='application/json',
        accept='application/json',
        modelId=model_id,
        performanceConfigLatency='standard'
    )
    output = json.loads(response['body'].read())
    base64_str = output['images'][0]
    image_data = base64.b64decode(base64_str)
    with open("output_image.jpeg", "wb") as f:
        f.write(image_data)
    with open("response.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    print("Response written to response.json")
def jpg_to_base64_str(filepath):
    with open(filepath, "rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
        encoded_str = encoded_bytes.decode('utf-8')
    return encoded_str

if __name__ == "__main__":
    main()