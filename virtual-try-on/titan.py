import boto3
import json
from botocore.config import Config

from utils import *

def titan_function(titan_prompt):
    keys_file = read_json("keys.json")
    region = keys_file.get("AWS_DEFAULT_REGION")
    access_key = keys_file.get("AWS_ACCESS_KEY_ID")
    secret_key = keys_file.get("AWS_SECRET_ACCESS_KEY")
    session_token = keys_file.get("AWS_SESSION_TOKEN")

    #bedrock_client = boto3.client(service_name="bedrock", region_name=region,aws_access_key_id=access_key,
    #aws_secret_access_key=secret_key, aws_session_token=session_token)
    bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region,aws_access_key_id=access_key,
    aws_secret_access_key=secret_key, aws_session_token=session_token)

    inpaint(bedrock_runtime,'amazon.titan-image-generator-v2:0',"prompt titan.json",titan_prompt)

def inpaint(bedrock_runtime,model_id,prompt_path,titan_prompt):
    prompt = read_json(prompt_path)
    prompt['inPaintingParams']['image'] = read_jpg_to_base64_str('user.jpg')
    prompt_text = prompt['inPaintingParams']['text']
    prompt_text = prompt_text + "\n\n" + titan_prompt
    prompt['inPaintingParams']['text'] = prompt_text
    print(prompt_text)
    body_bytes = json.dumps(prompt).encode("utf-8")

    response = bedrock_runtime.invoke_model(
        body=body_bytes,
        contentType='application/json',
        accept='application/json',
        modelId=model_id,
        performanceConfigLatency='standard'
    )

    output = json.loads(response['body'].read())
    write_json_to_file(output,"titan response.json")

    base64_str = output['images'][0]
    write_base_64_image(base64_str)

    print("Response written to response.json")
