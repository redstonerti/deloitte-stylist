import boto3
import json
from botocore.config import Config

from utils import *

def describe_image(base_64_string,description):
    config = Config(
        connect_timeout=60,
        read_timeout=3600,
        retries={'max_attempts': 1}
    )
    keys_file = read_json("keys.json")
    region = keys_file.get("AWS_DEFAULT_REGION")
    access_key = keys_file.get("AWS_ACCESS_KEY_ID")
    secret_key = keys_file.get("AWS_SECRET_ACCESS_KEY")
    session_token = keys_file.get("AWS_SESSION_TOKEN")

    client = boto3.client("bedrock-runtime", region_name=region,aws_access_key_id=access_key,aws_secret_access_key=secret_key, config=config,aws_session_token=session_token)

    request_body = read_json("nova prompt.json")
    prompt_text = request_body['messages'][0]['content'][0]['text']
    prompt_text = prompt_text + description
    request_body['messages'][0]['content'][0]['text'] = prompt_text
    request_body['messages'][0]['content'][1]['image']['source']['bytes'] = base_64_string

    try:
        response = client.invoke_model(
            modelId="us.amazon.nova-pro-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body).encode("utf-8")
        )

        response_body = response['body'].read().decode("utf-8")
        response_json = json.loads(response_body)

        write_json_to_file(response_json,"nova response.json")
        return response_json

    except Exception as e:
        print(f"Error invoking model: {e}")

