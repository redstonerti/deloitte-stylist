import boto3
import json
from botocore.config import Config
import logging
import os
import base64
from pathlib import Path

from dotenv import load_dotenv
from botocore.exceptions import ClientError
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_jpg_to_base64_str(path_string):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jpg_path = os.path.join(script_dir, path_string)
    with open(jpg_path, "rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
        encoded_str = encoded_bytes.decode('utf-8')
    return encoded_str

def read_json(path_string):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, path_string)
    with open(json_path, "r", encoding="utf-8") as f:
        json_file = json.load(f)
    return json_file

def write_json_to_file(json_object,path_string):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_string = os.path.join(script_dir, path_string)
    with open(path_string, "w", encoding="utf-8") as f:
        json.dump(json_object, f, indent=4)

def main():
    # Configure boto3 client with increased timeouts (60 minutes)
    config = Config(
        connect_timeout=60,
        read_timeout=3600,  # 60 minutes in seconds
        retries={'max_attempts': 1}
    )
    keys_file = read_json("keys.json")
    region = keys_file.get("AWS_DEFAULT_REGION")
    access_key = keys_file.get("AWS_ACCESS_KEY_ID")
    secret_key = keys_file.get("AWS_SECRET_ACCESS_KEY")
    session_token = keys_file.get("AWS_SESSION_TOKEN")

    client = boto3.client("bedrock-runtime", region_name=region,aws_access_key_id=access_key,aws_secret_access_key=secret_key, config=config,aws_session_token=session_token)

    # Prepare the request body according to the Nova API schema
    request_body = read_json("nova pro prompt.json")
    request_body['messages'][0]['content'][1]['image']['source']['bytes'] = read_jpg_to_base64_str('sneakers.jpeg')

    try:
        response = client.invoke_model(
            modelId="us.amazon.nova-pro-v1:0",  # inference profile ID for nova pro in us-west-2
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body).encode("utf-8")
        )

        # Read and parse the response body
        response_body = response['body'].read().decode("utf-8")
        response_json = json.loads(response_body)

        write_json_to_file(response_json,"nova response.json")
        #print(json.dumps(response_json, indent=2))

    except Exception as e:
        print(f"Error invoking model: {e}")

if __name__ == "__main__":
    main()

