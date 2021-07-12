import re
import boto3
import os
import asyncio
from trainml import TrainML

ssm_client = boto3.client('ssm')
s3_client = boto3.client('s3')

trainml_user= ssm_client.get_parameter(
    Name=os.environ.get("TRAINML_USER_PATH"),
    WithDecryption=True
).get("Parameter").get("Value")
trainml_key= ssm_client.get_parameter(
    Name=os.environ.get("TRAINML_KEY_PATH"),
    WithDecryption=True
).get("Parameter").get("Value")

trainml_client = TrainML(user=trainml_user, key=trainml_key)

async def create_job(input_uri, output_uri):
    print(input_uri)
    print(output_uri)

def lambda_handler(event, context):
    print(event)
    for record in event["Records"]:
        input_uri = f"s3://{record['s3']['bucket']['name']}/{record['s3']['object']['key']}"
        output_key = "processed/" + re.sub(r"incoming\/", "", record["s3"]["object"]["key"])
        output_uri = f"s3://{record['s3']['bucket']['name']}/{output_key}"
        asyncio.run(create_job(input_uri,output_uri))