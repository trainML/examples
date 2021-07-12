import re
import boto3
import os
import asyncio
from trainml import TrainML

ssm_client = boto3.client("ssm")
s3_client = boto3.client("s3")

trainml_user = (
    ssm_client.get_parameter(
        Name=os.environ.get("TRAINML_USER_PATH"), WithDecryption=True
    )
    .get("Parameter")
    .get("Value")
)
trainml_key = (
    ssm_client.get_parameter(
        Name=os.environ.get("TRAINML_KEY_PATH"), WithDecryption=True
    )
    .get("Parameter")
    .get("Value")
)

trainml_client = TrainML(user=trainml_user, key=trainml_key)


async def create_job(name, input_uri, output_uri):
    print(input_uri)
    print(output_uri)

    job = await trainml_client.jobs.create(
        name=name,
        type="inference",
        gpu_type="RTX 2080 Ti",
        gpu_count=1,
        disk_size=1,
        workers=[
            "python inference/s3_triggered/trainml_model/predict.py",
        ],
        data=dict(
            input_type="aws",
            input_uri=input_uri,
            output_type="aws",
            output_uri=output_uri,
        ),
        model=dict(
            source_type="git",
            source_uri="https://github.com/trainML/examples.git",
        ),
    )
    return job


def lambda_handler(event, context):
    print(event)
    for record in event["Records"]:
        input_uri = f"s3://{record['s3']['bucket']['name']}/{record['s3']['object']['key']}"
        file_name = re.sub(r"^incoming\/", "", record["s3"]["object"]["key"])
        file_name = re.sub(r".zip$", "", file_name)
        output_uri = f"s3://{record['s3']['bucket']['name']}/processed"
        job = asyncio.run(create_job(file_name, input_uri, output_uri))

        ## Job information should be saved in a persistent datastore to pull for status and verify completion
        print(job.id)