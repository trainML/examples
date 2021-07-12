#!/bin/bash

if [[ ! $(which jq) ]]
then
    echo 'Please install jq'
    exit 1
fi

aws cloudformation create-stack --stack-name trainml-inference-s3-trigger-example --template-body file://create-deployment.yml --capabilities CAPABILITY_NAMED_IAM

rm -f lambda.zip
cd lambda 
pip install -r requirements.txt --target ./package
cd package
zip -qr ../../lambda.zip *
cd ..
zip -gq ../lambda.zip *.py
cd ..
aws cloudformation wait stack-create-complete --stack-name trainml-inference-s3-trigger-example
BUCKET=$(aws cloudformation describe-stack-resource --logical-resource-id DeploymentBucket --stack-name trainml-inference-s3-trigger-example | jq -r .StackResourceDetail.PhysicalResourceId)
aws s3 cp lambda.zip s3://${BUCKET}/lambda.zip
aws cloudformation update-stack --stack-name trainml-inference-s3-trigger-example --template-body file://update-deployment.yml --capabilities CAPABILITY_NAMED_IAM
aws cloudformation wait stack-update-complete --stack-name trainml-inference-s3-trigger-example