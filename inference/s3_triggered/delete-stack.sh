#!/bin/bash
DEPLOY_BUCKET=$(aws cloudformation describe-stack-resource --logical-resource-id DeploymentBucket --stack-name trainml-example-inference-s3-trigger | jq -r .StackResourceDetail.PhysicalResourceId)
DATA_BUCKET=$(aws cloudformation describe-stack-resource --logical-resource-id DataBucket --stack-name trainml-example-inference-s3-trigger | jq -r .StackResourceDetail.PhysicalResourceId)
aws s3 rm s3://${DEPLOY_BUCKET} --recursive
aws s3 rm s3://${DATA_BUCKET} --recursive
aws cloudformation delete-stack --stack-name trainml-example-inference-s3-trigger
aws cloudformation wait stack-delete-complete --stack-name trainml-example-inference-s3-trigger