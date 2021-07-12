#!/bin/bash

DATA_BUCKET=$(aws cloudformation describe-stack-resource --logical-resource-id DataBucket --stack-name trainml-inference-s3-trigger-example | jq -r .StackResourceDetail.PhysicalResourceId)
aws s3 cp images.zip s3://$DATA_BUCKET/incoming/images-$(date -u +'%Y-%m-%d_%H-%M-%S').zip