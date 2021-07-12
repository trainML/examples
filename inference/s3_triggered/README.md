
Create [Systems Manager Parameter](https://aws.amazon.com/systems-manager/features/) SecureString parameters:

/trainml/api_user
/trainml/api_key

using default ssm KMS key.

```
    {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [

                "arn:aws:s3:::trainml-example-*",
                "arn:aws:s3:::trainml-example-*/*"
            ]
        },
        {
            "Sid": "VisualEditor1",
            "Effect": "Allow",
            "Action": "s3:PutObject",
            "Resource": [
                "arn:aws:s3:::trainml-example-*/*"
            ]
        }
    ]
}
```


```
./create-stack.sh us-east-1
```

```
./delete-stack.sh us-east-1
```

https://github.com/trainML/examples/raw/master/inference/s3_triggered/images.zip