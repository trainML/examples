
Create [Systems Manager Parameter](https://aws.amazon.com/systems-manager/features/) SecureString parameters:

/trainml/api_user
/trainml/api_key

using default ssm KMS key.


```
./create-stack.sh us-east-1
```

```
./delete-stack.sh us-east-1
```

https://github.com/trainML/examples/raw/master/inference/s3_triggered/images.zip