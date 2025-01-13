**TODO**


Access key
AKIAZDZTBW75AKDVJLMT

Secret access key
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


PowerShell 7.4.6
PS C:\Users\Thomas> aws configure
AWS Access Key ID [None]: AKIAZDZTBW75AKDVJLMT
AWS Secret Access Key [None]: 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
Default region name [None]: eu-central-1
Default output format [None]: json
PS C:\Users\Thomas> aws s3 ls
PS C:\Users\Thomas> aws ecr create-repository --repository-name hyperparameter-tuner
{
    "repository": {
        "repositoryArn": "arn:aws:ecr:eu-central-1:626635421690:repository/hyperparameter-tuner",
        "registryId": "626635421690",
        "repositoryName": "hyperparameter-tuner",
        "repositoryUri": "626635421690.dkr.ecr.eu-central-1.amazonaws.com/hyperparameter-tuner",
        "createdAt": "2025-01-07T13:58:44.363000+01:00",
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": {
            "scanOnPush": false
        },
        "encryptionConfiguration": {
            "encryptionType": "AES256"
        }
    }
}

PS C:\Users\Thomas>



docker tag aws.io/hyperparameter-tuner:latest 626635421690.dkr.ecr.eu-central-1.amazonaws.com/hyperparameter-tuner:latest

docker push 626635421690.dkr.ecr.eu-central-1.amazonaws.com/hyperparameter-tuner:latest




aws batch register-job-definition \
    --job-definition-name fargate-job-definition \
    --type container \
    --platform-capabilities FARGATE \
    --container-properties '{
        "image": "626635421690.dkr.ecr.eu-central-1.amazonaws.com/hyperparameter-tuner:latest",
        "vcpus": 10,
        "memory": 8192,
        "command": ["python", "-m", "src.scripts.hyperparameter_tuning", "--config_path", "experiment_configs/ETTh/config_rnn_ETTh_0.1-0.1-0.1.yaml"],
        "environment": [
            {"name": "GOOGLE_APPLICATION_CREDENTIALS", "value": "/app/service-account-key.json"},
            {"name": "SYNC_CONFIG_FROM_GCS", "value": "True"}
        ]
    }' \
    --execution-role-arn "arn:aws:iam::626635421690:role/AWSBatchExecutionRole"




aws batch update-compute-environment \
    --compute-environment fargate-compute-env \
    --compute-resources "type=FARGATE,maxvCpus=256,minvCpus=0,desiredvCpus=128" \
    --region us-east-1



---------------
---------------------------------
----------------



PowerShell 7.4.6
PS C:\Users\Thomas> aws configure
AWS Access Key ID [None]: AKIAZDZTBW75AKDVJLMT
AWS Secret Access Key [None]: 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
Default region name [None]: eu-central-1
Default output format [None]: json
PS C:\Users\Thomas> aws s3 ls
PS C:\Users\Thomas> aws ecr create-repository --repository-name hyperparameter-tuner
{
    "repository": {
        "repositoryArn": "arn:aws:ecr:eu-central-1:626635421690:repository/hyperparameter-tuner",
        "registryId": "626635421690",
        "repositoryName": "hyperparameter-tuner",
        "repositoryUri": "626635421690.dkr.ecr.eu-central-1.amazonaws.com/hyperparameter-tuner",
        "createdAt": "2025-01-07T13:58:44.363000+01:00",
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": {
            "scanOnPush": false
        },
        "encryptionConfiguration": {
            "encryptionType": "AES256"
        }
    }
}

PS C:\Users\Thomas> aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 626635421690.dkr.ecr.eu-central-1.amazonaws.com/hyperparameter-tuner-1.amazonaws.com
Login Succeeded
PS C:\Users\Thomas> docker tag hyperparameter-tuner:latest 626635421690.dkr.ecr.eu-central-1.amazonaws.com/hyperparameter-tuner:latest
Error response from daemon: No such image: hyperparameter-tuner:latest
PS C:\Users\Thomas> docker tag aws.io/hyperparameter-tuner:latest 626635421690.dkr.ecr.eu-central-1.amazonaws.com/hyperparameter-tuner:latest
PS C:\Users\Thomas> docker push 626635421690.dkr.ecr.eu-central-1.amazonaws.com/hyperparameter-tuner:latest:latest
invalid reference format
PS C:\Users\Thomas> docker push 626635421690.dkr.ecr.eu-central-1.amazonaws.com/hyperparameter-tuner:latest
The push refers to repository [626635421690.dkr.ecr.eu-central-1.amazonaws.com/hyperparameter-tuner]
f8cd1cdf3fc0: Pushed
2653005e0531: Pushed
9c145d29440b: Pushed
afefa9ef4935: Pushed
b4f91ff4d97a: Pushed
bf690d497a2a: Pushed
5b514e64649f: Pushed
d45cccea3813: Pushed
b775cc45b4ba: Pushed
0ccb284b60a3: Pushed
8b296f486960: Pushed
latest: digest: sha256:9d69b83d108da18d11ced417ac20b494bc162c2ca0b0e67d622335263b6d75dd size: 2631
PS C:\Users\Thomas>
PS C:\Users\Thomas>
PS C:\Users\Thomas> aws ecr describe-images --repository-name aws.io/hyperparameter-tuner

An error occurred (RepositoryNotFoundException) when calling the DescribeImages operation: The repository with name 'aws.io/hyperparameter-tuner' does not exist in the registry with id '626635421690'
PS C:\Users\Thomas> aws ecr describe-images --repository-name hyperparameter-tuner
{
    "imageDetails": [
        {
            "registryId": "626635421690",
            "repositoryName": "hyperparameter-tuner",
            "imageDigest": "sha256:9d69b83d108da18d11ced417ac20b494bc162c2ca0b0e67d622335263b6d75dd",
            "imageTags": [
                "latest"
            ],
            "imageSizeInBytes": 3939604093,
            "imagePushedAt": "2025-01-07T15:02:10.799000+01:00",
            "imageManifestMediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "artifactMediaType": "application/vnd.docker.container.image.v1+json"
        }
    ]
}

PS C:\Users\Thomas> aws batch create-compute-environment \