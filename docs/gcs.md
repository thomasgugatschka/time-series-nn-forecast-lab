create Google Cloud Account

in IAM & Admin create service account and set role as "Storage Admin" and "Storage Object Admin"

create a key for the service account and download the json file

store json in project root with the name service-account-key.json 

enable Artifact Registry API
    -> to allow docker push to gcr.io

gcloud auth configure-docker
    -> to authenticate docker to push to gcr.io

--> then push to gcr.io should be possible
    docker push gcr.io/time-series-nn-forecast-lab/hyperparameter-tuner:latest

enable Batch API

You are about to enable:

    Batch API
    Compute Engine API
    Cloud Logging API