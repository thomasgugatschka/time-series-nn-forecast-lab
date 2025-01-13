
# GCP Cloud

**TODO**



## Overview

## Setup


To set up GCP for batch processing:
1. **Terraform Configuration**: Automate resource provisioning using Terraform scripts.
2. **Service Account Key**: Provide a `service-account-key.json` file with appropriate permissions for accessing GCS and managing batch jobs.
3. **Permissions**:
   - Grant necessary IAM roles to the service account, such as:
     - `roles/storage.admin` (for GCS access)
     - `roles/batch.admin` (for managing Batch jobs)
   - Ensure the Compute Engine API and Batch API are enabled in your GCP project.
4. **GCS Bucket**: Create and configure a GCS bucket for storing experiment results and configurations.


---

---

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


--------------------


Create with Terraform the storage bucket
-> init & apply

Craete batch_job.yaml




Submit the batch job
    gcloud batch jobs submit --config=batch_job.yaml --location=us-central1



stream logs -> opne ssh to container through google cloud
-> docker ps
-> docker logs <container_id>
-> docker logs -f <container_id> # to stream logs




---
