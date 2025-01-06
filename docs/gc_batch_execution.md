
Create with Terraform the storage bucket
-> init & apply

Craete batch_job.yaml




Submit the batch job
    gcloud batch jobs submit --config=batch_job.yaml --region=us-central1



stream logs -> opne ssh to container through google cloud
-> docker ps
-> docker logs <container_id>
-> docker logs -f <container_id> # to stream logs

