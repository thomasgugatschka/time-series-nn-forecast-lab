# TODO change to europe
provider "google" {
  project = "time-series-nn-forecast-lab"
  region  = "us-central1"
  zone = "us-central1-a"
}

resource "google_storage_bucket" "experiment_results" {
  name     = "forecast-lab-experiments"
  location = "US"
}


# TODO how to set the compute engine?
# TODO how to add GPU ?
# TODO how to add docker image?

# resource "google_compute_instance" "hyperparameter_vm" {
#   name         = "hyperparameter-tuning-vm"
#   machine_type = "e2-standard-4"
#   zone         = "us-central1-a"
#
#   boot_disk {
#     initialize_params {
#       image = "projects/debian-cloud/global/images/family/debian-11"
#       size  = 50
#     }
#   }
#
#   network_interface {
#     network = "default"
#     access_config {}
#   }
#
#   metadata_startup_script = <<-EOT
#     #!/bin/bash
#     apt-get update
#     apt-get install -y docker.io
#     gcloud auth configure-docker
#     docker pull gcr.io/your-project-id/hyperparameter-tuning:latest
#   EOT
# }
