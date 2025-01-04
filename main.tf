provider "google" {
  project = "time-series-nn-forecast-lab"
  region  = "us-central1"
}

resource "google_storage_bucket" "experiment_results" {
  name     = "my-experiment-results"
  location = "US"
}

resource "google_compute_instance" "hyperparameter_vm" {
  name         = "hyperparameter-tuning-vm"
  machine_type = "e2-standard-4"
  zone         = "us-central1-a"

  boot_disk {
    initialize_params {
      image = "projects/debian-cloud/global/images/family/debian-11"
      size  = 50
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata_startup_script = <<-EOT
    #!/bin/bash
    apt-get update
    apt-get install -y docker.io
    gcloud auth configure-docker
    docker pull gcr.io/your-project-id/hyperparameter-tuning:latest
  EOT
}
