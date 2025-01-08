provider "google" {
  project = "time-series-nn-forecast-lab"
  region  = "us-central1"
  zone    = "us-central1-a"
}

# GCS Bucket for Experiment Results
resource "google_storage_bucket" "experiment_results" {
  name     = "forecast-lab-experiments"
  location = "US"
}

