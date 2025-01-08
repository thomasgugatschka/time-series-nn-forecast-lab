import os
import subprocess

def load_generated_files(dataset_dir):
    """Load the list of generated configuration files."""
    index_file = os.path.join(dataset_dir, "generated_files_index.txt")
    with open(index_file, "r") as f:
        return [line.strip() for line in f.readlines()]

def create_batch_job_yaml(config_path, location, template_path, output_path):
    """Create a batch job YAML file based on a template."""
    with open(template_path, "r") as template_file:
        template = template_file.read()

    # Replace placeholders with actual values
    job_yaml = template.replace("{{CONFIG_PATH}}", config_path).replace("{{LOCATION}}", location)

    # Write the YAML file
    with open(output_path, "w") as output_file:
        output_file.write(job_yaml)

    print(f"Batch job YAML created: {output_path}")

def submit_batch_job(yaml_path, location):
    """Submit a GCP Batch job."""
    cmd = [
        "gcloud", "batch", "jobs", "submit",
        f"--config={yaml_path}",
        f"--location={location}"
    ]
    print(f"Running command: {' '.join(cmd)}")  # Debugging log
    subprocess.run(cmd, check=True, shell=True)
    print(f"Submitted batch job: {yaml_path}")

def main():
    # Define paths and parameters
    datasets = ["ETTh"]  # List of datasets
    target_root = "../../configs/experiment_configs"  # Root directory for configurations
    batch_template_path = "batch_job_template.yaml"  # Path to the batch job YAML template
    batch_output_dir = "batch_jobs"  # Directory to store batch job YAML files

    #location = "europe-west4"
    #location = "europe-west3"
    #location = "us-central1"
    location = "us-east1"

    # Ensure the output directory exists
    os.makedirs(batch_output_dir, exist_ok=True)

    # Loop over each dataset
    for dataset_name in datasets:
        dataset_dir = os.path.join(target_root, dataset_name)
        config_files = load_generated_files(dataset_dir)

        # Loop over each configuration file
        for config_file in config_files:
            # Use the relative path directly without adding "experiment_configs" again
            config_path = os.path.join("experiment_configs", dataset_name, config_file)

            # Generate the batch job YAML
            batch_yaml_path = os.path.join(batch_output_dir, f"batch_{os.path.basename(config_file).replace('.yaml', '')}.yaml")
            create_batch_job_yaml(config_path.replace("\\", "/"), location, batch_template_path, batch_yaml_path)

            # Submit the batch job
            submit_batch_job(batch_yaml_path, location)

if __name__ == "__main__":
    main()
