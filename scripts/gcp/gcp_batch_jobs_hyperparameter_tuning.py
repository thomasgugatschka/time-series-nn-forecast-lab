import os
import subprocess


def load_generated_files(dataset_dir):
    """Load the list of generated configuration files."""
    index_file = os.path.join(dataset_dir, "generated_files_index.txt")
    with open(index_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def create_batch_job_yaml(config_path, location, template_path, output_path, memory_mib, cpu_milli):
    """Create a batch job YAML file based on a template."""
    with open(template_path, "r") as template_file:
        template = template_file.read()

    # Replace placeholders with actual values
    job_yaml = (
        template
        .replace("{{CONFIG_PATH}}", config_path)
        .replace("{{LOCATION}}", location)
        .replace("{{MEMORY_MIB}}", str(memory_mib))
        .replace("{{CPU_MILLI}}", str(cpu_milli))
    )

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


def get_resource_requirements(config_file):
    """Determine the resource requirements (memory and CPU) based on the config file name."""

    # Resource settings must match the config file in regard to the CPU.

    # TODO -> REFACTOR -> Have somewhere a MASTER Config which links the BATCH GENERATION with this FILE GENERATION
    resource_settings = {
        "tcn":          {"memory_mib": 131072, "cpu_milli": 32000},
        "transformer":  {"memory_mib": 131072, "cpu_milli": 32000},
    }

    for key, value in resource_settings.items():
        if key in config_file:
            return value
    return {"memory_mib": 65536, "cpu_milli": 32000}  # Default resource settings


def main():
    # Define paths and parameters
    datasets = ["weatherstation"]  # List of datasets
    target_root = "../../configs/experiment_configs"  # Root directory for configurations
    batch_template_path = "batch_job_template.yaml"  # Path to the batch job YAML template
    batch_output_dir = "batch_jobs"  # Directory to store batch job YAML files

    # Locations for job submission
    locations = [
        "europe-west4", "us-central1", "us-east1", "australia-southeast2",
        "us-east5", "us-west1", "me-west1", "europe-west3",
        "us-east4", "us-west4", "asia-east1", "europe-west9",
        "europe-southwest1", "us-south1", "asia-south1",
        "asia-south2", "us-west2", "us-west3", "me-central1"
    ]

    # Location iterator to loop through the list cyclically
    location_count = len(locations)
    location_index = 0

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

            # Get the current location, cycling through the list
            location = locations[location_index]
            location_index = (location_index + 1) % location_count  # Increment and wrap around

            # Determine resource requirements
            resources = get_resource_requirements(config_file)

            # Generate the batch job YAML
            batch_yaml_path = os.path.join(batch_output_dir,
                                           f"batch_{os.path.basename(config_file).replace('.yaml', '')}.yaml")
            create_batch_job_yaml(
                config_path.replace("\\", "/"), location, batch_template_path, batch_yaml_path,
                memory_mib=resources["memory_mib"], cpu_milli=resources["cpu_milli"]
            )

            # Submit the batch job
            submit_batch_job(batch_yaml_path, location)


if __name__ == "__main__":
    main()

# TODO -> REFACTOR -> Script should have input query for the dataset

# TODO -> FEATURE -> Add GENERIC interface to the DataManager to load files from Cloud Storage
#   Needed so TRAINING, EVALUATION and TUNING (resume) can properly done in the Cloud.

# TODO -> FEATURE -> IF this (above) FEATURE is DONE then the CLOUD BATCH JOB can be changed to SPOT provisioned
#   SPOT is a lot cheaper but can only be done if RESUMING is possible, and this is only possible if the DataManager is
#   able to load the files from the Cloud Storage for tuning (=restore the state of the experiment).
