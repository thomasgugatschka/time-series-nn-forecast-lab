import os
import re
import boto3

def load_generated_files(dataset_dir):
    """Load the list of generated configuration files."""
    index_file = os.path.join(dataset_dir, "generated_files_index.txt")
    with open(index_file, "r") as f:
        return [line.strip() for line in f.readlines()]

def sanitize_job_name(config_path):
    """
    Sanitize the job name to comply with AWS Batch naming rules.

    :param config_path: Path of the configuration file.
    :return: A sanitized job name.
    """
    # Extract the config file name without extension
    config_name = config_path.split('/')[-1].replace('.yaml', '')
    # Replace invalid characters with underscores using regex
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', config_name)
    return f"hyperparameter_job_{sanitized_name}"


def submit_jobs(job_queue, job_definition, config_files, output_dir):
    """
    Submit AWS Batch jobs for a list of configuration files.

    :param job_queue: Name of the AWS Batch job queue.
    :param job_definition: Name of the AWS Batch job definition (with revision, e.g., fargate-job-definition:3).
    :param config_files: List of configuration file paths to pass as arguments.
    :param output_dir: Directory to store job submission logs.
    """
    # Initialize AWS Batch client
    batch_client = boto3.client("batch", region_name="eu-central-1")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for config_path in config_files:
        # Generate a sanitized job name
        job_name = sanitize_job_name(config_path)

        response = batch_client.submit_job(
            jobName=job_name,
            jobQueue=job_queue,
            jobDefinition=job_definition,
            containerOverrides={
                "command": [
                    "python",
                    "-m",
                    "src.scripts.hyperparameter_tuning",
                    "--config_path",
                    config_path
                ]
            }
        )

        # Save job submission details
        job_log_path = os.path.join(output_dir, f"{job_name}_log.txt")
        with open(job_log_path, "w") as log_file:
            log_file.write(f"Submitted job for config: {config_path}\n")
            log_file.write(f"Job Name: {response['jobName']}\n")
            log_file.write(f"Job ID: {response['jobId']}\n")

        print(f"Submitted job for config: {config_path}")
        print(f"Job Name: {response['jobName']}")
        print(f"Job ID: {response['jobId']}")

def main():
    # Define paths and parameters
    datasets = ["ETTh"]  # List of datasets
    target_root = "../../configs/experiment_configs"  # Root directory for configurations
    batch_output_dir = "batch_jobs"  # Directory to store batch job logs

    job_queue = "fargate-queue"  # Replace with your job queue name
    job_definition = "fargate-job-definition:1"  # Replace with your job definition name (with revision)

    # Loop over each dataset
    for dataset_name in datasets:
        dataset_dir = os.path.join(target_root, dataset_name)
        config_files = load_generated_files(dataset_dir)

        # Convert relative paths for Batch jobs
        config_files = [os.path.join("experiment_configs", dataset_name, config_file) for config_file in config_files]

        # Submit jobs for the config files
        submit_jobs(job_queue, job_definition, config_files, batch_output_dir)

if __name__ == "__main__":
    main()
