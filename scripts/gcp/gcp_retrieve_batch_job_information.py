import subprocess
import json
import csv
import logging
import os

# Set up logging to print to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Set your project ID
PROJECT_ID = "time-series-nn-forecast-lab"

# Define which columns to include in the output
AVAILABLE_COLUMNS = {
    "Job ID": "job_id",
    "Location": "location",
    "State": "state",
    "Run Duration (s)": "run_duration",
    "Machine Type": "machine_type",
    "CPU (milli)": "cpu_milli",
    "Memory (MiB)": "memory_mib",
    "Docker Image": "docker_image",
    "Command": "command",
    "Environment Variables": "env_vars_str",
}

# USER-DEFINED SETTINGS
OUTPUT_DIR = "./batch_results"
SELECTED_COLUMNS = "Run Duration (s), Machine Type, CPU (milli), Memory (MiB), Command"


# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all jobs in the project
def get_batch_jobs():
    logging.info("Fetching batch jobs list...")
    try:
        command = f"gcloud batch jobs list --project={PROJECT_ID} --format=json"
        output = subprocess.check_output(command, shell=True, text=True)
        jobs = json.loads(output)
        logging.info(f"Retrieved {len(jobs)} jobs from Google Cloud Batch.")
        return jobs
    except subprocess.CalledProcessError as e:
        logging.error(f"Error fetching batch jobs: {e}")
        return []

# Extract job ID and location from full job name
def extract_job_details(job_name):
    try:
        parts = job_name.split("/")
        job_id = parts[-1]  # Last part is the job ID
        location = parts[3]  # Location is the fourth part
        return job_id, location
    except IndexError:
        logging.warning(f"Failed to extract location from job name: {job_name}")
        return None, None

# Get job details
def get_job_details(job_id, location):
    if not location:
        logging.warning(f"Skipping job {job_id} due to missing location.")
        return {}

    logging.info(f"Fetching details for job {job_id} in {location}...")
    try:
        command = f"gcloud batch jobs describe {job_id} --project={PROJECT_ID} --location={location} --format=json"
        output = subprocess.check_output(command, shell=True, text=True)
        logging.info(f"Successfully retrieved details for job {job_id}.")
        return json.loads(output)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error fetching details for job {job_id}: {e}")
        return {}

# Generate CSV report
def generate_report():
    logging.info("Starting batch jobs report generation...")
    jobs = get_batch_jobs()

    if not jobs:
        logging.info("No batch jobs found. Exiting.")
        return

    report_file = os.path.join(OUTPUT_DIR, "batch_jobs_report.csv")
    latex_file = os.path.join(OUTPUT_DIR, "batch_jobs_report.tex")

    selected_column_keys = [col.strip() for col in SELECTED_COLUMNS.split(",") if col.strip() in AVAILABLE_COLUMNS]

    job_data = []

    with open(report_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(selected_column_keys)  # Write header row

        for job in jobs:
            full_job_name = job.get("name", "N/A")
            job_id, location = extract_job_details(full_job_name)

            if not job_id or not location:
                logging.warning(f"Skipping job {full_job_name} due to missing location.")
                continue  # Skip jobs with missing location

            state = job.get("state", "UNKNOWN")

            # Fetch detailed info
            details = get_job_details(job_id, location)

            run_duration = details.get("status", {}).get("runDuration", "N/A").replace("s", "")
            machine_type = (
                details.get("taskGroups", [{}])[0]
                .get("instances", [{}])[0]
                .get("machineType", "N/A")
            )

            compute_resource = (
                details.get("taskGroups", [{}])[0]
                .get("taskSpec", {})
                .get("computeResource", {})
            )
            cpu_milli = compute_resource.get("cpuMilli", "N/A")
            memory_mib = compute_resource.get("memoryMib", "N/A")

            # Extract Docker Image and Command
            runnables = (
                details.get("taskGroups", [{}])[0]
                .get("taskSpec", {})
                .get("runnables", [{}])[0]
                .get("container", {})
            )
            docker_image = runnables.get("imageUri", "N/A")
            command = " ".join(runnables.get("commands", []))

            # Extract environment variables
            env_vars = (
                details.get("taskGroups", [{}])[0]
                .get("taskSpec", {})
                .get("environment", {})
                .get("variables", {})
            )
            env_vars_str = ", ".join([f"{k}={v}" for k, v in env_vars.items()])

            row_data = {
                "job_id": job_id,
                "location": location,
                "state": state,
                "run_duration": run_duration,
                "machine_type": machine_type,
                "cpu_milli": cpu_milli,
                "memory_mib": memory_mib,
                "docker_image": docker_image,
                "command": command,
                "env_vars_str": env_vars_str,
            }

            # Write selected columns to CSV
            writer.writerow([row_data[AVAILABLE_COLUMNS[col]] for col in selected_column_keys])
            job_data.append([row_data[AVAILABLE_COLUMNS[col]] for col in selected_column_keys])

            logging.info(f"Added job {job_id} from {location} to the report.")

    # Generate LaTeX table
    generate_latex_table(latex_file, selected_column_keys, job_data)

    logging.info(f"Batch jobs report completed. CSV saved to {report_file}. LaTeX table saved to {latex_file}.")

# Generate LaTeX table
def generate_latex_table(output_path, headers, data):
    with open(output_path, "w") as file:
        file.write("\\begin{table}[h]\n")
        file.write("\\centering\n")
        file.write("\\begin{tabular}{" + " | ".join(["c"] * len(headers)) + "}\n")
        file.write("\\hline\n")
        file.write(" & ".join(headers) + " \\\\\n")
        file.write("\\hline\n")

        for row in data:
            file.write(" & ".join(str(x) for x in row) + " \\\\\n")

        file.write("\\hline\n")
        file.write("\\end{tabular}\n")
        file.write("\\caption{Batch Jobs Report}\n")
        file.write("\\label{tab:batch_jobs_report}\n")
        file.write("\\end{table}\n")

# Run the report generation
if __name__ == "__main__":
    generate_report()
