## Installation

Follow these steps to set up the `time-series-nn-forecast-lab` project.

**TODO: Confirm all installation steps with FRESH System**

---

### Prerequisites

Ensure you have the following installed:

- **Python 3.12**: The project is compatible with Python 3.12. Install it from [python.org](https://www.python.org/).
- **Git LFS** *(optional)*: Required for handling large files such as datasets, if project is maintained on GitHub. [Install Git LFS](https://git-lfs.com/).
- **Docker** *(optional)*: For running the project in a containerized environment. [Get Docker](https://www.docker.com/).
- **Google Cloud SDK** *(optional)*: Required for deploying experiments and syncing configurations to Google Cloud. [Install the SDK](https://cloud.google.com/sdk/docs/install).
- **AWS CLI** *(optional)*: Required for deploying experiments and syncing configurations to Amazon Web Services. [Install AWS CLI](https://aws.amazon.com/cli/).

---

### Steps

1. **Clone the Repository**  
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/<your-repo>/time-series-nn-forecast-lab.git
   cd time-series-nn-forecast-lab
    ```
   
2. **Set Up Git LFS**  
   If not already installed, set up Git LFS for handling large files:
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Create a Virtual Environment** *(optional but recommended)*  
   Set up a virtual environment to isolate dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install Required Packages**  
   Install all Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set Up Environment Variables**  
   Configure environment variables by creating a `.env` file in the root directory:
   ```plaintext
   PROJECT_BASE_DIR=/path/to/time-series-nn-forecast-lab
   USE_GCS=False
   ```
   > Refer to `.env.example` for a list of all configurable variables.

6. **Prepare Data and Configurations**  
   - Place your raw datasets in the `data/` directory.  
   - Define your configurations in `configs/` (e.g., `hyperparameter_search_space.json`, `dataset_configs/`).

7. **Run Tests** *(optional)*  
   Ensure everything is working as expected:
   ```bash
   pytest
   ```

---

### Docker Setup *(optional)*

To run the project in a containerized environment:

1. **Build the Docker Image**  
   ```bash
   docker build -t time-series-nn-forecast-lab .
   ```

2. **Run the Docker Container**  
   ```bash
   docker run -it --rm -v $(pwd):/app time-series-nn-forecast-lab
   ```

---

### Google Cloud Setup *(optional)*

For GCP integration, ensure you have the following:

- A valid `service-account-key.json` placed in the root directory.
- Enable the required APIs (e.g., Cloud Storage) for your GCP project.
- Authenticate using the Google Cloud SDK:
   ```bash
   gcloud auth activate-service-account --key-file=service-account-key.json
   gcloud config set project <your-gcp-project-id>
   ```

---

### AWS Setup *(optional)*

For AWS integration, ensure the following:

- Your AWS CLI is configured with the necessary credentials:
   ```bash
   aws configure
   ```
- Define S3 buckets or other required services in your scripts as needed.

---

