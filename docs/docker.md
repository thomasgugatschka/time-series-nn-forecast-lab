# Docker Images

1. Start docker (e.g. Docker Desktop)

2. Build Container Image from Dockerfile
- **GCP**
```bash
docker build -t gcr.io/time-series-nn-forecast-lab/forecast-lab-container:latest .
```
This name must also be the same which is used in the `batch_job_template.yaml` file, otherwise the job will not be able to find the image.

- **Local**
```bash
docker build -t forecast-lab-container:latest .
```
GCP image can of course also be used locally.

- **AWS**
  - **TODO**

3. List available images
```bash
docker images
```

4. Run Container Image
- **Local with default `CMD` (script) defined in Dockerfile and default config.yaml**
```bash
docker run -it --rm forecast-lab-container:latest
```
- **Local with defined script and default config.yaml**
```bash
docker run -it --rm forecast-lab-container:latest python -m src.scripts.hyperparameter_tuning
```
- **Local with defined script and default config.yaml**
```bash
docker run -it --rm forecast-lab-container:latest python -m src.scripts.hyperparameter_tuning --config_path "experiment_configs/config_gru_ETTh_0.7-0.15.yaml"
```

5. Push Container Image to Container Registry (Cloud)
- **GCP**
```bash
docker push gcr.io/time-series-nn-forecast-lab/forecast-lab-container:latest
```
- **AWS**
  - **TODO**

6. Confirm functionality with simple Batch Job