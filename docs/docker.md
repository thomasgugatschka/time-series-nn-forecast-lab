
**TODO**


# Build docker image
docker build -t time-series-nn-forecast-lab-image .


# for gcp
#       Build the Docker image
docker build -t gcr.io/time-series-nn-forecast-lab/hyperparameter-tuner:latest .
#       Push the image to Google Container Registry
docker push gcr.io/time-series-nn-forecast-lab/hyperparameter-tuner:latest



# list available images
docker images


# run docker container wiht default config
docker run -it --rm time-series-forecast-lab-image:latest python -m src.scripts.hyperparameter_tuning
#### or like that only if CMD in Dockerfile is properly defined
docker run -it --rm time-series-forecast-lab-image:latest

# run docker container with new config
docker run -it --rm time-series-nn-forecast-lab-image:latest python -m src.scripts.hyperparameter_tuning --config_path "experiment_configs/config_gru_ETTh_0.7-0.15.yaml"