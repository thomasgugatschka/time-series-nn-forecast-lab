
# Time Series Neural Network Forecast Lab

This project is focused on implementing and experimenting with various neural network models for time series forecasting. It is part of my master's thesis.

# TODO: ADAPT TO NEW STRUCTURE AND SETUP!!!!

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

# TODO FIX STRUCTURE :)

    ├── time-series-nn-forecast-lab/           # Main project directory
        ├── data/                                  # Contains all raw datasets
        ├── configs/                               # Configuration files for different datasets/models
        ├── models/                                # Different neural network model implementations
        ├── utilities/                             # Utility functions to make the pipeline generic and reusable
        ├── scripts/                               # Scripts that will actually run the training, evaluation, etc.
        ├── docs/                                  # Documentation for the project
        ├── trained_models/                        # Directory for saving trained models
        ├── results/                               # Results from training, evaluation, and experiments
        ├── docker/                                # Docker-related files for containerizing the application
        ├── notebooks/                             # Jupyter Notebooks for exploration and experimentation
        ├── experiments/                           # Saved configurations and results for various hyperparameter tuning runs
        ├── logs/                                  # Log files to track training, errors, etc.
        ├── requirements.txt                       # Dependencies and packages required for running the project
        ├── setup.py                               # Installation script if this project is to be packaged
        ├── README.md                              # High-level overview and instructions for the project
        └── .gitignore                             # Files and directories to be ignored by Git


## Installation

1. Clone the repository:
    \```
    git clone https://github.com/Thomas-G93/time-series-nn-forecast-lab.git
    cd time-series-nn-forecast-lab
    \```

2. Create and activate a virtual environment:
    \```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    \```

3. Install the required packages:
    \```
    pip install -r requirements.txt
    \```

## Usage

### Configuration

Configuration files for different datasets and models are located in the `configs/` directory. Modify these files to set your desired parameters.

### Training

To train a model, run the `train.py` script with the appropriate configuration file:
\```
python scripts/train.py --config configs/etth1_config.yaml
\```

### Evaluation

To evaluate a trained model, run the `evaluate.py` script:
\```
python scripts/evaluate.py --model_path trained_models/gru_trained_model_1.pt --config configs/etth1_config.yaml
\```

### Prediction

To make predictions using a trained model, run the `predict.py` script:
\```
python scripts/predict.py --model_path trained_models/gru_trained_model_1.pt --config configs/etth1_config.yaml
\```

## Results

Results from training, evaluation, and experiments are stored in the `results/` directory. This includes plots, evaluation metrics, and experiment summaries.

## Unit Tests
To run all unit tests, execute the following command:
\```
python -m unittest discover -s tests
\```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Python Version

This project uses Python 3.12.