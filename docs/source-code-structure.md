# Source Code Structure

    src/
    ├── callbacks/                              # Callback functions to manage training events and checkpoints
    │   ├── base_callback.py                    # Base class for implementing callbacks
    │   ├── logging_callback.py                 # Logs training progress at all stages of training process
    │   ├── early_stopping_callback.py          # Stops training when a metric plateaus
    │   ├── model_checkpoint_callback.py        # Saves model checkpoints during training
    │   ├── plotting_callback.py                # Generates plots during training
    │   └── tensorboard_callback.py             # Logs metrics and graphs to TensorBoard
    
    ├── data_pipeline/                          # Data preprocessing and feature engineering pipeline
    │   ├── preprocessors/                      # Preprocessing specific to model architectures
    │   │   ├── base_data_pipeline.py           # Abstract class for preprocessing pipelines
    │   │   ├── rnn_data_pipeline.py            # Preprocessing for RNN-based models (Generic Pipeline)
    │   │   └── transformer_data_pipeline.py    # Preprocessing for Transformer-based models
    │   ├── ingestion/                          # Data ingestion logic
    │   │   └── raw_data_loader.py              # Loads raw data from various formats
    │   ├── preparation/                        # Prepares data for models
    │   │   ├── data_splitter.py                # Splits data into train/val/test
    │   │   └── data_loader_creator.py          # Creates PyTorch data loaders
    │   ├── utils/                              # Utility functions for data preprocessing
    │   │   ├── data_pipeline_factory.py        # Factory to create pipelines dynamically
    │   │   └── data_transformation.py          # Data transformation utilities
    
    ├── manager/                            # High-level managers for facades and orchestration
    │   ├── callback_manager.py             # Coordinates the execution of multiple callbacks
    │   ├── config_manager.py               # Handles configuration file parsing and validation
    │   ├── data_manager.py                 # Manages data loading and preprocessing
    │   ├── env_manager.py                  # Manages environment file configurations
    │   ├── experiment_manager.py           # Manages experiment lifecycle and metadata
    │   ├── file_manager.py                 # Singleton for file I/O operations
    │   ├── gc_manager.py                   # Google Cloud Storage manager
    │   ├── hyperparameter_manager.py       # Manages hyperparameter search spaces
    │   ├── path_manager.py                 # Singleton for managing project paths
    │   └── plot_manager.py                 # Handles plot creation and visualization

    ├── models/                             # Neural network models and their wrappers
    │   ├── bilstm/                         # Bidirectional Long Short-Term Memory
    │   │   ├── bilstm_model.py             # BiLSTM model definition
    │   │   └── bilstm_wrapper.py           # BiLSTM wrapper for training, validation and prediction
    │   ├── cnn/                            # Convolutional Neural Network
    │   │   ├── cnn_model.py                # CNN model definition
    │   │   └── cnn_wrapper.py              # CNN wrapper for training, validation and prediction
    │   ├── gru/                            # Gated Recurrent Unit
    │   │   ├── gru_model.py                # GRU model definition
    │   │   └── gru_wrapper.py              # GRU wrapper for training, validation and prediction
    │   ├── lstm/                           # Long Short-Term Memory
    │   │   ├── lstm_model.py               # LSTM model definition
    │   │   └── lstm_wrapper.py             # LSTM wrapper for training, validation and prediction
    │   ├── rnn/                            # Recurrent Neural Network
    │   │   ├── rnn_model.py                # RNN model definition
    │   │   └── rnn_wrapper.py              # RNN wrapper for training, validation and prediction
    │   ├── tcn/                            # Temporal Convolutional Network
    │   │   ├── tcn_model.py                # TCN model definition
    │   │   └── tcn_wrapper.py              # TCN wrapper for training, validation and prediction
    │   ├── transformer/                    # Transformer
    │   │   ├── transformer_model.py        # Transformer model definition
    │   │   └── transformer_wrapper.py      # Transformer wrapper for training, validation and prediction
    │   ├── base_model.py                   # Abstract base class for all models
    │   ├── base_wrapper.py                 # Wrapper for handling model training and evaluation
    │   ├── model_factory.py                # Factory for creating model instances
    │   ├── wrapper_factory.py              # Factory for creating wrapper instances
    │   └── model_utils.py                  # Helpers for model creation and initialization

    ├── scripts/                            # Main scripts to fulfill use cases
    │   ├── aggregate_results.py            # Aggregates results from multiple experiments 
    │   ├── analyze_tuning_results.py       # Analyzes hyperparameter tuning results
    │   ├── evaluate_models.py              # Evaluates trained models
    │   ├── train_models.py                 # Script for training models
    │   ├── hyperparameter_tuning.py        # Hyperparameter tuning using Ray Tune
    │   └── visualize_dataset.py            # Visualizes datasets for analysis

    ├── strategies/                         # Encapsulates strategies for schedulers and search algorithms
    │   ├── scheduler_strategy.py           # Scheduler strategies for Ray Tune
    │   ├── search_algorithm_strategy.py    # Search algorithms for hyperparameter tuning
    │   └── strategy_factory.py             # Factory for creating strategy instances
    
    ├── training_pipeline/                  # Training pipeline setup and execution
    │   ├── training_pipeline.py            # Core pipeline logic for training models
    │   └── training_pipeline_builder.py    # Builder for dynamically configuring training pipelines
    
    ├── utilities/                          # Utility functions and shared components
    │   ├── experiment_utils.py             # Experiment utilities (e.g., seed setting)
    │   ├── metrics_utils.py                # Computes metrics like MAE, RMSE, etc.
    │   ├── logging_utils.py                # Logging configuration and utilities
    │   └── model_utils.py                  # Helpers for model creation and initialization
    



