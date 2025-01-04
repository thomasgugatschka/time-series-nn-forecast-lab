import numpy as np
import pandas as pd
from src.manager.path_manager import path_manager
from src.manager.file_manager import file_manager

# TODO -> add this to some other folder... since just supporting but not "daily" operation !!!!


def generate_multivariate_sinusoidal_data(num_samples=1000, num_features=5, stop=20):
    t = np.linspace(0, stop, num_samples)
    data = np.zeros((num_samples, num_features))
    for i in range(num_features):
        frequency = 1 + i * 0.1  # Vary frequency
        phase = i * np.pi / 4  # Vary phase
        amplitude = 10 + i * 5  # Increase amplitude
        data[:, i] = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return data, t

def save_data_to_csv(data, filename):
    feature_names = [f'feature_{i + 1}' for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=feature_names)
    file_manager.save_csv(df, filename)


if __name__ == "__main__":
    num_samples = 19_000
    num_features = 5
    stop = 100

    # Generate data
    data, t = generate_multivariate_sinusoidal_data(num_samples, num_features, stop)

    # Define output paths using PathManager
    sinusoidal_dir = path_manager.get_path_data() / 'sinusoidal'
    file_manager.create_directory(sinusoidal_dir)

    csv_filename = sinusoidal_dir / f'multivariate_sinusoidal_ns{num_samples}_nf{num_features}_st{stop}.csv'

    # Save data to CSV
    save_data_to_csv(data, csv_filename)

