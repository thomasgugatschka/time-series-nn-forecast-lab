import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from configs.paths import get_path

def generate_multivariate_sinusoidal_data(num_samples=1000, num_features=5, stop=20):
    t = np.linspace(0, stop, num_samples)
    data = np.zeros((num_samples, num_features))
    for i in range(num_features):
        frequency = 1 + i * 0.1  # Vary frequency
        phase = i * np.pi / 4    # Vary phase
        amplitude = 10 + i * 5   # Increase amplitude
        data[:, i] = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return data, t

def save_data_to_csv(data, filename):
    feature_names = [f'feature_{i+1}' for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=feature_names)
    df.to_csv(filename, index=False)

def plot_data(data, t, plot_filename):
    feature_names = [f'feature_{i+1}' for i in range(data.shape[1])]
    plt.figure(figsize=(10, 6))
    for i in range(data.shape[1]):
        plt.plot(t, data[:, i], label=feature_names[i])
    plt.title('Multivariate Sinusoidal Data')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig(plot_filename)
    plt.close()

if __name__ == "__main__":
    num_samples = 100000
    num_features = 5
    stop = 25
    data, t = generate_multivariate_sinusoidal_data(num_samples, num_features, stop)
    output_path = os.path.join(get_path('data_dir', 'sinus'), f'multivariate_sinusoidal_ns{num_samples}_nf{num_features}_st{stop}.csv')
    save_data_to_csv(data, output_path)
    plot_filename = os.path.join(get_path('data_dir', 'sinus'), f'multivariate_sinusoidal_ns{num_samples}_nf{num_features}_st{stop}.png')
    plot_data(data, t, plot_filename)