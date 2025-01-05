import yaml
import json
import pickle
import os
import pandas as pd
from pathlib import Path
from google.cloud import storage
from src.manager.env_manager import EnvManager

from src.utilities.logging_utils import get_logger
logger = get_logger(__name__)

# SINGELTON CLASS

#TODO add DEBUG level messages for each action


class FileManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FileManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):

        env_manager = EnvManager()
        env_manager.load_env()


    def load_yaml(self, file_path: str):
        """
        Load a YAML file and return its content.

        Args:
            file_path (str): Path to the YAML file to be loaded.

        Returns:
            dict: Parsed content of the YAML file.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"YAML file not found at {file_path}")

        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    def save_yaml(self, data: dict, file_path: str):
        """
        Save a dictionary to a YAML file.

        Args:
            data (dict): Data to be saved.
            file_path (str): Path to save the YAML file.
        """
        with open(file_path, 'w') as f:
            yaml.safe_dump(data, f)

    def load_json(self, file_path: str):
        """
        Load a JSON file and return its content.

        Args:
            file_path (str): Path to the JSON file to be loaded.

        Returns:
            dict: Parsed content of the JSON file.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found at {file_path}")

        with open(file_path, 'r') as f:
            return json.load(f)

    def save_json(self, data: dict, file_path: str):
        """
        Save a dictionary to a JSON file.

        Args:
            data (dict): Data to be saved.
            file_path (str): Path to save the JSON file.
        """
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.

        Args:
            file_path (str): Path to the CSV file to be loaded.

        Returns:
            pd.DataFrame: Loaded pandas DataFrame.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found at {file_path}")

        return pd.read_csv(file_path, encoding='utf-8')

    def save_csv(self, df: pd.DataFrame, file_path: str):
        """
        Save a pandas DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): DataFrame to save.
            file_path (str): Path to save the CSV file.
        """
        df.to_csv(file_path, index=False, encoding='utf-8')

    def load_pickle(self, file_path: str):
        """
        Load a pickle file and return its content.

        Args:
            file_path (str): Path to the pickle file to be loaded.

        Returns:
            object: Parsed content of the pickle file.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Pickle file not found at {file_path}")

        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def save_pickle(self, data, file_path: str):
        """
        Save an object to a pickle file.

        Args:
            data (object): Data to be saved.
            file_path (str): Path to save the pickle file.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def save_plot(self, plt_obj, file_path: str):
        """
        Save a matplotlib plot to a file.

        Args:
            plt_obj: Matplotlib plot object.
            file_path (str): Path to save the plot.
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        plt_obj.savefig(file_path)

    def create_directory(self, dir_path: str):
        """
        Create a directory if it does not exist.

        Args:
            dir_path (str): Path of the directory to create.
        """
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists at the given path.

        Args:
            file_path (str): Path to the file.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return Path(file_path).is_file()

    def delete_file(self, file_path: str):
        """
        Delete a file if it exists.

        Args:
            file_path (str): Path to the file to delete.
        """
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass


# Create the singleton instance
file_manager = FileManager()
