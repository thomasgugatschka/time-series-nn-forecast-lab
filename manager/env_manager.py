from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Optional
from utilities.logging_utils import get_logger

logger = get_logger(__name__)

class EnvManager:
    """
    Singleton class for managing environment variables.
    Implements lazy loading of .env files to minimize overhead.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EnvManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.env_loaded = False
            self._initialized = True

    def load_env(self, env_file: str = ".env"):
        """
        Load environment variables from a specified .env file.
        This method will load the file only once per instance.

        Args:
            env_file (str): The path to the .env file to load.
        """
        if not self.env_loaded:
            # Ensure the .env file is resolved relative to the project root
            env_path = Path(__file__).resolve().parent.parent / env_file
            if env_path.exists():
                load_dotenv(env_path)
                logger.info(f"Loaded environment variables from {env_file}")
            else:
                raise FileNotFoundError(f"{env_file} not found.")
            self.env_loaded = True

    @staticmethod
    def get_env_variable(key: str, default: Optional[str] = None, as_bool: bool = False) -> str | bool:
        """
        Retrieve an environment variable, with optional boolean conversion.

        Args:
            key (str): The environment variable name.
            default (str, optional): Default value if the variable is not set.
            as_bool (bool, optional): Convert the value to a boolean if True.

        Returns:
            str | bool: The environment variable value, optionally as a boolean.
        """
        value = os.getenv(key, default)
        if as_bool:
            return str(value).lower() in ("true", "1", "yes")
        return value