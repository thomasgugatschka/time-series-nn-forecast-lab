
import logging
import logging.config
import yaml
from dotenv import load_dotenv
import os
from pathlib import Path

# TODO - CLEANUP

# TODO -> GOAL is that logger does not need any class or function from the project
# so no path_manager or file_manager

# TODO -> add method to disable coloured logging

def setup_logging():

    # Load environment variables from .env file
    load_dotenv()

    # Load logging configuration from YAML file
    base_dir = Path(os.getenv("PROJECT_BASE_DIR", Path(__file__).resolve().parent.parent)).resolve()
    config_path = base_dir / 'configs' / 'logging.yaml'


    with open(config_path, 'r') as file:
        config = yaml.safe_load(file) # NOT doen through file manager due circular dependency
        config['handlers']['file']['filename'] = str(base_dir / 'logs' / 'app.log')
        logging.config.dictConfig(config)

def get_logger(module_name):
    return logging.getLogger(module_name)
