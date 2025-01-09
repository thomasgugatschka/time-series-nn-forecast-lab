
import logging
import logging.config
import yaml
from dotenv import load_dotenv
import os
from pathlib import Path

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
