# File: utilities/logging_utils.py
import logging
import logging.config
import yaml

from manager.path_manager import path_manager

def setup_logging():
    #from manager.file_manager import file_manager

    # Load logging configuration from YAML file
    config_path = path_manager.get_path_config() / 'logging.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file) # NOT doen through file manager due circular dependency
        config['handlers']['file']['filename'] = str(path_manager.get_path_log() / 'app.log')
        logging.config.dictConfig(config)

def get_logger(module_name):
    return logging.getLogger(module_name)
