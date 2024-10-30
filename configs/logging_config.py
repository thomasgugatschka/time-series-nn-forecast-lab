import logging
import logging.config
from configs.paths import get_path
import colorlog
import inspect
import os

use_common_log_file = True      # Set to True to use a common log file for all scripts
                                # Set to False to use a log file with the same name as the calling script

include_timestamp_and_name = False  # Set to True to include timestamp and logger name in log messages
                                    # Set to False to exclude them

# Logging configuration template
LOGGING_CONFIG_TEMPLATE = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s' if include_timestamp_and_name else '%(levelname)s - %(message)s',
        },
        'colored': {
            '()': 'colorlog.ColoredFormatter',
            'format': '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s' if include_timestamp_and_name else '%(log_color)s%(levelname)s - %(message)s',
            'log_colors': {
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            },
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'colored',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'level': 'INFO',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
}

def setup_logging():
    if use_common_log_file:
        log_file_name = 'app.log'
    else:
        # Automatically detect the calling script's file name
        caller_frame = inspect.stack()[1]
        caller_file = os.path.basename(caller_frame.filename)
        log_file_name = f"{os.path.splitext(caller_file)[0]}.log"

    log_file_path = get_path('log_dir') / log_file_name
    logging_config = LOGGING_CONFIG_TEMPLATE.copy()
    logging_config['handlers']['file']['filename'] = log_file_path
    logging.config.dictConfig(logging_config)