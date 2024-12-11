
import os
import pandas as pd
from utilities.logging_utils import get_logger
from manager.file_manager import file_manager

logger = get_logger(__name__)


class RawDataLoader:
    """
    Handles loading raw data from files and performing basic operations like column drops.
    """

    def __init__(self):
        # TODO should this be kept if id not really have anything to initialize???
        pass

    # REFACTOR: Currently only works with CSV files. Add support for other file types.
    def load_data(self, data_dir, sources, columns_to_drop=None):
        """
        Load data from specified sources and perform initial preprocessing like dropping columns.

        Args:
            sources (list): List of source file names without extension.
            data_dir (str): Directory containing the data files.
            columns_to_drop (list, optional): Columns to drop after loading data. Defaults to None.

        Returns:
            pd.DataFrame: Combined dataframe containing data from all sources.
        """
        df_list = []
        for source in sources:
            target_file = os.path.join(data_dir, f"{source}.csv")
            logger.info(f"Loading data from {target_file}")

            # Use the file_manager to load data
            df = file_manager.load_csv(target_file)

            logger.info(f"Data shape before dropping columns: {df.shape}")
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
                logger.info(f"Data shape after dropping columns: {df.shape}")
            df_list.append(df)
            logger.info(f"Appended data from {source}.csv")

        return pd.concat(df_list, axis=1)

# Singleton instance
raw_data_loader = RawDataLoader()
