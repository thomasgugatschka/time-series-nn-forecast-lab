from google.cloud import storage
from pathlib import Path
import os
from src.utilities.logging_utils import get_logger

logger = get_logger(__name__)

class GCSManager:
    def __init__(self, bucket_name: str):
        """
        Initialize the GCSManager.

        Args:
            bucket_name (str): Name of the GCS bucket.
        """
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    # IMPROVEMENT -> parallel uploads with asyncio and gsutil...

    def upload_file(self, local_path: str, gcs_path: str):
        """
        Upload a local file to GCS.

        Args:
            local_path (str): Path to the local file.
            gcs_path (str): Destination path in GCS.
        """
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        logger.info(f"Uploaded {local_path} to gs://{self.bucket_name}/{gcs_path}")

    def file_needs_upload(self, local_path: Path, gcs_path: str) -> bool:
        """
        Determine if a local file needs to be uploaded to GCS.

        Args:
            local_path (Path): Path to the local file.
            gcs_path (str): Path to the file in GCS.

        Returns:
            bool: True if the file needs to be uploaded, False otherwise.
        """
        blob = self.bucket.get_blob(gcs_path)  # Fetch blob metadata from GCS

        # If the file doesn't exist in GCS, upload it
        if not blob:
            logger.debug(f"File {gcs_path} does not exist in GCS. Needs upload.")
            return True

        # Compare file size
        local_file_size = local_path.stat().st_size
        if blob.size != local_file_size:
            logger.debug(f"Size mismatch for {gcs_path}. Needs upload.")
            return True

        # Optional: Compare MD5 checksum
        local_md5 = self._calculate_md5(local_path)
        if blob.md5_hash and blob.md5_hash != local_md5:
            logger.debug(f"MD5 mismatch for {gcs_path}. Needs upload.")
            return True

        logger.debug(f"File {gcs_path} is unchanged. Skipping upload.")
        return False

    @staticmethod
    def _calculate_md5(file_path: Path) -> str:
        """
        Calculate the MD5 checksum of a file.

        Args:
            file_path (Path): Path to the file.

        Returns:
            str: MD5 checksum of the file in base64 format.
        """
        import hashlib
        import base64

        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return base64.b64encode(hash_md5.digest()).decode("utf-8")

    def download_file(self, gcs_path: str, local_path: str):
        """
        Download a file from GCS to the local filesystem.

        Args:
            gcs_path (str): Path to the file in GCS.
            local_path (str): Destination path locally.
        """
        blob = self.bucket.blob(gcs_path)
        os.makedirs(Path(local_path).parent, exist_ok=True)
        blob.download_to_filename(local_path)
        logger.info(f"Downloaded gs://{self.bucket_name}/{gcs_path} to {local_path}")

    def download_to_string(self, gcs_path: str) -> str:
        """
        Download a file from GCS and return its content as a string.

        Args:
            gcs_path (str): Path to the file in GCS.

        Returns:
            str: File content as a string.
        """
        blob = self.bucket.blob(gcs_path)
        if not blob.exists():
            raise FileNotFoundError(f"File {gcs_path} not found in GCS bucket {self.bucket_name}")
        data = blob.download_as_text()
        logger.info(f"Downloaded content from gs://{self.bucket_name}/{gcs_path}")
        return data

    def delete_file(self, gcs_path: str):
        """
        Delete a file from GCS.

        Args:
            gcs_path (str): Path to the file in GCS.
        """
        blob = self.bucket.blob(gcs_path)
        if blob.exists():
            blob.delete()
            logger.info(f"Deleted gs://{self.bucket_name}/{gcs_path}")
        else:
            logger.info(f"File {gcs_path} does not exist in GCS bucket {self.bucket_name}")

    def list_files(self, prefix: str = ""):
        """
        List files in the GCS bucket under a given prefix.

        Args:
            prefix (str): Prefix to filter files.

        Returns:
            list: List of file paths.
        """
        return [blob.name for blob in self.bucket.list_blobs(prefix=prefix)]

    def file_exists(self, gcs_path: str) -> bool:
        """
        Check if a file exists in GCS.

        Args:
            gcs_path (str): Path to the file in GCS.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        blob = self.bucket.blob(gcs_path)
        return blob.exists()
