import subprocess
from utilities.logging_utils import get_logger

logger = get_logger(__name__)

# def start_tensorboard(log_dir, port=6006):
#     """
#     Starts TensorBoard in a separate process.
#
#     Args:
#         log_dir (str): Path to the directory where TensorBoard logs are stored.
#         port (int): Port to run TensorBoard on (default is 6006).
#     """
#
#     # Construct the URL for TensorBoard
#     tensorboard_url = f"http://localhost:{port}"
#
#     # Log the information about starting TensorBoard along with the clickable link
#     logger.info(f"Starting TensorBoard: {tensorboard_url}")
#
#     # Run the TensorBoard command
#     command = [
#         "tensorboard",
#         "--logdir", log_dir,
#         "--port", str(port),
#         "--bind_all"
#     ]
#     subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def start_tensorboard(root_log_dir, port=6006):
    """
    Start TensorBoard on the root directory that contains logs for multiple experiments.

    Args:
        root_log_dir (str): Path to the root directory where all TensorBoard logs are stored.
        port (int): Port to run TensorBoard on (default is 6006).
    """
    # Construct the URL for TensorBoard
    tensorboard_url = f"http://localhost:{port}"

    # Log the information about starting TensorBoard along with the clickable link
    logger.info(f"Starting TensorBoard at {tensorboard_url} and watching {root_log_dir}")

    # Run the TensorBoard command for the root directory
    command = [
        "tensorboard",
        "--logdir", root_log_dir,
        "--port", str(port),
        "--bind_all"
    ]

    # Start TensorBoard in the background
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

