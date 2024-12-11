from abc import ABC
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    def __init__(self, model_name="BaseModel"):
        super().__init__()
        self.model_name = model_name

    def get_model_name(self):
        """
        Get the name of the model.

        Returns:
            str: The name of the model.
        """
        return self.model_name

    @staticmethod
    def validate_parameters(**params):
        """
        Validate the parameters to ensure they are correctly provided.

        Args:
            **params: Parameter key-value pairs to validate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        errors = []
        for key, value in params.items():

            if value is None:
                errors.append(f"Missing required parameter: {key}")
            elif isinstance(value, (int, float)) and value <= 0:
                errors.append(f"Parameter '{key}' must be greater than 0, got {value}")

        if errors:
            raise ValueError("Invalid parameters provided:\n" + "\n".join(errors))