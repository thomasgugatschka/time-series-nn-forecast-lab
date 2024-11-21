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
