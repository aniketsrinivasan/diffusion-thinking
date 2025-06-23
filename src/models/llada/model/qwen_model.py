from src.models.llada.model.base_model import BaseModel, BaseModelConfig

import torch
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass 
class QwenModelConfig(BaseModelConfig):
    model_name: str = "Qwen/Qwen2.5-3B"
    trust_remote_code: bool = True
    torch_dtype: str = "bf16"
    device: Optional[str] = None

    # layer freezing configs
    freeze_embeddings: bool = True
    freeze_transformer_blocks: Optional[Tuple[int, ...]] = None  # either None, or a tuple of block indices


class QwenModel(BaseModel):
    def __init__(self, config: QwenModelConfig):
        super().__init__(config)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run the Qwen model forward pass. 

        Args:
            input_ids: The input IDs to the model. Should be tokenized using the same tokenizer as the model.

        Returns:
            The logits of the model. 
        """
        return self.model(input_ids, **kwargs)