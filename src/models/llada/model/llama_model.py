from src.models.llada.model.base_model import BaseModel, BaseModelConfig

import torch
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass 
class LlamaModelConfig(BaseModelConfig):
    model_name: str = "meta-llama/Llama-3.2-3B"
    trust_remote_code: bool = True
    torch_dtype: str = "bf16"
    device: Optional[str] = None

    # layer freezing configs
    freeze_embeddings: bool = True
    freeze_transformer_blocks: Optional[Tuple[int, ...]] = None  # either None, or a tuple of block indices


class LlamaModel(BaseModel):
    def __init__(self, config: LlamaModelConfig):
        super().__init__(config)

    def generate(self, prompt: str, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run the Llama model forward pass. 

        Args:
            input_ids: The (noise masked) input IDs to the model. Should be tokenized using the same tokenizer as the model, and 
                passed through the scheduler's "forward" process. 

        Returns:
            The logits of the model. 
        """
        raise NotImplementedError()