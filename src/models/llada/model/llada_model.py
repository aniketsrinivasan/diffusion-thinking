from src.models.llada.model.base_model import BaseModel, BaseModelConfig
from src.models.llada.generate import generate

import torch

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass 
class LLaDaModelConfig(BaseModelConfig):
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct"
    trust_remote_code: bool = True
    torch_dtype: str = "bf16"
    device: Optional[str] = None

    # layer freezing configs
    freeze_embeddings: bool = True
    freeze_transformer_blocks: Optional[Tuple[int, ...]] = None  # either None, or a tuple of block indices


class LLaDaModel(BaseModel):
    def __init__(self, config: LLaDaModelConfig):
        super().__init__(config)

    def generate(self, prompt: str, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        messages = [
            {"role": "user", "content": prompt}
        ]
        prompt_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = self.tokenizer(prompt_ids)['input_ids']
        input_ids = torch.tensor(input_ids).to(self.config.device).unsqueeze(0)
        
        return generate(
            self.model,
            input_ids,
            **kwargs
        ), input_ids

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run the LLaDa model forward pass. 

        Args:
            input_ids: The (noise masked) input IDs to the model. Should be tokenized using the same tokenizer as the model, and 
                passed through the scheduler's "forward" process. 

        Returns:
            The logits of the model. 
        """
        return self.model(input_ids, **kwargs)