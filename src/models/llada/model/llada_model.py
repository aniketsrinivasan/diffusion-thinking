from src.models.llada.model.base_model import BaseModel, BaseModelConfig
from src.models.llada.generate import generate, add_gumbel_noise, get_num_transfer_tokens

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


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

    def step(self, 
             x_t: torch.Tensor, 
             num_tokens_to_unmask: int,
             temperature: float = 0.3,
             remasking: str = 'low_confidence',
             cfg_scale: float = 0.0,
             prompt_ids: Optional[torch.Tensor] = None,
             mask_id: int = 126336) -> torch.Tensor:
        """
        Perform one step of the LLaDA diffusion denoising process.
        
        Args:
            x_t: Current sequence with some tokens masked [batch_size, seq_len]
            num_tokens_to_unmask: Number of tokens to unmask in this step
            temperature: Temperature for Gumbel sampling
            remasking: Remasking strategy ('low_confidence' or 'random')
            cfg_scale: Classifier-free guidance scale
            prompt_ids: Original prompt tokens (for CFG) [batch_size, prompt_len]
            mask_id: Token ID used for masking (default: 126336)
            
        Returns:
            torch.Tensor: Updated sequence with some tokens unmasked [batch_size, seq_len]
        """
        with torch.no_grad():
            mask_index = (x_t == mask_id)
            
            if cfg_scale > 0.0 and prompt_ids is not None:
                prompt_index = torch.zeros_like(x_t, dtype=torch.bool)
                prompt_index[:, :prompt_ids.shape[1]] = True
                
                un_x = x_t.clone()
                un_x[prompt_index] = mask_id
                
                x_combined = torch.cat([x_t, un_x], dim=0)
                logits_combined = self.model(x_combined).logits
                
                # split and apply CFG
                logits, un_logits = torch.chunk(logits_combined, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = self.model(x_t).logits
            
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            
            # get predicted tokens for all positions
            x0_pred = torch.argmax(logits_with_noise, dim=-1)  # [batch_size, seq_len]
            
            # calculate confidence scores based on remasking strategy
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                confidence = torch.gather(p, dim=-1, index=x0_pred.unsqueeze(-1)).squeeze(-1)
            elif remasking == 'random':
                confidence = torch.rand_like(x0_pred, dtype=torch.float, device=x0_pred.device)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")
            
            confidence = torch.where(mask_index, confidence, -np.inf)
            
            # select tokens to unmask based on confidence
            batch_size, seq_len = x_t.shape
            updated_x = x_t.clone()
            
            for batch_idx in range(batch_size):
                # get top-k most confident predictions for this batch item
                if num_tokens_to_unmask > 0:
                    _, select_indices = torch.topk(
                        confidence[batch_idx], 
                        k=min(num_tokens_to_unmask, mask_index[batch_idx].sum().item())
                    )
                    
                    transfer_mask = torch.zeros(seq_len, dtype=torch.bool, device=x_t.device)
                    transfer_mask[select_indices] = True
                    
                    updated_x[batch_idx] = torch.where(
                        transfer_mask, 
                        x0_pred[batch_idx], 
                        x_t[batch_idx]
                    )
            
            return updated_x

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