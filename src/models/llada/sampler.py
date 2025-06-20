"""
Define the forward (noising) process for LLaDa. 
"""

import torch


class LLaDaSamplerConfig:
    """
    Configuration for the LLaDa sampler. 
    """
    mask_id: int = 126336
    noising_method: str = "linear"
    noise_sample_method: str = "normal"


class LLaDaSampler:
    """
    Sampler for the LLaDa model. 
    """

    def __init__(self, config: LLaDaSamplerConfig):
        self.config = config
        
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Sample a timesteps Tensor for the noising process.

        Args:
            batch_size: The batch size to sample.

        Returns:
            The sampled timesteps [B].
        """
        if self.config.noise_sample_method == "normal":
            return torch.randn(batch_size, device=self.device)
        elif self.config.noise_sample_method == "uniform" or self.config.noise_sample_method == "linear":
            return torch.rand(batch_size, device=self.device)
        else:
            raise ValueError(f"Invalid noise sample method: {self.config.noise_sample_method}.")

    def add_noise_masks(self, input_ids: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add noise to the input IDs by a masking process.

        Args:
            input_ids: The input IDs to add noise to [B, L].

        Returns:
            noisy_batch: The noisy input IDs [B, L].
            mask_indices: The mask indices [B, L].
            mask_probability: The mask probability [B].
        """
        B, L = input_ids.shape
        timesteps = self.sample_timesteps(B)
        mask_probability = (1 - eps) * timesteps + eps

        # Sample mask indices using the mask probability.
        mask_indices = torch.rand((B, L), device=input_ids.device) < mask_probability
        noisy_batch = torch.where(mask_indices, self.config.mask_id, input_ids)  # add [MASK] token wherever masked
        return noisy_batch, mask_indices, mask_probability
