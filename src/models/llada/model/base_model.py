from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

from omegaconf import MISSING
import torch
from transformers import AutoModel, AutoTokenizer



class BaseModelConfig(ABC):
    model_name: str = MISSING
    trust_remote_code: bool = True
    torch_dtype: str = "bf16"
    device: Optional[str] = None

    # layer freezing configs
    freeze_embeddings: bool = True
    freeze_transformer_blocks: Optional[Tuple[int, ...]] = None  # either None, or a tuple of block indices


class BaseModel(ABC):
    def __init__(self, config: BaseModelConfig):
        if config.device is None:
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.dtype = torch.bfloat16 if self.config.torch_dtype == "bf16" else torch.float16

        self._load_models()
        self._apply_layer_freezing()

    def _load_models(self):
        """
        Load the LLaDa model and tokenizer. 
        """
        # AutoModelForCausalLM is used for models that need generation capability
        try:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, 
                trust_remote_code=self.config.trust_remote_code, 
                torch_dtype=self.dtype
            ).to(self.config.device)
        except Exception:
            # AutoModel is used for models that don't support causal LM
            self.model = AutoModel.from_pretrained(
                self.config.model_name, 
                trust_remote_code=self.config.trust_remote_code, 
                torch_dtype=self.dtype
            ).to(self.config.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, 
            trust_remote_code=self.config.trust_remote_code
        )

    def _apply_layer_freezing(self):
        """
        Apply layer freezing according to the configuration.
        """
        if not any([
            self.config.freeze_embeddings,
            self.config.freeze_transformer_blocks,
        ]):
            return
        
        frozen_params = 0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            should_freeze = self._should_freeze_parameter(name)
            
            if should_freeze:
                param.requires_grad = False
                frozen_params += param.numel()
                
        print(f"Frozen {frozen_params:,} parameters out of {total_params:,} "
              f"({100 * frozen_params / total_params:.1f}%)")

    def _should_freeze_parameter(self, param_name: str) -> bool:
        """
        Determine if a parameter should be frozen based on its name and configuration.
        
        Args:
            param_name: The full parameter name (e.g., "model.transformer.blocks.0.q_proj.weight")
            
        Returns:
            bool: True if the parameter should be frozen
        """
        # Freeze embeddings
        if self.config.freeze_embeddings:
            if "transformer.wte" in param_name:
                return True
        
        # Freeze specific transformer blocks
        if self.config.freeze_transformer_blocks:
            if "transformer.blocks." in param_name:
                if self.config.freeze_transformer_blocks is not None:
                    # Extract block number from parameter name
                    parts = param_name.split(".")
                    for i, part in enumerate(parts):
                        if part == "blocks" and i + 1 < len(parts):
                            try:
                                block_idx = int(parts[i + 1])
                                return block_idx in self.config.freeze_transformer_blocks
                            except (ValueError, IndexError):
                                continue
        
        return False

    def freeze_transformer_blocks(self, block_indices: Tuple[int, ...]):
        """
        Freeze specific transformer blocks after model initialization.
        
        Args:
            block_indices: Tuple of block indices to freeze (0-31 for LLaDA-8B)
        """
        total_blocks = self.model.config.n_layers
        block_indices = list(range(total_blocks))
        
        frozen_params = 0
        for name, param in self.model.named_parameters():
            if "transformer.blocks." in name:
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if part == "blocks" and i + 1 < len(parts):
                        try:
                            block_idx = int(parts[i + 1])
                            if block_idx in block_indices:
                                if param.requires_grad:
                                    param.requires_grad = False
                                    frozen_params += param.numel()
                        except (ValueError, IndexError):
                            continue
        
        print(f"Frozen {frozen_params:,} parameters in transformer blocks {block_indices}")

    def freeze_embeddings(self):
        """
        Freeze embedding layers after model initialization.
        """
        frozen_params = 0
        for name, param in self.model.named_parameters():
            if "transformer.wte" in name and param.requires_grad:
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"Frozen {frozen_params:,} parameters in embeddings")

    def get_frozen_parameters_info(self) -> dict:
        """
        Get information about frozen parameters.
        
        Returns:
            dict: Information about frozen vs trainable parameters
        """
        frozen_params = 0
        trainable_params = 0
        frozen_by_component = {
            "embeddings": 0,
            "transformer_blocks": {},
            "other": 0
        }
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()
                
                # Categorize frozen parameters
                if "transformer.wte" in name:
                    frozen_by_component["embeddings"] += param.numel()
                elif "transformer.ln_f" in name:
                    frozen_by_component["final_layer_norm"] += param.numel()
                elif "transformer.ff_out" in name:
                    frozen_by_component["output_projection"] += param.numel()
                elif "transformer.blocks." in name:
                    # Extract block number
                    parts = name.split(".")
                    for i, part in enumerate(parts):
                        if part == "blocks" and i + 1 < len(parts):
                            try:
                                block_idx = int(parts[i + 1])
                                if block_idx not in frozen_by_component["transformer_blocks"]:
                                    frozen_by_component["transformer_blocks"][block_idx] = 0
                                frozen_by_component["transformer_blocks"][block_idx] += param.numel()
                                break
                            except (ValueError, IndexError):
                                frozen_by_component["other"] += param.numel()
                                break
                    else:
                        frozen_by_component["other"] += param.numel()
                else:
                    frozen_by_component["other"] += param.numel()
        
        total_params = frozen_params + trainable_params
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "frozen_percentage": 100 * frozen_params / total_params if total_params > 0 else 0,
            "frozen_by_component": frozen_by_component
        }

    def tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize the text using the model's tokenizer.

        Args:
            text: The text to tokenize.

        Returns:
            The tokenized text, as token IDs of shape [1, seq_len].
        """
        input_ids = self.tokenizer(text)['input_ids']
        input_ids = torch.tensor(input_ids).to(self.config.device).unsqueeze(0)
        return input_ids
    
    def generate(self, prompt: str, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a sequence of prediction_ids from the model, given a natural language prompt as a string.
        
        Args:
            prompt: The input prompt string
            **kwargs: Generation parameters (max_length, temperature, etc.)
            
        Returns:
            tuple: (generated_tokens, input_tokens)
                - generated_tokens: Generated tokens [1, seq_len]
                - input_tokens: Input tokens [1, input_len]
        """
        messages = [
            {"role": "user", "content": prompt}
        ]
        prompt_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = self.tokenizer(prompt_ids)['input_ids']
        input_ids = torch.tensor(input_ids).to(self.config.device).unsqueeze(0)
        
        generation_kwargs = {
            'temperature': 0.5,
            'do_sample': True,
            'pad_token_id': self.tokenizer.eos_token_id,
            **kwargs  # override defaults with kwargs if necessary
        }
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                **generation_kwargs
            )

        return generated_ids[:, input_ids.shape[1]:], input_ids

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run the model forward pass. 

        Args:
            input_ids: The (noise masked) input IDs to the model. Should be tokenized using the same tokenizer as the model, and 
                passed through the scheduler's "forward" process. 

        Returns:
            The logits of the model. 
        """
        return self.model(input_ids, **kwargs)

