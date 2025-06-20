from dataclasses import dataclass, asdict
from typing import Optional
from omegaconf import MISSING, OmegaConf, DictConfig
from torch.utils.data import DataLoader
from accelerate import Accelerator, local_sgd
import torch
import time
from tqdm import tqdm
import gc
import os
import shutil
import json
import yaml

from src.models.llada.model.base_model import BaseModelConfig
from src.models.llada.model.dispatch import get_model
from src.models.llada.sampler import LLaDaSampler, LLaDaSamplerConfig
from src.models.llada.loss.loss import GRPOLoss, GRPOLossConfig
from src.data import TUFAIntegralsDataset, TUFAIntegralsDatasetConfig, custom_collate_fn
from src.models.llada.prompts import TRAINING_PROMPTS
from src.models.llada.loss.loss_utils import LossUtils


@dataclass 
class LLaDaTrainerConfig:
    run_name: str = MISSING  # name of the training run
    overwrite_run: bool = False  # whether to overwrite the training run if it already exists
    num_sample_outputs: int = 10  # number of sample outputs to get from the model every validation epoch

    # training objects
    model_config: BaseModelConfig = MISSING  # configuration for the model
    sampler_config: Optional[LLaDaSamplerConfig] = None  # configuration for the sampler (noise adding using mask tokens)
    loss_config: GRPOLossConfig = MISSING  # configuration for the loss
    train_dataset_config: TUFAIntegralsDatasetConfig = MISSING  # configuration for the training dataset
    val_dataset_config: TUFAIntegralsDatasetConfig = MISSING  # configuration for the validation dataset

    # training configuration
    pretrain_mode: bool = False  # whether to use the pretrain mode (if True, noised training on diffusion trajectories occurs)
    prompt_format: str = "integration_0"  # format of the prompt 
    grpo_group_size: int = 8  # number of samples to generate for each prompt (for GRPO)
    grpo_policy_update_steps: int = 8  # number of steps to update the policy (for GRPO)
    clear_cache_frequency: int = 1  # number of training steps to clear the cache

    # training parameters 
    num_epochs: int = 10
    batch_size: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: Optional[int] = 4
    max_grad_norm: float = 0.1
    save_epochs: int = 10
    backup_epochs: int = 5
    validate_epochs: int = 5
    project_dir: str = "experiments"


class LLaDaTrainer:
    """
    Trainer for the LLaDa model.
    """

    def __init__(self, config: LLaDaTrainerConfig):
        """
        Args:
            config (LLaDaTrainerConfig): Configuration for the trainer.
        """
        self.config = config

        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps if self.config.gradient_accumulation_steps is not None else 1,
            project_dir=self.config.project_dir,
        )

        self.global_step = 0
        self.epoch = 0
        self.prompt_format = TRAINING_PROMPTS[self.config.prompt_format]
        self.idle_device = "mps" if torch.backends.mps.is_available() else "cpu"

        self._setup_model()
        self._setup_sampler()
        self._setup_loss()
        self._setup_dataloaders()
        self.optimizer = torch.optim.AdamW(self.model.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        self._prepare_accelerator()

    def _clear_cuda_cache(self):
        """Clear CUDA cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _setup_model(self):
        """
        Setup the model, frozen model, and previous model. At initialization, they are all set to the configured model. 
        """
        self.model = get_model(self.config.model_config)
        self.frozen_model = get_model(self.config.model_config)
        self.prev_model = get_model(self.config.model_config)
        self.frozen_model.model.eval()
        self.prev_model.model.eval()

    def _setup_sampler(self):
        if self.config.sampler_config is not None:
            self.sampler = LLaDaSampler(self.config.sampler_config)
        else:
            self.sampler = None

    def _setup_loss(self):
        self.loss = GRPOLoss(self.config.loss_config)

    def _setup_dataloaders(self):
        self.train_dataloader = DataLoader(
            TUFAIntegralsDataset(self.config.train_dataset_config), 
            batch_size=self.config.batch_size, 
            shuffle=True,
            collate_fn=custom_collate_fn
        )
        self.val_dataloader = DataLoader(
            TUFAIntegralsDataset(self.config.val_dataset_config), 
            batch_size=self.config.batch_size, 
            shuffle=False,
            collate_fn=custom_collate_fn
        )

    def _prepare_accelerator(self):
        self.model, self.sampler, self.loss, self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.model, self.sampler, self.loss, self.train_dataloader, self.val_dataloader
        )

    def _validate_setup(self):
        assert self.model is not None, "Model is not setup."
        assert self.frozen_model is not None, "Frozen model is not setup."
        assert self.sampler is not None, "Sampler is not setup."
        assert self.loss is not None, "Loss is not setup."
        assert self.train_dataloader is not None, "Train dataloader is not setup."
        assert self.val_dataloader is not None, "Validation dataloader is not setup."
        assert self.accelerator is not None, "Accelerator is not setup."

    def _refresh_old_policy(self):
        """
        Updates the previous model to be the current model (deepcopies parameters and sets requires_grad to False). 
        """
        state_dict = {k: v.detach().clone() for k, v in self.model.model.state_dict().items()}
        self.prev_model.model.load_state_dict(state_dict)
        self.prev_model.model.eval()
        for p in self.prev_model.model.parameters():
            p.requires_grad = False
        self._clear_cuda_cache()

    def _sample_group(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        """
        Samples a group of outputs from the previous model. Used the GRPO training algorithm. Sampling from models that
        do not require backpropagation (e.g. frozen models) occurs with torch.no_grad().

        Args:
            prompt (str): The prompt to sample from.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]: The log probabilities of the old, current, and reference models, and the decoded outputs.
        """
        device = self.model.config.device

        seqs = []
        decoded = []

        with torch.no_grad():
            for i in range(self.config.grpo_group_size):
                out, input_ids = self.prev_model.generate(prompt)
                seqs.append(out.squeeze(0))
                decoded.append(
                    self.model.tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
                )
            
                del out, input_ids

        ys_ids = torch.stack(seqs, dim=0).to(device)

        with torch.no_grad():
            logp_old = torch.log_softmax(
                self.prev_model.forward(ys_ids).logits,
                dim=-1
            )
            logp_ref = torch.log_softmax(
                self.frozen_model.forward(ys_ids).logits,
                dim=-1
            )
            
        # requires grad for backpropagation:
        logp_curr = torch.log_softmax(
            self.model.forward(ys_ids).logits,
            dim=-1
        )
        token_ids = ys_ids.unsqueeze(-1)
        logp_old = logp_old.gather(dim=-1, index=token_ids).squeeze(-1)
        logp_curr = logp_curr.gather(dim=-1, index=token_ids).squeeze(-1)
        logp_ref = logp_ref.gather(dim=-1, index=token_ids).squeeze(-1)
        
        del token_ids, ys_ids
        return logp_old, logp_curr, logp_ref, decoded
    
    def setup_dirs(self):
        """
        Set up the project directory for training. 
        """
        if not os.path.exists(os.path.join(self.config.project_dir, self.config.run_name)):
            os.makedirs(os.path.join(self.config.project_dir, self.config.run_name))
            os.makedirs(os.path.join(self.config.project_dir, self.config.run_name, "checkpoints"))
            os.makedirs(os.path.join(self.config.project_dir, self.config.run_name, "sample_outputs"))
        else:
            if self.config.overwrite_run:
                shutil.rmtree(os.path.join(self.config.project_dir, self.config.run_name))
                os.makedirs(os.path.join(self.config.project_dir, self.config.run_name))
                os.makedirs(os.path.join(self.config.project_dir, self.config.run_name, "checkpoints"))
                os.makedirs(os.path.join(self.config.project_dir, self.config.run_name, "sample_outputs"))
            else:
                raise FileExistsError(f"Run {self.config.run_name} already exists. Set overwrite_run=True to overwrite.")
        # save config to the project directory:
        with open(os.path.join(self.config.project_dir, self.config.run_name, "config.yaml"), "w") as f:
            config_dict = asdict(self.config)
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def save_checkpoint(self, checkpoint_name: str, overwrite: bool = False):
        """
        Save the current model weights as a checkpoint to the project directory.
        """
        # save checkpoint to the project directory "checkpoints/" directory:
        if os.path.exists(os.path.join(self.config.project_dir, self.config.run_name, "checkpoints", checkpoint_name)):
            if overwrite:
                os.remove(os.path.join(self.config.project_dir, self.config.run_name, "checkpoints", checkpoint_name))
            else:
                raise FileExistsError(f"Checkpoint {checkpoint_name} already exists. Set overwrite=True to overwrite.")
        
        self.accelerator.save_state(os.path.join(self.config.project_dir, self.config.run_name, "checkpoints", checkpoint_name))

    @torch.no_grad()
    def save_sample_outputs(self):
        """
        Get sample outputs from the model(s). Samples data from the training and validation datasets. 
        """
        self.model.model.eval()
        
        # sample num_sample_outputs random indices from the training and validation datasets:
        train_indices = torch.randint(0, len(self.train_dataloader.dataset), (self.config.num_sample_outputs,))
        val_indices = torch.randint(0, len(self.val_dataloader.dataset), (self.config.num_sample_outputs,))
        
        # get the sample outputs from the model(s):
        train_outputs = []
        for i in train_indices:
            item = self.train_dataloader.dataset[i]
            input_prompt = self.prompt_format.replace("<integral>", item["variant"])

            # self.model prediction:
            curr_model_out, curr_model_input_ids = self.model.generate(input_prompt)
            curr_model_decoded = self.model.tokenizer.batch_decode(curr_model_out[:, curr_model_input_ids.shape[1]:], skip_special_tokens=True)
            curr_model_math_pred = LossUtils.get_prediction_from_model_output(curr_model_decoded[0])
            try:
                curr_model_math_pred = LossUtils.latex_to_sympy(curr_model_math_pred)
            except Exception as e:
                curr_model_math_pred = curr_model_math_pred

            # self.frozen_model prediction:
            frozen_model_out, frozen_model_input_ids = self.frozen_model.generate(input_prompt)
            frozen_model_decoded = self.frozen_model.tokenizer.batch_decode(frozen_model_out[:, frozen_model_input_ids.shape[1]:], skip_special_tokens=True)
            frozen_model_math_pred = LossUtils.get_prediction_from_model_output(frozen_model_decoded[0])
            try:
                frozen_model_math_pred = LossUtils.latex_to_sympy(frozen_model_math_pred)
            except Exception as e:
                frozen_model_math_pred = frozen_model_math_pred

            train_outputs.append({
                "curr_model_decoded": curr_model_decoded,
                "frozen_model_decoded": frozen_model_decoded,
                "curr_model_math_pred": curr_model_math_pred,
                "frozen_model_math_pred": frozen_model_math_pred,
                "ground_truth": item["variant"],
                "input_prompt": input_prompt,
                "full_training_sample": item
            })
        
        val_outputs = []
        for i in val_indices:
            item = self.val_dataloader.dataset[i]
            input_prompt = self.prompt_format.replace("<integral>", item["variant"])
            
            # self.model prediction:
            curr_model_out, curr_model_input_ids = self.model.generate(input_prompt)
            curr_model_decoded = self.model.tokenizer.batch_decode(curr_model_out[:, curr_model_input_ids.shape[1]:], skip_special_tokens=True)
            curr_model_math_pred = LossUtils.get_prediction_from_model_output(curr_model_decoded[0])
            try:
                curr_model_math_pred = LossUtils.latex_to_sympy(curr_model_math_pred)
            except Exception as e:
                curr_model_math_pred = curr_model_math_pred

            # self.frozen_model prediction:
            frozen_model_out, frozen_model_input_ids = self.frozen_model.generate(input_prompt)
            frozen_model_decoded = self.frozen_model.tokenizer.batch_decode(frozen_model_out[:, frozen_model_input_ids.shape[1]:], skip_special_tokens=True)
            frozen_model_math_pred = LossUtils.get_prediction_from_model_output(frozen_model_decoded[0])
            try:
                frozen_model_math_pred = LossUtils.latex_to_sympy(frozen_model_math_pred)
            except Exception as e:
                frozen_model_math_pred = frozen_model_math_pred

            val_outputs.append({
                "curr_model_decoded": curr_model_decoded,
                "frozen_model_decoded": frozen_model_decoded,
                "curr_model_math_pred": curr_model_math_pred,
                "frozen_model_math_pred": frozen_model_math_pred,
                "ground_truth": item["variant"],
                "input_prompt": input_prompt,
                "full_validation_sample": item
            })

        self.model.model.train()
        # save the sample outputs to the project directory:
        with open(os.path.join(self.config.project_dir, self.config.run_name, "sample_outputs", f"sample_outputs_epoch_{self.epoch}_train.json"), "w") as f:
            json.dump(train_outputs, f)
        with open(os.path.join(self.config.project_dir, self.config.run_name, "sample_outputs", f"sample_outputs_epoch_{self.epoch}_val.json"), "w") as f:
            json.dump(val_outputs, f)

    @torch.no_grad()
    def run_validation(self):
        """
        Run validation on the model.
        """
        pass

    def train_epoch(self):
        """
        Train self.model.model for one epoch. Responsibility of the caller to update self.epoch. 
        """
        self.model.model.train()
        self.frozen_model.model.eval()
        self.prev_model.model.eval()
        running_loss = 0.0
        start_time = time.time()

        with tqdm(self.train_dataloader, desc=f"Training epoch {self.epoch}") as pbar:
            with local_sgd.LocalSGD(
                self.accelerator,
                self.model,
                self.config.gradient_accumulation_steps,
            ) as loc_sgd:
                for batch_idx, data in enumerate(pbar):
                    batch_losses = []
                    
                    for item in data:
                        if not item.get("variant") or item["variant"].strip() == "":
                            continue
                            
                        # tokenize the data and add noise
                        input_prompt = self.prompt_format.replace("<integral>", item["variant"])

                        if batch_idx % self.config.clear_cache_frequency == 0:
                            self._clear_cuda_cache()

                        if self.config.pretrain_mode:
                            tokenized_data = self.model.tokenize(input_prompt)
                            tokenized_data, mask_indices, mask_probability = self.sampler.add_noise_masks(tokenized_data)

                        logp_old, logp_curr, logp_ref, decoded = self._sample_group(input_prompt)

                        loss = self.loss.get_simple_loss(
                            model_outputs=decoded,
                            ground_truth=item,
                            logprobs_new=logp_curr,
                            logprobs_old=logp_old,
                            logprobs_ref=logp_ref,
                        )
                        batch_losses.append(loss)
                    
                        del logp_old, logp_curr, logp_ref, decoded
                    
                    if batch_losses:
                        total_loss = torch.stack(batch_losses).mean()
                        
                        # update policy:
                        self.accelerator.backward(total_loss)
                        
                        if self.config.max_grad_norm > 0:
                            self.accelerator.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)
                        
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        loc_sgd.step()

                        if self.global_step % self.config.grpo_policy_update_steps == 0:
                            self._refresh_old_policy()

                        running_loss += total_loss.item()
                        self.global_step += 1
                        pbar.set_postfix(loss=running_loss / self.global_step)
                        
                        self._clear_cuda_cache()
                        batch_losses = []
                        del total_loss

        end_time = time.time()
        self.accelerator.print(f"Training epoch {self.epoch} completed in {end_time - start_time} seconds. Average loss: {running_loss / len(self.train_dataloader)}")
        self._clear_cuda_cache()

    def train(self):
        """
        Train self.model.model for self.config.num_epochs epochs.
        """
        self._validate_setup()
        self.setup_dirs()
        while self.epoch < self.config.num_epochs:
            self.train_epoch()

            if self.epoch > 0 and self.epoch % self.config.save_epochs == 0:
                self.save_checkpoint(checkpoint_name=f"checkpoint_{self.epoch}.pt")
            if self.epoch > 0 and self.epoch % self.config.backup_epochs == 0:
                self.save_checkpoint(checkpoint_name="backup_checkpoint.pt", overwrite=True)

            if self.epoch > 0 and self.epoch % self.config.validate_epochs == 0:
                self.save_sample_outputs()
                self.run_validation()

            self.epoch += 1

