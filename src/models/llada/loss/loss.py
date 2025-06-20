import torch
from dataclasses import dataclass

from src.models.llada.loss.loss_metrics import MathCorrectnessMetric, MathFormatMetric, MathEvalMetric


@dataclass
class GRPOLossConfig:
    use_math_eval: bool = True  # whether to use mathematical evaluations (numerical evaluation correctness)
    use_math_correctness: bool = True  # whether to use mathematical correctness (correctness of the mathematical expression)
    use_math_format: bool = True  # whether to use mathematical format (format of the mathematical expression)
    clip_epsilon: float = 0.2  # epsilon for the clipped objective
    kl_beta: float = 0.02  # beta for the KL divergence penalty
    regex_match: str = r"\\begin{equation}(.+?)\\end{equation}"  # regex to match the mathematical expression solution


class GRPOLoss:
    """
    Loss function for group regularized policy optimization (GRPO). Uses reward metrics (particularly on mathematical correctness)
    to approximate a reward function for the policy. 
    """

    def __init__(self, config: GRPOLossConfig):
        """
        Args:
            config (GRPOLossConfig): Configuration for the GRPO loss. 
        """
        self.config = config
        self.reward_functions = [
            MathEvalMetric(self.config.regex_match) if self.config.use_math_eval else None,
            MathCorrectnessMetric(self.config.regex_match) if self.config.use_math_correctness else None,
            MathFormatMetric(self.config.regex_match) if self.config.use_math_format else None,
        ]

    def get_reward(self, model_output: str, ground_truth: dict) -> torch.Tensor:
        """
        Get the total reward for one predicted model output.

        Args:
            model_output (str): The predicted model output.
            ground_truth (dict): The ground truth data.

        Returns:
            The total reward for the predicted model output [1].
        """
        reward = torch.tensor(0.0)
        if "".join(model_output).strip() == "":
            # encourage the model to make a prediction
            reward -= 1.0
            return reward
        for reward_function in self.reward_functions:
            if reward_function is not None:
                reward += reward_function(model_output, ground_truth)
        return reward
    
    def get_batch_reward(self, model_outputs: list[str], ground_truth: dict) -> torch.Tensor:
        """
        Get the batch reward for the predicted model outputs. Returns normalized rewards [G].

        Args:
            model_outputs (list[str]): The predicted model outputs.
            ground_truth (dict): The ground truth data.

        Returns:
            The normalized batch reward for the predicted model outputs [G].
        """
        rewards = []
        for model_output in model_outputs:
            rewards.append(self.get_reward(model_output, ground_truth))
        rewards = torch.stack(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return rewards
    
    def get_simple_loss(
            self, 
            model_outputs: list[str], 
            ground_truth: dict,
            logprobs_new: torch.Tensor,
            logprobs_old: torch.Tensor, 
            logprobs_ref: torch.Tensor,
        ) -> torch.Tensor:
        """
        Compute the actual GRPO loss. Returns a scalar loss to minimize.
        
        Args:
            model_outputs: List of G generated text outputs for this prompt [G]
            ground_truth: Ground truth data for this prompt [1]
            logprobs_new: Log probabilities from current policy [G, seq_len, vocab_size]
            logprobs_old: Log probabilities from old policy [G, seq_len, vocab_size] 
            logprobs_ref: Log probabilities from reference policy [G, seq_len, vocab_size]
            
        Returns:
            torch.Tensor: Scalar GRPO loss to minimize [1]
        """
        # get normalized rewards (advantages) for this group
        with torch.no_grad():
            rewards = self.get_batch_reward(model_outputs, ground_truth).to(logprobs_new.device)  # [B]
            print(f"REWARDS: {rewards}")
            advantages = rewards.unsqueeze(-1).expand_as(logprobs_new).to(logprobs_new.device)  # [G, seq_len]
        
        # compute importance ratio π_θ(o) / π_old(o)
        ratio = (logprobs_new - logprobs_old).exp()  # [G, seq_len]
        
        unclipped_objective = ratio * advantages  # [G, seq_len]
        clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
        clipped_objective = clipped_ratio * advantages
        surrogate_objective = torch.minimum(unclipped_objective, clipped_objective)

        surrogate_loss = surrogate_objective.sum(dim=-1)  # [G]

        kl_divergence = (logprobs_new - logprobs_ref)  # [G, seq_len]
        kl_divergence = kl_divergence.sum(dim=-1)  # [G]
        
        objective = surrogate_loss - self.config.kl_beta * kl_divergence  # [G]
        loss = -objective.mean()
        
        return loss

    def __call__(
            self, 
            model_outputs: list[str], 
            ground_truth: dict,
            diffusion_trajectories: list[list[tuple]],
            logits_new_fn: callable,
            logits_old_fn: callable,
            logits_ref_fn: callable
        ) -> torch.Tensor:
        """
        Compute GRPO loss for LLaDa diffusion model.
        
        Args:
            model_outputs: List of G generated text outputs for this prompt
            ground_truth: Ground truth data for this prompt
            diffusion_trajectories: List of G trajectories, each containing 
                                  [(x_t, mask_t, sampled_ids_t)] for each denoising step
            logits_new_fn: Function to get logits from current policy π_θ(x_t, step=t)
            logits_old_fn: Function to get logits from old policy π_old(x_t, step=t)  
            logits_ref_fn: Function to get logits from reference policy π_ref(x_t, step=t)
            
        Returns:
            torch.Tensor: Scalar GRPO loss to minimize
        """
        # Step 1: Get normalized rewards (advantages) for this group
        with torch.no_grad():
            rewards = self.get_batch_reward(model_outputs, ground_truth)  # [G]
            # Normalize advantages within the group (group-relative)
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # [G]
        
        total_losses = []
        
        # Step 2: Process each sample trajectory
        for sample_idx, (trajectory, advantage) in enumerate(zip(diffusion_trajectories, advantages)):
            sample_losses = []
            
            # Step 3: Process each denoising step in the trajectory
            for step_idx, (x_t, mask_t, sampled_ids_t) in enumerate(trajectory):
                # Get current step number (assuming trajectory is ordered from T->1)
                current_step = len(trajectory) - step_idx
                
                # Get logits from all three policies for this step
                with torch.no_grad():
                    logits_old = logits_old_fn(x_t, step=current_step)  # [seq_len, vocab_size]
                    logits_ref = logits_ref_fn(x_t, step=current_step)  # [seq_len, vocab_size]
                
                logits_new = logits_new_fn(x_t, step=current_step)  # [seq_len, vocab_size] (requires grad)
                
                # Convert logits to log probabilities
                logprobs_new = torch.log_softmax(logits_new, dim=-1)  # [seq_len, vocab_size]
                logprobs_old = torch.log_softmax(logits_old, dim=-1)  # [seq_len, vocab_size]
                logprobs_ref = torch.log_softmax(logits_ref, dim=-1)  # [seq_len, vocab_size]
                
                # Step 4: Get log probabilities for the actually sampled tokens
                # sampled_ids_t contains the token IDs that were actually generated by π_old
                logp_new_tokens = torch.gather(logprobs_new, -1, sampled_ids_t.unsqueeze(-1)).squeeze(-1)  # [seq_len]
                logp_old_tokens = torch.gather(logprobs_old, -1, sampled_ids_t.unsqueeze(-1)).squeeze(-1)  # [seq_len]
                logp_ref_tokens = torch.gather(logprobs_ref, -1, sampled_ids_t.unsqueeze(-1)).squeeze(-1)  # [seq_len]
                
                # Step 5: Compute importance ratio π_θ(tokens) / π_old(tokens)
                ratio = (logp_new_tokens - logp_old_tokens).exp()  # [seq_len]
                
                # Step 6: Apply advantages only to masked positions
                # Only tokens that were predicted at this step should get the advantage
                masked_advantages = advantage * mask_t.float()  # [seq_len]
                
                # Step 7: Compute clipped surrogate objective
                unclipped_objective = ratio * masked_advantages
                clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                clipped_objective = clipped_ratio * masked_advantages
                surrogate_objective = torch.minimum(unclipped_objective, clipped_objective)
                
                # Step 8: Sum over masked tokens for this step
                surrogate_loss = surrogate_objective.sum()  # scalar
                
                # Step 9: Compute KL divergence penalty (only for masked tokens)
                kl_divergence = (logp_new_tokens - logp_ref_tokens) * mask_t.float()  # [seq_len]
                kl_loss = kl_divergence.sum()  # scalar
                
                # Step 10: Combine surrogate objective and KL penalty for this step
                step_objective = surrogate_loss - self.config.kl_beta * kl_loss
                sample_losses.append(-step_objective)  # negative because we want to maximize objective
                
            # Sum losses across all steps for this sample
            if sample_losses:
                total_losses.append(torch.stack(sample_losses).sum())
        
        # Step 11: Average loss across all samples
        if total_losses:
            loss = torch.stack(total_losses).mean()
        else:
            loss = torch.tensor(0.0)
        
        return loss
