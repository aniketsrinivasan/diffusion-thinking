import torch
from dataclasses import dataclass
from typing import Optional, Tuple

from src.models.llada.loss.loss_metrics import MathCorrectnessMetric, MathFormatMetric, MathEvalMetric, LengthRewardMetric, AntiGamingMetric, MathSimilarityMetric


@dataclass
class GRPOLossConfig:
    clip_epsilon: float = 0.2  # epsilon for the clipped objective
    kl_beta: float = 0.2  # beta for the KL divergence penalty
    regex_match: Optional[str] = None  # regex to match the mathematical expression solution
    use_boxed: bool = True  # whether to use boxed content for the mathematical expression solution
    
    # metric weights
    math_eval_weight: Optional[Tuple[float, float]] = (-1.0, 1.0)
    math_correctness_weight: Optional[Tuple[float, float]] = (-1.0, 1.0)
    math_format_weight: Optional[Tuple[float, float]] = (-1.0, 1.0)
    length_reward_weight: Optional[Tuple[float, float]] = (-1.0, 1.0)
    anti_gaming_weight: Optional[Tuple[float, float]] = (-5.0, 0.0)  # Strong penalty for gaming
    math_similarity_weight: Optional[Tuple[float, float]] = (-1.0, 1.0)
    
    # anti-gaming specific settings
    anti_gaming_strong_penalty: bool = True  # whether to use strong penalties for gaming
    
    # math similarity specific settings
    similarity_threshold: float = 0.7  # minimum similarity for partial credit
    partial_credit: bool = True  # whether to give partial credit for structurally similar expressions


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
            MathEvalMetric(use_boxed=self.config.use_boxed, regex_match=self.config.regex_match, weight=self.config.math_eval_weight) if self.config.math_eval_weight is not None else None,
            MathCorrectnessMetric(use_boxed=self.config.use_boxed, regex_match=self.config.regex_match, weight=self.config.math_correctness_weight) if self.config.math_correctness_weight is not None else None,
            MathFormatMetric(use_boxed=self.config.use_boxed, regex_match=self.config.regex_match, weight=self.config.math_format_weight) if self.config.math_format_weight is not None else None,
            LengthRewardMetric(weight=self.config.length_reward_weight) if self.config.length_reward_weight is not None else None,
            AntiGamingMetric(weight=self.config.anti_gaming_weight) if self.config.anti_gaming_weight is not None else None,
            MathSimilarityMetric(use_boxed=self.config.use_boxed, weight=self.config.math_similarity_weight, similarity_threshold=self.config.similarity_threshold, partial_credit=self.config.partial_credit) if self.config.math_similarity_weight is not None else None,
        ]
        self.total_weight = sum(
            [
                self.config.math_eval_weight[1] if self.config.math_eval_weight is not None else 0,
                self.config.math_correctness_weight[1] if self.config.math_correctness_weight is not None else 0,
                self.config.math_format_weight[1] if self.config.math_format_weight is not None else 0,
                self.config.length_reward_weight[1] if self.config.length_reward_weight is not None else 0,
            ]
        )

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
            empty_penalty = -1.0 * self.total_weight
            reward += empty_penalty
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
        group_size = logprobs_new.shape[0]
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

        surrogate_loss = surrogate_objective.sum(dim=-1) / group_size  # [G]

        kl_divergence = (logprobs_new - logprobs_ref)  # [G, seq_len]
        kl_divergence = kl_divergence.sum(dim=-1) / group_size  # [G]
        
        objective = surrogate_loss - self.config.kl_beta * kl_divergence  # [G]
        loss = -objective.mean()
        
        return loss

    def get_trajectory_loss(
            self,
            model_outputs: list[str],
            ground_truth: dict,
            traj_new: list[list[dict]],
            traj_old: list[list[dict]],
            traj_ref: list[list[dict]],
    ) -> torch.Tensor:
        """
        GRPO loss for a **masked-token diffusion LM** (e.g. LLaDA)

        Parameters
        ----------
        model_outputs : list[str]
            Final decoded answers produced by π_old for this prompt            [G]
        ground_truth  : dict
            Ground-truth solution / labels                                     [1]
        traj_new / traj_old / traj_ref : list[list[dict]]
            Parallel trajectories for current θ, behaviour (old), and reference
            models.  Outer list length = G (group size); inner = T steps.
            Each step-dict MUST contain
                "logits" : FloatTensor [L, V]      # raw logits for this model/step
                "mask"   : BoolTensor  [L]         # 1 where tokens denoised at step t
                "tokens" : LongTensor  [L]         # token ids actually sampled by π_old
                "noise"  : float (optional)        # 1 − ᾱ_t, used to scale KL
        Returns
        -------
        torch.Tensor
            Scalar loss – minimise this.
        """

        # --------------------------------------------------------
        # 1. Group-relative advantage  (identical to simple loss)
        # --------------------------------------------------------
        with torch.no_grad():
            rewards = self.get_batch_reward(model_outputs, ground_truth)       # [G]
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)   # [G]

        eps  = self.config.clip_epsilon
        beta = self.config.kl_beta
        device = traj_new[0][0]["logits"].device

        sample_losses = []

        # --------------------------------------------------------
        # 2. Iterate over diffusion trajectories (one per sample)
        # --------------------------------------------------------
        for g_idx, adv in enumerate(advantages):
            step_losses = []
            for step_idx in range(len(traj_new[g_idx])):

                # ---- pull tensors for this step / model family
                logits_new = traj_new[g_idx][step_idx]["logits"]    # [L, V]
                logits_old = traj_old[g_idx][step_idx]["logits"]
                logits_ref = traj_ref[g_idx][step_idx]["logits"]

                mask_t   = traj_old[g_idx][step_idx]["mask"].float()      # [L]
                tokens_t = traj_old[g_idx][step_idx]["tokens"]            # [L]
                noise    = traj_old[g_idx][step_idx].get("noise", 1.0)    # scalar

                # ---- convert to log-probs
                logp_new = torch.log_softmax(logits_new, dim=-1)          # [L, V]
                logp_old = torch.log_softmax(logits_old, dim=-1)
                logp_ref = torch.log_softmax(logits_ref, dim=-1)

                # ---- gather log-prob for actually sampled ids
                idx = tokens_t.unsqueeze(-1)                              # [L, 1]
                lp_new_tok = logp_new.gather(-1, idx).squeeze(-1)         # [L]
                lp_old_tok = logp_old.gather(-1, idx).squeeze(-1)
                lp_ref_tok = logp_ref.gather(-1, idx).squeeze(-1)

                # =========================================================
                # 2a.  PPO-style clipped surrogate  (identical math to simple)
                # =========================================================
                ratio        = (lp_new_tok - lp_old_tok).exp()            # [L]
                masked_adv   = adv.to(device) * mask_t                    # [L]

                unclipped    = ratio * masked_adv
                clipped      = torch.clamp(ratio, 1 - eps, 1 + eps) * masked_adv
                surrogate    = torch.minimum(unclipped, clipped).sum()    # scalar

                # =========================================================
                # 2b.  Noise-aware KL penalty (only on masked positions)
                # =========================================================
                kl_step      = ((lp_new_tok - lp_ref_tok) * mask_t).sum()
                step_loss    = -(surrogate - beta * noise * kl_step)      # maximise → minimise
                step_losses.append(step_loss)

            # ---- sum losses for this trajectory
            if step_losses:
                sample_losses.append(torch.stack(step_losses).sum())

        # --------------------------------------------------------
        # 3. Aggregate across the group
        # --------------------------------------------------------
        if sample_losses:
            loss = torch.stack(sample_losses).mean()
        else:
            loss = torch.tensor(0.0, device=device)

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
