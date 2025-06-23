from src.models.llada.model.llada_model import LLaDaModelConfig
from src.models.llada.model.llama_model import LlamaModelConfig
from src.models.llada.model.qwen_model import QwenModelConfig
from src.models.llada.sampler import LLaDaSamplerConfig
from src.models.llada.loss.loss import GRPOLossConfig
from src.data import TUFAIntegralsDatasetConfig
from src.models.llada.trainer import LLaDaTrainerConfig
from src.models.llada.trainer import LLaDaTrainer


def main():
    model_config = QwenModelConfig(
        freeze_embeddings=True,
        freeze_transformer_blocks=None,
    )
    sampler_config = LLaDaSamplerConfig()
    loss_config = GRPOLossConfig(
        math_eval_weight=(-3.0, 5.0),
        math_correctness_weight=(-1.0, 5.0),
        math_format_weight=(-1.0, 1.0),
        length_reward_weight=(-5.0, 1.0),
        anti_gaming_weight=(-5.0, 0.0),
        math_similarity_weight=(-5.0, 10.0),
        similarity_threshold=0.7,
        partial_credit=True,
        anti_gaming_strong_penalty=True,
        kl_beta=0.8,
        use_boxed=True,
        regex_match=None,
    )
    train_dataset_config = TUFAIntegralsDatasetConfig(
        dataset_dir="data/integration/tufa_MIT_variants_cleaned",
        dataset_paths=["data/integration/tufa_llama_variants_cleaned/cleaned_data.json"],
    )
    val_dataset_config = TUFAIntegralsDatasetConfig(
        dataset_dir="data/integration/tufa_MIT_variants_cleaned",
        dataset_paths=["data/integration/tufa_llama_variants_cleaned/cleaned_data_short.json"],
    )
    config = LLaDaTrainerConfig(
        run_name="model_0_qwen",
        num_epochs=20,
        batch_size=1,
        grpo_group_size=8,
        grpo_policy_update_steps=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        validate_epochs=1,
        overwrite_run=True,
        project_dir="experiments",
        model_config=model_config,
        sampler_config=sampler_config,
        loss_config=loss_config,
        train_dataset_config=train_dataset_config,
        val_dataset_config=val_dataset_config,
        backup_epochs=2,
    )
    trainer = LLaDaTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()