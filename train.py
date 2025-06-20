from src.models.llada.model.llada_model import LLaDaModelConfig
from src.models.llada.model.llama_model import LlamaModelConfig
from src.models.llada.sampler import LLaDaSamplerConfig
from src.models.llada.loss.loss import GRPOLossConfig
from src.data import TUFAIntegralsDatasetConfig
from src.models.llada.trainer import LLaDaTrainerConfig
from src.models.llada.trainer import LLaDaTrainer


def main():
    model_config = LLaDaModelConfig(
        freeze_embeddings=True,
        freeze_transformer_blocks=list(range(0, 12)) + list(range(24, 31)),
    )
    sampler_config = LLaDaSamplerConfig()
    loss_config = GRPOLossConfig(
        use_math_eval=True
    )
    train_dataset_config = TUFAIntegralsDatasetConfig(
        dataset_dir="data/integration/tufa_MIT_variants_cleaned",
        dataset_paths=["data/integration/tufa_llama_variants_cleaned/cleaned_data_half.json"],
    )
    val_dataset_config = TUFAIntegralsDatasetConfig(
        dataset_dir="data/integration/tufa_MIT_variants_cleaned",
        dataset_paths=["data/integration/tufa_llama_variants_cleaned/cleaned_data_short.json"],
    )
    config = LLaDaTrainerConfig(
        run_name="model_1_llada",
        num_epochs=20,
        batch_size=1,
        grpo_group_size=8,
        gradient_accumulation_steps=4,
        validate_epochs=1,
        overwrite_run=False,
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