from src.models.llada.model.llama_model import LlamaModel, LlamaModelConfig
from src.models.llada.model.llada_model import LLaDaModel, LLaDaModelConfig
from src.models.llada.model.qwen_model import QwenModel, QwenModelConfig
from src.models.llada.model.base_model import BaseModel, BaseModelConfig


def get_model(model_config: BaseModelConfig) -> BaseModel:
    if isinstance(model_config, LlamaModelConfig):
        return LlamaModel(model_config)
    elif isinstance(model_config, LLaDaModelConfig):
        return LLaDaModel(model_config)
    elif isinstance(model_config, QwenModelConfig):
        return QwenModel(model_config)
    else:
        raise ValueError(f"Model {model_config.model_name} not supported")