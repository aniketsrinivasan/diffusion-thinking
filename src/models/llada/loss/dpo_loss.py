import torch
from dataclasses import dataclass

from src.models.llada.loss.loss_metrics import MathCorrectnessMetric, MathFormatMetric, MathEvalMetric
from src.models.llada.loss.loss_utils import LossUtils


@dataclass 
class DPOLossConfig:
    pass


class DPOLoss:
    """
    Loss function for DPO.
    """

    def __init__(self, config: DPOLossConfig):
        self.config = config

    def get_simple_loss():
        pass