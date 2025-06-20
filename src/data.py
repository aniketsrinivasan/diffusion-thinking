from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from omegaconf import MISSING
import json
import torch

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def custom_collate_fn(batch):
    """
    Custom collate function that handles None values by filtering them out
    and converting strings to appropriate format.
    """
    # Filter out any None entries
    filtered_batch = []
    for item in batch:
        if item is not None and isinstance(item, dict):
            # Check if required fields are not None
            if item.get("variant") is not None:
                filtered_batch.append(item)
    
    if not filtered_batch:
        # Return a dummy batch if all items were filtered out
        return [{"variant": "x", "original": "x", "solution": "x"}]
    
    return filtered_batch


@dataclass
class TUFAIntegralsDatasetConfig:
    dataset_dir: str = MISSING
    dataset_paths: list[str] = field(default_factory=list)
    nested_lists: bool = False


class TUFAIntegralsDataset(Dataset):
    """
    Load the TufaLabs MIT integrals dataset, as defined in https://github.com/Tufalabs/LADDER.
    See https://arxiv.org/pdf/2503.00735 for more details.
    """

    def __init__(self, config: TUFAIntegralsDatasetConfig):
        self.config = config
        self.dataset: list[dict] = []
        self._load_dataset()
        self._filter_dataset()

    def _load_dataset(self):
        for dataset_path in self.config.dataset_paths:
            with open(dataset_path, "r") as f:
                data = json.load(f)
        if not self.config.nested_lists:
            self.dataset += data
        else:
            for d_item in data:
                self.dataset += d_item

        logger.info(f"Loaded {len(self.dataset)} examples from {self.config.dataset_paths}")

    def _filter_dataset(self):
        """Filter out entries with None values in critical fields."""
        original_size = len(self.dataset)
        self.dataset = [
            item for item in self.dataset 
            if item is not None 
            and isinstance(item, dict)
            and item.get("variant") is not None
            and item.get("variant") != ""
        ]
        filtered_size = len(self.dataset)
        if filtered_size < original_size:
            logger.warning(f"Filtered out {original_size - filtered_size} entries with None/empty values")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns a dictionary with the following keys:
        - original: the original integral expression
        - requested_difficulty: the difficulty level requested by the user
        - variant: the variant of the integral expression
        - reasoning: the reasoning used to generate the variant
        - variant_response: the response from the model to the variant
        - verification_passed: whether the variant was verified to be correct
        - transformations_used: the transformations used to generate the integral variant
        - timestamp: the timestamp of the variant generation
        """
        item = self.dataset[idx]
        
        # Ensure all fields have default values if None
        item = {
            "original": item.get("original", ""),
            "requested_difficulty": item.get("requested_difficulty", "medium"),
            "variant": item.get("variant", ""),
            "reasoning": item.get("reasoning", ""),
            "variant_response": item.get("variant_response", ""),
            "verification_passed": item.get("verification_passed", False),
            "transformations_used": item.get("transformations_used", []),
            "timestamp": item.get("timestamp", ""),
            "solution": item.get("solution", "")  # Add solution field if exists
        }
        
        return item
    

def main():
    config = TUFAIntegralsDatasetConfig(
        dataset_dir="data/integration/tufa_llama_variants_cleaned",
        dataset_paths=["data/integration/tufa_llama_variants_cleaned/cleaned_data.json"],
    )
    dataset = TUFAIntegralsDataset(config)
    print(dataset[0])


if __name__ == "__main__":
    main()
