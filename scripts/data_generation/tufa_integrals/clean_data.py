import os
import json
from typing import Optional


def clean_data(dataset_dir: Optional[str] = None, input_path: Optional[str] = None, output_dir: str = "data/integration/tufa_llama_variants_cleaned"):
    """
    Clean the data from the dataset directory. Save results to the output directory. 
    """
    cleaned_data = []

    assert sum([dataset_dir is not None, input_path is not None]) == 1, "Only one of dataset_dir or input_path should be provided"

    if dataset_dir is not None:
        all_data = []
        for file in os.listdir(dataset_dir):
            with open(os.path.join(dataset_dir, file), 'r') as f:
                data = json.load(f)
            all_data.extend(data)
    else:
        with open(input_path, 'r') as f:
            all_data = json.load(f)

    # only add items that passed verification
    for item in all_data:
        if item['verification_passed'] and item['solution'] is not None and item["requested_difficulty"] == "easier":
            cleaned_data.append(item)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'cleaned_data_short_easier.json'), 'w') as f:
        # dump in a pretty format:
        json.dump(cleaned_data, f, indent=4)


def main():
    clean_data(input_path='data/integration/tufa_llama_variants_cleaned/cleaned_data.json', output_dir='data/integration/tufa_llama_variants_cleaned')


if __name__ == '__main__':
    main()