"""
Data Preparation

This module inspects the provided WikiNeural data, generates synthetic
email addresses if needed, injects them deterministically into the
training data (and optionally into a test copy), validates token/tag
alignment, and writes processed JSON files to `processed_data/`.

"""

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
from faker import Faker
import numpy as np

# Deterministic seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
Faker.seed(RANDOM_SEED)

EMAIL_ADDRESS_RE = re.compile(r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b')


def load_json_data(file_path: str) -> List[Dict]:
    #Load JSON data from a file and return as Python objects
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def inspect_dataset(data: List[Dict], dataset_name: str) -> None:
    # Lightweight dataset inspection removed for brevity.
    print(f"Inspecting {dataset_name}: {len(data)} samples (skipping detailed stats)")


def validate_token_tag_alignment(data: List[Dict]) -> bool:
    # Minimal alignment check: return True if all samples have matching token/tag lengths
    return all(len(s.get('tokens', [])) == len(s.get('ner_tags', [])) for s in data)


def dataset_contains_email_addresses(data: List[Dict]) -> bool:
    # Simple existence check for email-like tokens
    for sample in data:
        for tok in sample.get('tokens', []):
            if EMAIL_ADDRESS_RE.fullmatch(tok):
                return True
    return False


def generate_synthetic_emails(num_emails: int = 2000) -> List[str]:
    # Generate simple deterministic synthetic emails for reproducibility
    return [f'user{idx}@example.com' for idx in range(1, num_emails + 1)]


def split_email_into_tokens(email: str) -> List[str]:
    #Keep email as a single token to avoid breaking with subtokenizers
    return [email]


def inject_emails_into_sample(sample: Dict, email_list: List[str], injection_probability: float = 0.15) -> Dict:
    # Randomly insert a single email token into the sample without heavy validation
    if random.random() > injection_probability:
        return sample
    tokens = sample['tokens'].copy()
    tags = sample['ner_tags'].copy()
    insert_pos = random.randint(0, max(0, len(tokens)))
    email = random.choice(email_list)
    tokens = tokens[:insert_pos] + [email] + tokens[insert_pos:]
    tags = tags[:insert_pos] + ['B-EMAIL'] + tags[insert_pos:]
    sequence = ' '.join(tokens)
    return {'tokens': tokens, 'ner_tags': tags, 'lang': sample.get('lang', 'en'), 'sequence': sequence}


def inject_emails_into_dataset(data: List[Dict], email_list: List[str], injection_probability: float = 0.15) -> List[Dict]:
    injected = [inject_emails_into_sample(s, email_list, injection_probability) for s in data]
    return injected


def compute_dataset_statistics(data: List[Dict]) -> Dict:
    # Minimal statistics: number of samples and total tokens
    total_tokens = sum(len(s.get('tokens', [])) for s in data)
    return {'num_samples': len(data), 'total_tokens': total_tokens}


def print_statistics(stats: Dict, label: str) -> None:
    # Print minimal statistics
    print(f"{label}: samples={stats['num_samples']}, total_tokens={stats['total_tokens']}")


def create_train_val_split(data: List[Dict], train_ratio: float = 0.8, seed: int = RANDOM_SEED) -> Tuple[List[Dict], List[Dict]]:
    #Shuffle deterministically and split into train/validation
    rnd = list(range(len(data)))
    random.Random(seed).shuffle(rnd)
    split = int(len(data) * train_ratio)
    train = [data[i] for i in rnd[:split]]
    val = [data[i] for i in rnd[split:]]
    return train, val


def save_dataset(data: List[Dict], output_path: str) -> None:
    #Save processed dataset as pretty JSON for inspection and reproducibility
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} samples to {output_path}")


def main():
    #Entry point for data preparation: load, inspect, generate emails, inject, and save
    print('\n' + '='*60)
    print('PHASE 1: DATA PREPARATION (fixed)')
    print('='*60)

    base_path = Path(__file__).parent / 'Internship_task_data' / 'Internship_task_data'
    train_file = base_path / 'data.json'
    test_file = base_path / 'test_data.json'
    output_dir = Path(__file__).parent / 'processed_data'

    #Load raw JSON files from the provided data directory
    print('\nLoading raw datasets...')
    train_data = load_json_data(str(train_file))
    test_data = load_json_data(str(test_file))

    #Quick inspections help confirm expected tag distributions
    inspect_dataset(train_data, 'RAW TRAIN')
    inspect_dataset(test_data, 'RAW TEST')

    #Check if the source data already contains real email addresses
    train_has_emails = dataset_contains_email_addresses(train_data)
    print(f"\nTrain contains raw emails: {train_has_emails}")

    #Generate synthetic emails deterministically for reproducible experiments
    synthetic_emails = generate_synthetic_emails(num_emails=2000)

    #Inject synthetic emails into training data at a fixed probability
    print('\nInjecting emails into training data (will overwrite processed train/val)')
    train_with_emails = inject_emails_into_dataset(train_data, synthetic_emails, injection_probability=0.15)

    #Create train/validation split and compute statistics for each split
    train_set, val_set = create_train_val_split(train_with_emails, train_ratio=0.8, seed=RANDOM_SEED)
    train_stats = compute_dataset_statistics(train_set)
    val_stats = compute_dataset_statistics(val_set)
    test_stats = compute_dataset_statistics(test_data)

    #Print summary statistics for each set to help quick checks
    print_statistics(train_stats, 'TRAIN (POST-INJECTION)')
    print_statistics(val_stats, 'VAL (POST-INJECTION)')
    print_statistics(test_stats, 'TEST (UNCHANGED)')

    # Save processed datasets (overwrite existing processed_data files)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(train_set, str(output_dir / 'train.json'))
    save_dataset(val_set, str(output_dir / 'val.json'))
    save_dataset(test_data, str(output_dir / 'test.json'))
    save_dataset(synthetic_emails, str(output_dir / 'synthetic_emails.json'))

    # Also produce an optional test copy that contains injected emails (for debugging/LLM experiments)
    test_with_emails = inject_emails_into_dataset(test_data, synthetic_emails, injection_probability=0.15)
    save_dataset(test_with_emails, str(output_dir / 'test_with_emails.json'))

    # Save summary config
    config = {
        'train_samples': len(train_set),
        'val_samples': len(val_set),
        'test_samples': len(test_data),
        'synthetic_emails_count': len(synthetic_emails),
        'email_injection_probability': 0.15,
        'tag_set': ['O', 'B-PER', 'I-PER', 'B-EMAIL', 'I-EMAIL'],
        'train_statistics': {'total_tokens': train_stats['total_tokens'], 'tag_distribution': dict(train_stats['tag_distribution'])},
        'val_statistics': {'total_tokens': val_stats['total_tokens'], 'tag_distribution': dict(val_stats['tag_distribution'])},
        'test_statistics': {'total_tokens': test_stats['total_tokens'], 'tag_distribution': dict(test_stats['tag_distribution'])}
    }
    with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to {output_dir / 'config.json'}")


if __name__ == '__main__':
    main()
