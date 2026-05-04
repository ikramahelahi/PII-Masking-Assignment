"""
Utility helpers for NER training: loading data, label mapping, token-label alignment.

"""
import json
import re
from typing import List, Dict, Tuple, Iterable


def load_json(path: str) -> List[Dict]:
    #Load JSON file and return Python objects
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


EMAIL_ADDRESS_RE = re.compile(r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b')

# Note: lightweight email regex is available above if needed elsewhere.


def get_label_list(data: List[Dict]) -> List[str]:
    labels = set()
    for sample in data:
        labels.update(sample.get('ner_tags', []))
    # Ensure common tags present and consistent order
    desired = ['O', 'B-PER', 'I-PER', 'B-EMAIL', 'I-EMAIL']
    present = [l for l in desired if l in labels]
    # Add any other labels (unlikely) at the end
    others = sorted([l for l in labels if l not in present])
    #Return preferred ordering of common labels followed by any extras
    return present + others


def align_labels_with_tokens(tokenizer, tokens: List[str], tags: List[str], label2id: Dict[str, int], max_length: int = 128) -> Dict:

    # Use tokenizer to split into word pieces
    #Tokenize while preserving word boundaries so we can align labels
    encoded = tokenizer(tokens,
                        is_split_into_words=True,
                        truncation=True,
                        padding='max_length',
                        max_length=max_length,
                        return_tensors=None)

    word_ids = encoded.word_ids()
    aligned_labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            # Start of a new word
            aligned_labels.append(label2id.get(tags[word_idx], label2id.get('O', 0)))
        else:
            # Subsequent token of a word
            aligned_labels.append(-100)
        previous_word_idx = word_idx

    encoded['labels'] = aligned_labels
    return encoded


def _flatten_binary_labels(true_labels: List[List[str]], pred_labels: List[List[str]], entity_types: Iterable[str]):
    entity_set = tuple(entity_types)
    gold = []
    pred = []
    for true_seq, pred_seq in zip(true_labels, pred_labels):
        for true_tag, pred_tag in zip(true_seq, pred_seq):
            gold.append(1 if any(true_tag == f'B-{entity}' or true_tag == f'I-{entity}' for entity in entity_set) else 0)
            pred.append(1 if any(pred_tag == f'B-{entity}' or pred_tag == f'I-{entity}' for entity in entity_set) else 0)
    #Flatten sequences to binary arrays where 1 means token is part of the entity set
    return gold, pred


def compute_binary_metrics_from_arrays(gold: List[int], pred: List[int]) -> Dict:
    tp = fp = fn = tn = 0
    for gold_value, pred_value in zip(gold, pred):
        if gold_value == 1 and pred_value == 1:
            tp += 1
        elif gold_value == 0 and pred_value == 1:
            fp += 1
        elif gold_value == 1 and pred_value == 0:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    support = tp + fn
    #Return standard binary classification metrics computed from counts
    return {
        'support': support,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'fnr': fnr,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
    }


def compute_entity_metrics(true_labels: List[List[str]], pred_labels: List[List[str]], entity_type: str) -> Dict:
    #Compute metrics for a single entity type (e.g., PER or EMAIL)
    gold, pred = _flatten_binary_labels(true_labels, pred_labels, (entity_type,))
    metrics = compute_binary_metrics_from_arrays(gold, pred)
    metrics['entity'] = entity_type
    return metrics


def compute_overall_pii_metrics(true_labels: List[List[str]], pred_labels: List[List[str]], entity_types: Iterable[str] = ('PER', 'EMAIL')) -> Dict:
    #Compute combined PII metrics treating PER and EMAIL together
    gold, pred = _flatten_binary_labels(true_labels, pred_labels, entity_types)
    metrics = compute_binary_metrics_from_arrays(gold, pred)
    metrics['entity'] = 'PII'
    return metrics


class SimpleNERDataset:
    """A small dataset wrapper that returns tokenized inputs and labels as tensors when requested by a DataLoader."""
    def __init__(self, encodings: List[Dict]):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        #Return the raw encoding dict for a single example
        item = self.encodings[idx]
        return {k: v for k, v in item.items()}
