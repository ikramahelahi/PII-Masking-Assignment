"""
NER training script

"""
import argparse
import random
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from ner_utils import (
    SimpleNERDataset,
    align_labels_with_tokens,
    compute_entity_metrics,
    compute_overall_pii_metrics,
    get_label_list,
    load_json,
)


def set_seed(seed: int = 42):
    #Set random seeds so experiments are reproducible across runs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_batch(batch: List[Dict]):
    #Combine a list of samples into batched tensors for the model
    input_ids = torch.tensor([b['input_ids'] for b in batch], dtype=torch.long)
    attention_mask = torch.tensor([b['attention_mask'] for b in batch], dtype=torch.long)
    labels = torch.tensor([b['labels'] for b in batch], dtype=torch.long)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def prepare_encodings(data: List[Dict], tokenizer, label2id: Dict[str, int], max_length: int, max_samples: int = None):
    #Tokenize each sample and align BIO tags to token pieces
    encodings = []
    subset = data if max_samples is None else data[:max_samples]
    for sample in subset:
        tokens = sample['tokens']
        tags = sample['ner_tags']
        enc = align_labels_with_tokens(tokenizer, tokens, tags, label2id, max_length=max_length)
        encodings.append({
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels': enc['labels'],
        })
    return encodings


def decode_predictions(preds: List[List[int]], labels: List[List[int]], id2label: Dict[int, str]):
    true_labels = []
    pred_labels = []
    for pred_seq, label_seq in zip(preds, labels):
        t, p = [], []
        for p_id, l_id in zip(pred_seq, label_seq):
            if l_id == -100:
                continue
            t.append(id2label.get(int(l_id), 'O'))
            p.append(id2label.get(int(p_id), 'O'))
        true_labels.append(t)
        pred_labels.append(p)
    #Return human-readable label sequences for evaluation
    return pred_labels, true_labels


def evaluate(model, data_loader, device, id2label):
    #Run the model in evaluation mode and collect predictions for metrics
    model.eval()
    all_preds = []
    all_labels = []
    total_val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()
            val_batches += 1

            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=-1)

            preds = pred_ids.cpu().numpy().tolist()
            labels_np = labels.cpu().numpy().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels_np)

    pred_labels, true_labels = decode_predictions(all_preds, all_labels, id2label)
    seqeval_f1 = f1_score(true_labels, pred_labels)
    seqeval_precision = precision_score(true_labels, pred_labels)
    seqeval_recall = recall_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels)
    val_loss = total_val_loss / max(val_batches, 1)

    #Compute per-entity binary metrics for reporting and CSV output
    per_metrics = compute_entity_metrics(true_labels, pred_labels, 'PER')
    email_metrics = compute_entity_metrics(true_labels, pred_labels, 'EMAIL')
    overall_metrics = compute_overall_pii_metrics(true_labels, pred_labels)

    return {
        'val_loss': val_loss,
        'seqeval_f1': seqeval_f1,
        'seqeval_precision': seqeval_precision,
        'seqeval_recall': seqeval_recall,
        'report': report,
        'per_entity': {
            'PER': per_metrics,
            'EMAIL': email_metrics,
            'PII': overall_metrics,
        },
    }


def _metrics_row_from_eval(epoch: int, train_loss: float, metrics: Dict) -> Dict:
    per_metrics = metrics['per_entity']['PER']
    email_metrics = metrics['per_entity']['EMAIL']
    overall_metrics = metrics['per_entity']['PII']
    #Format a row suitable for CSV/printing for the given epoch
    return {
        'Epoch': epoch,
        'Training Loss': round(train_loss, 6),
        'Validation Loss': round(metrics['val_loss'], 6),
        'Per Precision': round(per_metrics['precision'], 6),
        'Per Recall': round(per_metrics['recall'], 6),
        'Per F1': round(per_metrics['f1'], 6),
        'Per Fpr': round(per_metrics['fpr'], 6),
        'Per Fnr': round(per_metrics['fnr'], 6),
        'Email Precision': round(email_metrics['precision'], 6),
        'Email Recall': round(email_metrics['recall'], 6),
        'Email F1': round(email_metrics['f1'], 6),
        'Email Fpr': round(email_metrics['fpr'], 6),
        'Email Fnr': round(email_metrics['fnr'], 6),
        'Overall Precision': round(overall_metrics['precision'], 6),
        'Overall Recall': round(overall_metrics['recall'], 6),
        'Overall F1': round(overall_metrics['f1'], 6),
        'Accuracy': round(overall_metrics['accuracy'], 6),
    }


def _print_table(rows: List[Dict], columns: List[str]) -> None:
    #Print a simple aligned table to stdout for quick inspection
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(str(row[column])))

    header = ' | '.join(column.ljust(widths[column]) for column in columns)
    separator = '-+-'.join('-' * widths[column] for column in columns)
    print(header)
    print(separator)
    for row in rows:
        print(' | '.join(str(row[column]).ljust(widths[column]) for column in columns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='processed_data', help='Directory with train/val/test json files')
    parser.add_argument('--model', type=str, default='distilbert-base-uncased', help='Hugging Face model name')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs (dry-run default=1)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--dry-run', action='store_true', help='Quick run on a small subset')
    parser.add_argument('--max-samples', type=int, default=200, help='Max samples to use in dry-run')
    parser.add_argument('--test-file', type=str, default='test.json', help='Test file name inside data-dir (e.g. test.json or test_with_emails.json)')
    args = parser.parse_args()

    #Prepare deterministic randomness for reproducible runs
    set_seed()

    data_dir = Path(args.data_dir)
    train_path = data_dir / 'train.json'
    val_path = data_dir / 'val.json'
    test_path = data_dir / args.test_file

    train_data = load_json(train_path)
    val_data = load_json(val_path)
    test_data = load_json(test_path)

    max_train = args.max_samples if args.dry_run else None
    max_val = int(args.max_samples / 4) if args.dry_run else None
    max_test = int(args.max_samples / 4) if args.dry_run else None

    label_list = get_label_list(train_data)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    #Show label ordering used for token classification
    print('\nLabels:', label_list)

    #Load tokenizer from the chosen model name
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    #Create encodings aligned to labels; this is the main preprocessing step
    print('\nPreparing tokenized encodings (this may take a moment)...')
    train_enc = prepare_encodings(train_data, tokenizer, label2id, args.max_length, max_samples=max_train)
    val_enc = prepare_encodings(val_data, tokenizer, label2id, args.max_length, max_samples=max_val)
    test_enc = prepare_encodings(test_data, tokenizer, label2id, args.max_length, max_samples=max_test)

    train_dataset = SimpleNERDataset(train_enc)
    val_dataset = SimpleNERDataset(val_enc)
    test_dataset = SimpleNERDataset(test_enc)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    #Choose device (GPU when available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    #Initialize model for token classification with the correct number of labels
    model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=len(label_list))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    out_dir = Path('models') / f"{args.model.replace('/', '_')}_finetuned_small"
    out_dir.mkdir(parents=True, exist_ok=True)

    print('\nStarting training...')
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(len(train_loader), 1)
        print(f'Epoch {epoch + 1} completed. Avg training loss: {avg_train_loss:.4f}')

        # Evaluate on validation set (summary only)
        metrics = evaluate(model, val_loader, device, id2label)
        print(f"Validation F1: {metrics['seqeval_f1']:.4f} | Accuracy: {metrics['per_entity']['PII']['accuracy']:.4f}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f'\nSaved model to {out_dir}')

    print('\nEvaluating on test set...')
    test_metrics = evaluate(model, test_loader, device, id2label)
    print(f"Test F1: {test_metrics['seqeval_f1']:.4f} | Accuracy: {test_metrics['per_entity']['PII']['accuracy']:.4f}")


if __name__ == '__main__':
    main()
