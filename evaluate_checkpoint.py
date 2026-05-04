import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from ner_utils import (
    load_json,
    get_label_list,
    align_labels_with_tokens,
    SimpleNERDataset,
    compute_entity_metrics,
    compute_overall_pii_metrics,
)


def collate_batch(batch):
    #Collate a batch into tensors for evaluation
    input_ids = torch.tensor([b['input_ids'] for b in batch], dtype=torch.long)
    attention_mask = torch.tensor([b['attention_mask'] for b in batch], dtype=torch.long)
    labels = torch.tensor([b['labels'] for b in batch], dtype=torch.long)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def decode_predictions(preds, labels, id2label):
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
    #Convert numeric label ids back to human-readable tags
    return pred_labels, true_labels


def evaluate(model, data_loader, device, id2label):
    #Run model on the dataset and accumulate predictions and loss
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            batches += 1
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            labels_np = labels.cpu().numpy().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels_np)

    pred_labels, true_labels = decode_predictions(all_preds, all_labels, id2label)
    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels)
    #Compute entity-level binary metrics for PER and EMAIL
    per_metrics = compute_entity_metrics(true_labels, pred_labels, 'PER')
    email_metrics = compute_entity_metrics(true_labels, pred_labels, 'EMAIL')
    overall_metrics = compute_overall_pii_metrics(true_labels, pred_labels)
    return {
        'loss': total_loss / max(batches, 1),
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'report': report,
        'per_entity': {
            'PER': per_metrics,
            'EMAIL': email_metrics,
            'PII': overall_metrics,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='processed_data')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)

    #Load train data to build label list (keep ordering consistent with training)
    train = load_json(str(data_dir / 'train.json'))
    labels = get_label_list(train)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    #Show which labels are being used when decoding predictions
    print('\nLabels used for evaluation:', labels)

    #Load tokenizer and model from the saved checkpoint directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    test = load_json(str(data_dir / 'test.json'))

    # Prepare encodings
    print('Preparing encodings for test set...')
    encodings = []
    for sample in test:
        tokens = sample['tokens']
        tags = sample['ner_tags']
        enc = align_labels_with_tokens(tokenizer, tokens, tags, label2id, max_length=128)
        encodings.append({'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'], 'labels': enc['labels']})

    dataset = SimpleNERDataset(encodings)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print('Running evaluation on test set...')
    metrics = evaluate(model, loader, device, id2label)

    print(f"\nTest loss: {metrics['loss']:.4f} | Test F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
    print('Test per-entity metrics (PER, EMAIL, PII):')
    print('Entity | Support | accuracy | precision | recall | f1 | fpr | fnr')
    for entity_name in ['PER', 'EMAIL', 'PII']:
        m = metrics['per_entity'][entity_name]
        print(
            f"{entity_name} | {m['support']} | {m['accuracy']:.6f} | {m['precision']:.6f} | "
            f"{m['recall']:.6f} | {m['f1']:.6f} | {m['fpr']:.6f} | {m['fnr']:.6f}"
        )
    print('\nDetailed classification report:\n')
    print(metrics['report'])


if __name__ == '__main__':
    main()
