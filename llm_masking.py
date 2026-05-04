"""
llm_masking.py

Simplified zero-shot PII masking using an LLM

"""

from pathlib import Path
from types import SimpleNamespace
import json
import re
import time
from typing import List, Dict, Optional
from urllib import request

from ner_utils import load_json


MASK = {"PER": "[PERSON]", "EMAIL": "[EMAIL]"}

EMAIL_RE = re.compile(r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b')

# Simple one-place configuration. Edit values here for a single fixed
# run. To restore CLI behavior, revert to the argparse block below.
CONFIG = {
    "data_dir": "processed_data",
    "model": "llama3.2:1b-instruct",
    "hf_model": "google/flan-t5-small",
    "base_url": "http://localhost:11434",
    "backend": "auto",
    "samples": 200,
    "dry_run": False,
    "output": "phase3_results.json",
    # filename inside data_dir; set to 'test_with_emails.json' to evaluate on injected emails
    "test_file": "test_with_emails.json",
}


def build_prompt(text: str) -> str:
    # Stronger, explicit prompt asking for a strict JSON schema and token spans
    return (
        "Mask person names with [PERSON] and emails with [EMAIL].\n"
        "IMPORTANT: Return ONLY a single valid JSON object with NO extra text or explanation.\n"
        "The JSON must contain two keys:\n"
        "  - masked_text: the input text with PII replaced by [PERSON] or [EMAIL] exactly as tokens.\n"
        "  - entities: a list of objects, each with fields {\"text\": original entity text, \"type\": \"PER|EMAIL\", \"start\": start_token_index, \"end\": end_token_index} where start/end are 0-based token indices in the original tokenization.\n"
        "If you cannot provide token indices, still return entities with the exact original span text.\n"
        "Do NOT include markdown fences or commentary — only the JSON object.\n"
        f"Text: {text}"
    )


def extract_json_block(s: str) -> Optional[Dict]:
    s = s.strip()
    # Try simple JSON parse; if it fails, extract the first {...} block and try again.
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                return None
    return None


def call_ollama(model: str, prompt: str, base_url: str = "http://localhost:11434") -> str:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False}
    data = json.dumps(payload).encode("utf-8")
    url = base_url.rstrip("/") + "/api/chat"
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=120) as r:
        raw = r.read().decode("utf-8")
    parsed = json.loads(raw)
    return parsed.get("message", {}).get("content", "")

#A small helper class to call HF seq2seq models when Ollama is not available


# Removed HFResponder fallback to keep the script focused and simple.


def find_span(tokens: List[str], entity_tokens: List[str], used: List[bool]) -> Optional[int]:
    lower = [t.lower() for t in tokens]
    ent_lower = [t.lower() for t in entity_tokens]
    L = len(ent_lower)
    for i in range(0, len(tokens) - L + 1):
        if any(used[i:i+L]):
            continue
        if lower[i:i+L] == ent_lower:
            return i
    return None


def predicted_entities_to_tags(tokens: List[str], entities: List[Dict]) -> List[str]:
    tags = ["O"] * len(tokens)
    used = [False] * len(tokens)
    for ent in entities:
        text = str(ent.get("text", "")).strip()
        typ = str(ent.get("type", "")).upper()
        if not text or typ not in {"PER", "EMAIL"}:
            continue
        ent_tokens = text.split()
        start = find_span(tokens, ent_tokens, used)
        if start is None:
            continue
        tags[start] = f"B-{typ}"
        used[start] = True
        for j in range(1, len(ent_tokens)):
            tags[start+j] = f"I-{typ}"
            used[start+j] = True
    return tags

#Align masked tokens (may contain [PERSON]/[EMAIL]) back to original token indices


def align_masked_to_original_tags(tokens: List[str], masked_tokens: List[str]) -> List[str]:
    # Very simple alignment: walk original tokens and masked tokens.
    tags = ["O"] * len(tokens)
    i = 0
    j = 0
    while i < len(tokens) and j < len(masked_tokens):
        m = masked_tokens[j]
        if m == MASK['PER'] or m == MASK['EMAIL']:
            typ = 'PER' if m == MASK['PER'] else 'EMAIL'
            # mark current token as beginning of entity
            tags[i] = f'B-{typ}'
            # mark following token as I-* if masked token replaced multiple words is unknown, keep simple
            if i + 1 < len(tokens):
                tags[i+1] = f'I-{typ}'
            i += 1
            j += 1
        else:
            # advance when tokens match (case-insensitive); otherwise advance both conservatively
            if tokens[i].lower() == m.lower():
                i += 1; j += 1
            else:
                i += 1; j += 1
    return tags


def titlecase_name_fallback(tokens: List[str], existing_tags: List[str]) -> List[str]:
    # Heuristic: sequences of Titlecase tokens (not sentence start) that are not all uppercase
    tags = existing_tags.copy()
    i = 0
    stopwords = set(['The', 'A', 'An', 'In', 'On', 'At', 'For', 'And', 'Or', 'But', 'Mr', 'Mrs', 'Ms'])
    # Require at least two consecutive Titlecase tokens to reduce single-token FP
    while i < len(tokens):
        if tags[i] != 'O':
            i += 1
            continue
        tok = tokens[i]
        if tok and tok[0].isupper() and tok not in stopwords and not tok.isupper():
            j = i + 1
            while j < len(tokens) and tokens[j] and tokens[j][0].isupper() and not tokens[j].isupper():
                j += 1
            span_len = j - i
            # mark only multi-token spans (>=2) to reduce FP
            if span_len >= 2:
                tags[i] = 'B-PER'
                for k in range(i+1, j):
                    tags[k] = 'I-PER'
                i = j
            else:
                i += 1
        else:
            i += 1
    return tags


def collapse_binary(tags: List[str], typ: str) -> List[int]:
    return [1 if t == f"B-{typ}" or t == f"I-{typ}" else 0 for t in tags]


def compute_binary_metrics(gold: List[int], pred: List[int]) -> Dict:
    tp = fp = fn = tn = 0
    for g, p in zip(gold, pred):
        if g == 1 and p == 1:
            tp += 1
        elif g == 0 and p == 1:
            fp += 1
        elif g == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _print_entity_table(metrics: Dict):
    # metrics dict contains 'PER' and 'EMAIL'
    rows = []
    for entity_name in ['PER', 'EMAIL']:
        m = metrics[entity_name]
        tp = int(m.get('tp', 0))
        fp = int(m.get('fp', 0))
        fn = int(m.get('fn', 0))
        tn = int(m.get('tn', 0))
        support = tp + fn
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        rows.append({
            'Entity': entity_name,
            'Support': support,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'accuracy': round(m.get('accuracy', 0.0), 6),
            'precision': round(m.get('precision', 0.0), 6),
            'recall': round(m.get('recall', 0.0), 6),
            'f1': round(m.get('f1', 0.0), 6),
            'fpr': round(fpr, 6),
            'fnr': round(fnr, 6),
        })

    # print table
    cols = ['Entity', 'Support', 'tp', 'fp', 'fn', 'tn', 'accuracy', 'precision', 'recall', 'f1', 'fpr', 'fnr']
    widths = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(r[c])))
    header = ' | '.join(c.ljust(widths[c]) for c in cols)
    sep = '-+-'.join('-' * widths[c] for c in cols)
    print('\n' + header)
    print(sep)
    for r in rows:
        print(' | '.join(str(r[c]).ljust(widths[c]) for c in cols))

#Evaluate a set of samples using the chosen backend and return simple binary metrics


def evaluate(samples: List[Dict], backend: str, model_name: str, base_url: str, dry_run: bool, hf_model: Optional[str]):
    # No HFResponder fallback; only Ollama calls are supported in this simplified script.

    gold_per = []
    pred_per = []
    gold_email = []
    pred_email = []
    examples = []

    for i, s in enumerate(samples):
        tokens = s["tokens"]
        gold_tags = s["ner_tags"]
        prompt = build_prompt(" ".join(tokens))

        #Dry-run mode produces an oracle output derived from gold tags (useful for debugging)
        if dry_run:
            # produce perfect oracle JSON
            ents = []
            j = 0
            while j < len(tokens):
                t = gold_tags[j]
                if t.startswith("B-"):
                    typ = t.split("-", 1)[1]
                    chunk = [tokens[j]]
                    j2 = j + 1
                    while j2 < len(tokens) and gold_tags[j2] == f"I-{typ}":
                        chunk.append(tokens[j2]); j2 += 1
                    ents.append({"text": " ".join(chunk), "type": typ})
                    j = j2
                else:
                    j += 1
            raw = json.dumps({"masked_text": " ".join([MASK.get(t.split("-",1)[1], tok) if t.startswith("B-") else tok for tok, t in zip(tokens, gold_tags)]), "entities": ents})
        else:
            try:
                # Call the selected backend (Ollama only in simplified script)
                if backend == "ollama":
                    raw = call_ollama(model_name, prompt, base_url)
                else:
                    raw = "{}"
            except Exception:
                raw = "{}"

        parsed = extract_json_block(raw) or {}
        pred_entities = parsed.get("entities", []) if isinstance(parsed, dict) else []

        # Fallbacks when model doesn't return structured entities
        if pred_entities:
            #If the model returned structured entities, convert to BIO tags
            pred_tags = predicted_entities_to_tags(tokens, pred_entities)
        else:
            # 1) If masked_text is present, align mask tokens to original tokens
            masked_text = parsed.get("masked_text") if isinstance(parsed, dict) else None
            if masked_text:
                mtoks = masked_text.split()
                pred_tags = align_masked_to_original_tags(tokens, mtoks)
                # if still all O, try titlecase heuristic
                if all(t == 'O' for t in pred_tags):
                    pred_tags = titlecase_name_fallback(tokens, pred_tags)
            else:
                # 2) extract emails from raw text using regex as a fallback
                emails_found = EMAIL_RE.findall(raw or '')
                if emails_found:
                    ents = []
                    for e in emails_found:
                        ents.append({'text': e, 'type': 'EMAIL'})
                    pred_tags = predicted_entities_to_tags(tokens, ents)
                else:
                    # last resort: apply a simple Titlecase heuristic to find names
                    pred_tags = titlecase_name_fallback(tokens, ["O"] * len(tokens))

        gold_per.extend(collapse_binary(gold_tags, "PER"))
        pred_per.extend(collapse_binary(pred_tags, "PER"))
        gold_email.extend(collapse_binary(gold_tags, "EMAIL"))
        pred_email.extend(collapse_binary(pred_tags, "EMAIL"))

        if len(examples) < 3:
            examples.append({"input": " ".join(tokens), "gold_masked": parsed.get("masked_text") or " ".join(tokens), "pred_masked": parsed.get("masked_text") or " ".join(tokens), "raw": raw})

    return {"PER": compute_binary_metrics(gold_per, pred_per), "EMAIL": compute_binary_metrics(gold_email, pred_email), "examples": examples}


def main():
    # Use the single-file CONFIG for a simple, reproducible run.
    # Edit the CONFIG dict at the top of this file to change behavior.
    args = SimpleNamespace(**CONFIG)

    data = load_json(Path(args.data_dir) / args.test_file)
    samples = data[: min(args.samples, len(data))]

    backend = args.backend
    if args.dry_run:
        backend = "dry-run"
    elif backend == "auto":
        try:
            call_ollama(args.model, "{\"test\":true}", args.base_url)
            backend = "ollama"
        except Exception:
            backend = "none"

    print("Phase3: LLM masking — samples:", len(samples), "mode:", backend)

    results = evaluate(samples, backend, args.model, args.base_url, args.dry_run, args.hf_model)

    # Print formatted table like ner_training
    _print_entity_table(results)

    out = {"mode": backend, "samples": len(samples), "results": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
