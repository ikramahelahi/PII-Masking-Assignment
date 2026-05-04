# PII-Masking-Assignment

This repository contains an internship project focused on detecting and masking Personally Identifiable Information (PII) in text, specifically person names (PER) and email addresses (EMAIL).

---

## 📌 Overview

Two approaches are implemented and compared:

- A fine-tuned transformer model (BIO-based token classification)
- A zero-shot LLM (prompt-based extraction)

The goal is to evaluate performance trade-offs between supervised learning and prompt-based methods for PII detection.

---

## 📊 Dataset

The dataset is based on the WikiNeural NER dataset. Since email entities are limited in the original data, synthetic emails were added using realistic name combinations and common email domains (e.g., gmail.com, yahoo.com).

Two evaluation setups were used:
- Synthetic test set (augmented data)
- Independent real-world dataset for final evaluation

---

## 🧠 Models

**Fine-tuned Transformer**
- DeBERTa-v3-small
- Token classification using BIO tagging
- Trained with weighted loss to handle class imbalance

**Zero-shot LLM**
- LLaMA 3.2-1B
- Prompt-based entity extraction
- No fine-tuning performed

---

## 📈 Results Summary

- The fine-tuned model achieves strong and consistent performance across both entity types, especially for emails.
- The LLM shows high precision in some cases but lower recall, often missing entities due to prompt limitations.

---

## ⚖️ Key Insight

The supervised transformer is more reliable for production-level PII masking, while the LLM is better suited for quick prototyping and flexible extraction tasks.

---

## 🔗 Repository

https://github.com/ikramahelahi/PII-Masking-Assignment

---

## 👤 Author

Ikramah Elahi  
Internship Project — PII Masking System
